from transformers import TFViTModel
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Lambda, LeakyReLU
from tensorflow.keras import mixed_precision
import numpy as np
from loss import modified_accuracy, modified_categorical_crossentropy, modified_mse, modified_mae

class CustomSaveCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom callback to save the model weights every n epochs.
    
    Inherits from:
        tf.keras.callbacks.Callback.

    Args:
        filepath (str): The path to save the model weights.
        save_freq (int): The frequency to save the model weights.
        **kwargs: Additional keyword arguments.
        
    Attributes:
        epochs_since_last_save (int): The number of epochs since the last save.
        checkpoint_callback (ModelCheckpoint): The ModelCheckpoint callback.
    """
    def __init__(self, filepath, save_freq=5, **kwargs):
        super(CustomSaveCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.epochs_since_last_save = 0
        self.checkpoint_callback = ModelCheckpoint(filepath=self.filepath,
                                                   save_weights_only=True,
                                                   save_best_only=True,
                                                   monitor='val_loss',
                                                   mode='min',
                                                   **kwargs)

    def set_model(self, model):
        """
        Sets the model for the callback.
        
        Args:
            model (Model): The model to set.
        """
        super(CustomSaveCheckpoint, self).set_model(model)
        self.checkpoint_callback.set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function to save the model weights every n epochs.

        Args:
            epoch (int): The current epoch.
            logs (dict): The logs for the current epoch.
        """
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.save_freq:
            self.epochs_since_last_save = 0
            self.checkpoint_callback.on_epoch_end(epoch, logs)

class Model(tf.keras.Model):
    """
    A class to represent the model.

    Inherits from:
        tf.keras.Model.
    
    Args:
        base (TFViTModel): The base model.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, base, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.base.trainable = False
        
        self.conv_32 = Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu')
        self.conv_64 = Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu')
        self.maxpool_2 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()

        self.dense_512 = Dense(512, activation=LeakyReLU(negative_slope=0.01), 
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense_256 = Dense(256, activation=LeakyReLU(negative_slope=0.01), 
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense_128 = Dense(128, activation=LeakyReLU(negative_slope=0.01),
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))

        self.dense_64 = Dense(64, activation=LeakyReLU(negative_slope=0.01),
                              kernel_initializer='glorot_uniform',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))
        
        self.dropout = Dropout(0.2)
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        
        self.output_dense_class = Dense(225*13, activation=LeakyReLU(negative_slope=0.01), 
                                         kernel_initializer='glorot_uniform',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.output_reshape_class1 = Reshape((225, 13))
        self.softmax_single_class = Dense(13, activation='softmax')
        self.output_reshape_class2 = Reshape((225, 13), name='class_output')
        
        self.output_dense_reg = Dense(225*4, activation=LeakyReLU(negative_slope=0.01), 
                                      kernel_initializer='glorot_uniform',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.output_reshape_reg1 = Reshape((225, 4))
        self.sigmoid_single_reg = Dense(4, activation='sigmoid')
        self.output_reshape_reg2 = Reshape((225, 4), name='reg_output')
        
    def call(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            dict: The class and regression outputs.
        """
        # Extract features from the base model
        x = self.base(inputs)
        x = tf.expand_dims(x['last_hidden_state'], axis=-1)
        x = self.conv_32(x)
        x = self.maxpool_2(x)
        x = self.conv_64(x)
        x = self.maxpool_2(x)
        x = self.flatten(x)
        
        # Class output
        y = self.dense_512(x)
        y = self.dropout(y)
        y = self.dense_256(y)
        y = self.dropout(y)
        y = self.output_dense_class(y)
        y = self.output_reshape_class1(y)
        softmax_layers = []
        for i in range(225):
            row = Lambda(lambda x: x[:, i, :])(y)
            softmax = self.softmax_single_class(row)
            softmax_layers.append(softmax)
        
        y = self.concatenate(softmax_layers)
        y = self.output_reshape_class2(y)
            
        # Regression output
        z = self.dense_256(x)
        z = self.dropout(z)
        z = self.dense_128(z)
        z = self.dropout(z)
        z = self.dense_128(z)
        z = self.dense_64(z)
        z = self.output_dense_reg(z)
        z = self.output_reshape_reg1(z)
        reg_layers = []
        for i in range(225):
            row = Lambda(lambda x: x[:, i, :])(z)
            out = self.sigmoid_single_reg(row)
            reg_layers.append(out)
        z = self.concatenate(reg_layers)
        z = self.output_reshape_reg2(z)
        return {'class_output': y, 'reg_output': z}
    
class ViTModel:
    """
    A class to represent the ViT model.

    Args:
        namespace_config (Namespace): The namespace configuration.
        X_train (ndarray): The training feature data.
        cls_train (ndarray): The training class labels.
        offset_train (ndarray): The training offset labels.
        X_val (ndarray): The validation feature data.
        cls_val (ndarray): The validation class labels.
        offset_val (ndarray): The validation offset labels.
        X_test (ndarray): The testing feature data.
        cls_test (ndarray): The testing class labels.
        offset_test (ndarray): The testing offset labels.
    """
    def __init__(self, namespace_config,
                 X_train, cls_train, offset_train, 
                 X_val, cls_val, offset_val,
                 X_test, cls_test, offset_test):
        self.X_train = X_train
        self.cls_train = cls_train
        self.offset_train = offset_train
        self.X_val = X_val
        self.cls_val = cls_val
        self.offset_val = offset_val
        self.X_test = X_test
        self.cls_test = cls_test
        self.offset_test = offset_test

        self.summary = namespace_config.training.summary
        self.weights = namespace_config.training.weights
        self.optimizer = namespace_config.training.optimizer
        self.class_loss_weight = namespace_config.training.class_loss_weight
        self.reg_loss_weight = namespace_config.training.reg_loss_weight
        self.run_eagerly = namespace_config.training.run_eagerly
        
        self.batch_size = namespace_config.training.batch_size
        self.epochs = namespace_config.training.epochs
        
        # Load the base model
        self.extract = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.callbacks= [CustomSaveCheckpoint(filepath='weights_epoch_{epoch:02d}.weights.h5', save_freq=5, verbose=1)]

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        self.model = Model(self.extract)
        self.dummy_input()                 # To build the model
        self.print_summary()
        self.load_weights(self.weights)

    def dummy_input(self):
        """
        Builds the model with a dummy input.
        """
        self.model(np.array([self.X_train[0]]))

    def print_summary(self):
        """
        Prints the model summary.
        """
        if self.summary:
            self.model.summary()

    def load_weights(self, weights):
        """
        Loads the model weights.

        Args:
            weights (str): The path to the model weights.
        """
        if self.weights:
            self.model.load_weights(weights)

    def compile(self):
        """
        Compiles the model.
        """
        self.model.compile(optimizer=self.optimizer,
              loss={'class_output': modified_categorical_crossentropy,
                    'reg_output': modified_mse},
              loss_weights={'class_output': self.class_loss_weight, 'reg_output': self.reg_loss_weight},
              metrics={'class_output': modified_accuracy,
                       'reg_output': modified_mae},
              run_eagerly=self.run_eagerly)
        
    def train(self):
        """
        Trains the model.
        """
        self.compile()
        self.history = self.model.fit(self.X_train, {'class_output': self.cls_train, 'reg_output': self.offset_train}, 
                                      batch_size=8,
                                      epochs=200,
                                      validation_data=(self.X_val, {'class_output': self.cls_val, 'reg_output': self.offset_test}),
                                      callbacks=self.callbacks)
        
    def evaluate(self):
        """
        Evaluates the model on the test dataset.
        """
        pass

    def test(self, input_image, output_path):
        """
        Tests the model on a single image.

        Args:
            input_image (str): The path to the input image.
            output_path (str): The path to save the output image.
        """
        pass
    
