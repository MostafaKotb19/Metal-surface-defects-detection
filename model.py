from transformers import TFViTModel
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Lambda, LeakyReLU
from tensorflow.keras import mixed_precision
import numpy as np
from loss import modified_accuracy, modified_categorical_crossentropy, modified_mse, modified_mae

class EarlyStop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('class_output_accuracy2') is not None and logs.get('class_output_accuracy2') > 0.85 \
        and logs.get('val_class_output_accuracy2') is not None and logs.get('val_class_output_accuracy2') > 0.85:
            self.model.stop_training = True

class CustomSaveCheckpoint(tf.keras.callbacks.Callback):
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
        super(CustomSaveCheckpoint, self).set_model(model)
        self.checkpoint_callback.set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.save_freq:
            self.epochs_since_last_save = 0
            self.checkpoint_callback.on_epoch_end(epoch, logs)

class Model(tf.keras.Model):
    def __init__(self, base, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.base.trainable = False
        self.conv1 = Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu')
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.dense_class1 = Dense(512, activation=LeakyReLU(negative_slope=0.01), 
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.drop = Dropout(0.2)
        self.dense_class2 = Dense(256, activation=LeakyReLU(negative_slope=0.01), 
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.output_dense_class1 = Dense(225*13, activation=LeakyReLU(negative_slope=0.01), 
                                         kernel_initializer='glorot_uniform',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.output_reshape_class1 = Reshape((225, 13))
        self.softmax_single_class = Dense(13, activation='softmax')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.output_reshape_class2 = Reshape((225, 13), name='class_output')
        
        self.dense_reg1 = Dense(256, activation=LeakyReLU(negative_slope=0.01),
                                kernel_initializer='glorot_uniform',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense_reg2 = Dense(128, activation=LeakyReLU(negative_slope=0.01),
                                kernel_initializer='glorot_uniform',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense_reg3 = Dense(128, activation=LeakyReLU(negative_slope=0.01),
                                kernel_initializer='glorot_uniform',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense_reg4 = Dense(64, activation=LeakyReLU(negative_slope=0.01),
                                kernel_initializer='glorot_uniform',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.output_dense_reg1 = Dense(225*4)
        self.output_reshape_reg1 = Reshape((225, 4))
        self.sigmoid_single_reg = Dense(4, activation='sigmoid')
        self.output_reshape_reg2 = Reshape((225, 4), name='reg_output')
        
    def call(self, inputs):
        x = self.base(inputs)
        x = tf.expand_dims(x['last_hidden_state'], axis=-1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        
        y = self.dense_class1(x)
        y = self.drop(y)
        y = self.dense_class2(y)
        y = self.drop(y)
        y = self.output_dense_class1(y)
        y = self.output_reshape_class1(y)
        softmax_layers = []
        for i in range(225):
            row = Lambda(lambda x: x[:, i, :])(y)
            softmax = self.softmax_single_class(row)
            softmax_layers.append(softmax)
        
        y = self.concatenate(softmax_layers)
        y = self.output_reshape_class2(y)
            
        z = self.dense_reg1(x)
        z = self.drop(z)
        z = self.dense_reg2(z)
        z = self.drop(z)
        z = self.dense_reg3(z)
        z = self.dense_reg4(z)
        z = self.output_dense_reg1(z)
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

        
        self.extract = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.callbacks= [EarlyStop(), CustomSaveCheckpoint(filepath='weights_epoch_{epoch:02d}.weights.h5', save_freq=5, verbose=1)]

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        self.model = Model(self.extract)
        self.dummy_input()
        self.print_summary()
        self.load_weights(self.weights)

    def dummy_input(self):
        self.model(np.array([self.X_train[0]]))

    def print_summary(self):
        if self.summary:
            self.model.summary()

    def load_weights(self, weights):
        if self.weights:
            self.model.load_weights(weights)

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
              loss={'class_output': modified_categorical_crossentropy,
                    'reg_output': modified_mse},
              loss_weights={'class_output': self.class_loss_weight, 'reg_output': self.reg_loss_weight},
              metrics={'class_output': modified_accuracy,
                       'reg_output': modified_mae},
              run_eagerly=self.run_eagerly)
        
    def train(self):
        self.compile()
        self.history = self.model.fit(self.X_train, {'class_output': self.cls_train, 'reg_output': self.offset_train}, 
                                      batch_size=8,
                                      epochs=200,
                                      validation_data=(self.X_val, {'class_output': self.cls_val, 'reg_output': self.offset_test}),
                                      callbacks=self.callbacks)
        
    def evaluate(self):
        pass

    def test(self, input_image, output_path):
        pass
    
