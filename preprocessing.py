import numpy as np
import anchor_boxes
from tensorflow.keras.utils import to_categorical
from utils import extract_max, extract_min, max_min_scaler
from transformers import AutoImageProcessor

class ImagePreprocessing:
    def __init__(self, X_train, X_val, X_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.preprocess()

    def preprocess(self):
        self.X_train = np.array(self.X_train, dtype= 'float32')
        self.X_val = np.array(self.X_val, dtype= 'float32')
        self.X_test = np.array(self.X_test, dtype= 'float32')

        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        self.X_train= image_processor(self.X_train, return_tensors="tf")
        self.X_val= image_processor(self.X_val, return_tensors="tf")
        self.X_test= image_processor(self.X_test, return_tensors="tf")

        self.X_train= self.X_train['pixel_values']
        self.X_val = self.X_val['pixel_values']
        self.X_test = self.X_test['pixel_values']
        

class DimsPreprocessing:
    def __init__(self, y_train, y_val, y_test, namespace_config):
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.namespace_config = namespace_config
        self.preprocess()

    def preprocess(self):
        self.classes_train, self.dims_train = self.class_dim_split(self.y_train)
        self.classes_val, self.dims_val = self.class_dim_split(self.y_val)
        self.classes_test, self.dims_test = self.class_dim_split(self.y_test)

        AnchorBoxes = anchor_boxes.AnchorBoxes(self.namespace_config, (224, 224))
        self.offset_train, self.cls_train = AnchorBoxes.class_offset_split(self.dims_train, self.classes_train)
        self.offset_val, self.cls_val = AnchorBoxes.class_offset_split(self.dims_val, self.classes_val)
        self.offset_test, self.cls_test = AnchorBoxes.class_offset_split(self.dims_test, self.classes_test)

        self.extract_max_min()

        self.offset_train = self.scale_dimensions(self.offset_train)
        self.offset_val = self.scale_dimensions(self.offset_val)
        self.offset_test = self.scale_dimensions(self.offset_test)

        self.num_classes = self.namespace_config.dataset.num_classes

        self.one_hot_encoding()
        self.np_arraying()

        self.offset_train = np.array(self.offset_train, dtype = 'float')
        self.offset_val = np.array(self.offset_val, dtype = 'float')
        self.offset_test = np.array(self.offset_test, dtype = 'float')
        self.cls_train = np.array(self.cls_train, dtype = 'float')
        self.cls_val = np.array(self.cls_val, dtype = 'float')
        self.cls_test = np.array(self.cls_test, dtype = 'float')

    def class_dim_split(self, y):
        classes = []
        dims = []
        for i in range(len(y)):
            tmp_classes = []
            tmp_dims = []
            for j in range(len(y[i])):
                tmp_classes.append(y[i][j][0])
                tmp_dims.append(y[i][j][1:])
            classes.append(tmp_classes)
            dims.append(tmp_dims)
        return classes, dims
    
    def extract_max_min(self):
        """
        Extract the maximum and minimum values for each dimension.
        """
        self.x1_max_val = extract_max(self.offset_train, 0)
        self.y1_max_val = extract_max(self.offset_train, 1)
        self.x2_max_val = extract_max(self.offset_train, 2)
        self.y2_max_val = extract_max(self.offset_train, 3)

        self.x1_min_val = extract_min(self.offset_train, 0)
        self.y1_min_val = extract_min(self.offset_train, 1)
        self.x2_min_val = extract_min(self.offset_train, 2)
        self.y2_min_val = extract_min(self.offset_train, 3)

    def scale_dimensions(self, dims):
        """
        Scale the dimensions of the bounding boxes.
        
        Args:
            dims (list): A list of bounding box dimensions.
            
        Returns:
            list: The scaled bounding box dimensions.
        """

        scaled_dims = max_min_scaler(dims, self.x1_max_val, self.x1_min_val, 0)
        scaled_dims = max_min_scaler(scaled_dims, self.y1_max_val, self.y1_min_val, 1)
        scaled_dims = max_min_scaler(scaled_dims, self.x2_max_val, self.x2_min_val, 2)
        scaled_dims = max_min_scaler(scaled_dims, self.y2_max_val, self.y2_min_val, 3)

        return scaled_dims
    
    def one_hot_encoding(self):
        """
        Perform one-hot encoding on the class labels.
        
        Args:
            cls (list): A list of class labels.
            
        Returns:
            list: The one-hot encoded class labels.
        """
        for i in range(len(self.cls_train)):
            self.cls_train[i] = np.array(self.cls_train[i], dtype = 'float').astype('float64').reshape((-1,1))
            self.cls_train[i] = to_categorical(self.cls_train[i], num_classes=self.num_classes)

        for i in range(len(self.cls_val)):
            self.cls_val[i] = np.array(self.cls_val[i], dtype = 'float').astype('float64').reshape((-1,1))
            self.cls_val[i] = to_categorical(self.cls_val[i], num_classes=self.num_classes)

        for i in range(len(self.cls_test)):
            self.cls_test[i] = np.array(self.cls_test[i], dtype = 'float').astype('float64').reshape((-1,1))
            self.cls_test[i] = to_categorical(self.cls_test[i], num_classes=self.num_classes)

    def np_arraying(self):
        for i in range(len(self.offset_train)):
            self.offset_train[i] = np.array(self.offset_train[i], dtype = 'float')

        for i in range(len(self.offset_val)):
            self.offset_val[i] = np.array(self.offset_val[i], dtype = 'float')

        for i in range(len(self.offset_test)):
            self.offset_test[i] = np.array(self.offset_test[i], dtype = 'float')