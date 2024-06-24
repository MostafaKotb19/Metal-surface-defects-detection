import xml.etree.ElementTree as ET
import os
import cv2
from sklearn.model_selection import train_test_split

# Mapping of class names to integers
MAPPING = {
        "1_chongkong" : 0,
        "2_hanfeng" : 1,
        "3_yueyawan": 2,
        "4_shuiban": 3,
        "5_youban": 4,
        "6_siban": 5,
        "7_yiwu": 6,
        "inclusion" :6,
        "9_zhehen": 7,
        "10_yaozhed": 8,
        "scratches": 9,
        "rolled-in_scale": 10,
        "8_yahen": 11,
    }

class Dataset:
    """
    A class to represent a dataset.

    Args:
        namespace_config (Namespace): A namespace containing the configuration settings.

    Args from NamespaceConfig:
        dataset (Namespace): Namespace containing dataset configuration.
        dataset.dataset_path (str): The path to the dataset.
        dataset.w1 (int): The width of the dataset images. (GC10-DET)
        dataset.w2 (int): The width of the dataset images. (GC10-DET)
        dataset.l1 (int): The length of the dataset images. (NEU-DET)
        dataset.l2 (int): The length of the dataset images. (NEU-DET)

    Attributes:
        w (int): The width of the output images.
        l (int): The length of the output images.
        X (list): A list of images.
        y (list): A list of labels.
        X_train (list): A list of training images.
        y_train (list): A list of training labels.
        X_val (list): A list of validation images.
        y_val (list): A list of validation labels.
        X_test (list): A list of test images.
        y_test (list): A list of test labels.
    """

    def __init__(self, namespace_config):

        self.mapping = MAPPING
        
        self.dataset_path = namespace_config.dataset.dataset_path
        self.w1 = namespace_config.dataset.w1
        self.w2 = namespace_config.dataset.w2
        self.l1 = namespace_config.dataset.l1
        self.l2 = namespace_config.dataset.l2

        self.w = 224
        self.l = 224

        self.calculate_factors(w=self.w, l=self.l)

        all_items = os.listdir(self.dataset_path)
        self.categories = [item for item in all_items if os.path.isdir(os.path.join(self.dataset_path, item)) 
                           and item != 'lable']
        
        self.X, self.y = self.load_data(self.dataset_path, self.categories)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.split_data(self.X, self.y)

    def calculate_factors(self, w, l):
        """
        Calculate factors for resizing images.
        """
        self.w_factor = self.w1/w
        self.l_factor = self.l1/l
        self.w_factor_2 = self.w2/w
        self.l_factor_2 = self.l2/l

    def get_annotations(self, directory, class_name, idx):
        """
        Retrieves annotations from an XML file and returns a list of bounding boxes.

        Args:
            directory (str): The path to the XML file.
            class_name (str): The name of the class.
            idx (int): The index.

        Returns:
            list: A list of bounding boxes, where each box is represented as a list of values.

        Raises:
            Exception: If an error occurs while parsing the XML file.
        """
        try:
            tree = ET.parse(directory)
            root = tree.getroot()
            boxes = []
            defect = -1
            objs = root.findall('object')
            for i in objs:
                name = i.find('name').text
                defect = self.mapping.get(name, 12)
                if defect != 12 and defect != 11:
                    box = i.find('bndbox')
                    xmin = float(box.find('xmin').text)
                    ymin = float(box.find('ymin').text)
                    xmax = float(box.find('xmax').text)
                    ymax = float(box.find('ymax').text)

                
                    if xmin > xmax:
                        temp = xmin
                        xmin = xmax
                        xmax = temp
                    if ymin > ymax:
                        temp = ymin
                        ymin = ymax
                        ymax = temp

                    box = []
                    for i in range(5):
                        box.append(-1)
                    box[0] = defect
                    if class_name !='inclusion':
                        box[1] = (xmin)/self.w_factor
                        box[2] = (ymin)/self.l_factor
                        box[3] = (xmax)/self.w_factor
                        box[4] = (ymax)/self.l_factor
                    elif (class_name=='inclusion' and idx<217):   # GC10-DET
                        box[1] = (xmin)/self.w_factor
                        box[2] = (ymin)/self.l_factor
                        box[3] = (xmax)/self.w_factor
                        box[4] = (ymax)/self.l_factor
                    
                    elif (class_name=='inclusion' and idx>= 217): # NEU-DET
                        box[1] = (xmin)/self.w_factor_2
                        box[2] = (ymin)/self.l_factor_2
                        box[3] = (xmax)/self.w_factor_2
                        box[4] = (ymax)/self.l_factor_2
                        
                    boxes.append(box)
                else:
                    defect = -1
            if boxes:
                return boxes
            print(class_name, idx)
            return 0
        except:
            return 0

    def load_data(self, dataset_path, categories):
        """
        Load data from the specified data directory for the given categories.

        Args:
            dataset_path (str): The path to the directory containing the image data.
            categories (list): A list of category names.

        Returns:
            tuple: A tuple containing the feature data and labels.
        """
        X = []
        y = []
        for category in categories:
            category_path= os.path.join(dataset_path, category)
            cnt=1

            for img in os.listdir(category_path):
                img_path= os.path.join(category_path, img)
                img= cv2.imread(img_path)
                annotation = self.get_annotations(f"{dataset_path}/lable/{category} ({cnt}).xml", category, cnt)
                cnt+=1

                if (annotation):
                    img = cv2.resize(img, (self.w, self.l))
                    X.append(img)
                    y.append(annotation)
        return X, y
    
    def split_data(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

        return X_train, y_train, X_val, y_val, X_test, y_test