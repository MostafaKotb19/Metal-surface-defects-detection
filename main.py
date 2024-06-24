from model import ViTModel
from dataset import Dataset
from preprocessing import ImagePreprocessing, DimsPreprocessing
import argparse
import yaml
from utils import dict2namespace

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--evaluate', action='store_true', default=False, help='test the model on the whole test dataset')
    
    parser.add_argument('--test', action='store_true', default=False, help='test the model on a single image')
    parser.add_argument('-i', '--input', type=str, default=None, help='Path to the image to test the model on')
    parser.add_argument('-o', '--output', type=str, default=None, help='Path to save the output image')
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)
    
    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    dataset = Dataset(namespace_config)
    image_preprocessing = ImagePreprocessing(dataset.X_train, dataset.X_val, dataset.X_test)
    dims_preprocessing = DimsPreprocessing(dataset.y_train, dataset.y_val, dataset.y_test, namespace_config)

    model = ViTModel(namespace_config, 
                     image_preprocessing.X_train, dims_preprocessing.cls_train, dims_preprocessing.offset_train,
                     image_preprocessing.X_val, dims_preprocessing.cls_val, dims_preprocessing.offset_val,
                     image_preprocessing.X_test, dims_preprocessing.cls_test, dims_preprocessing.offset_test)
    
    if args.train:
        model.train()

    elif args.evaluate:
        model.evaluate()

    elif args.test:
        model.test(args.input, args.output)
        


