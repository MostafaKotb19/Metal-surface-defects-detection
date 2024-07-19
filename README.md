# Metal Surface Defects Detection

This repository contains the implementation of a model for automated detection of defects on metal surfaces using Vision Transformers (ViTs). The model leverages the self-attention mechanism of ViTs for enhanced feature extraction, which is then used for defect classification and localization through a combination of CNNs and Multi-Layer Perceptrons (MLPs). This approach aims to improve the efficiency and accuracy of defect detection in metal manufacturing, addressing the limitations of traditional manual inspection methods. The source code is included to facilitate further research and development in this field.

## Install

To create an environment with the requirements:
```
conda env create -f environment.yml
conda activate metal_detection
```

## Usage

To train this model:
```
python main.py --train --config config.yaml
```

To evaluate the model on the whole test dataset:
```
python main.py --evaluate --config config.yaml
```

To test the model on a single image:
```
python main.py --test --config config.yaml --input /path/to/input/image --output /path/to/output/image
```

## Description

[config.yaml](config.yaml) contains the configuration for the anchor boxes, dataset, and training.

[main.py](main.py) is the main executable. It contains the CLI parsing and the main running.

[anchor_boxes.py](anchor_boxes.py) contains the creation of the anchor boxes and offset calculations.

[dataset.py](dataset.py) loads the dataset in the train-val-test format.

[loss.py](loss.py) contains all the used loss functions.

[model.py](model.py) represents the used model based on the ViT.

[preprocessing.py](preprocessing.py) performs all the needed perprocessing on images and labels.

[utils.py](utils.py) contains useful utils, like min-max scaling, and dict2namespace.
