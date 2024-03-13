# DAC-Net

This repository is the official implementation of DAC-Net : An Light-weight U-shaped Network Based Efficient Convolution And Dual-Attention for Thyroid Nodule Segmentation using PyTorch.



## Main Environments

- python 3.9
- pytorch 2.1.0
- torchvision 0.16.0

## Requirements

Install from the `requirements.txt` using:

```
pip install -r requirements.txt
```

## Prepare the dataset.

- The DDTI and TN3K datasets, divided into a 7:1:2 ratio, can be found here ([GoogleDrive](https://drive.google.com/drive/folders/1za9f38XKx-VYPxxb_xx83Dpk-Wg3Yaw8?usp=drive_link)).


- Then prepare the datasets in the following format for easy use of the code:

```
├── datasets
    ├── DDTI
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── TN3k
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol
```

## Train the Model

First, modify the model, dataset and training hyperparameters in `Config.py`

Then simply run the training code.

```
python3 train_model.py
```

## Evaluate the Model

Please make sure the right model and dataset is selected in `Config.py`

Then simply run the evaluation code.

```
python3 test_model.py
```




## Citation

If you find this work useful in your research or use this dataset in your work, please consider citing the following papers:
