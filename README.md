# Beyond one-meter resolution: A comparative study on super-resolution methods

This repository contains the master's dissertation where different Super-resolution (SR) algorithms were compared. The main goal was compare different SR algorithms:

* LIIF (Local Implicit Image Function)
* MSRN (Multi-Scale Residual Network)
* FSRCNN (Fast Super-Resolution Convolutional Neural Network)
* ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)

## Environment

* Python 3
* Pytorch 1.6.0
* TensorboardX
* yaml, numpy, tqdm, openCV

## Reproduce Experiments

### Directory structure
```bash
├──

```

### Data

  1. Create a directory Data/Dataset. We use UC Merced Land Use Dataset that could be downloaded from (http://weegee.vision.ucmerced.edu/datasets/landuse.html)
  2. Using the function train_val_test_split from utils/utils.py the Dataset will be split into 3 partitions: Training (60%), Validation (20%) and Test (20%).

### Run the code

1. Choose the configuration to execute, modify the file config/config.py, where the method and the scale could be selected.
2. Run:
```bash
$ python main.py 
```

## Results

<p align="center">
    <img src="" width="700" align="center">
</p>


