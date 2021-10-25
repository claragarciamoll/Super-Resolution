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
├── arch
│   ├── esrgan
│   │   ├── __init__.py
│   │   ├── arch_util.py
│   │   ├── discriminator_arch.py
│   │   ├── rrdbnet_arch.py
│   │   ├── vgg_arch.py
├── config
│   ├── config_deblurring.py
│   ├── config_esrgan.py
│   ├── config_liif.py
│   ├── config_msrn.py
│   ├── config_srcnn.py
│   ├── config.py
│   ├── test_esrgan.yaml
│   ├── test_liif.yaml
│   ├── train_esrgan.yaml
│   ├── train_liif.yaml
├── Data
│   ├── UCMerced_LandUse
│   │   ├── train
│   │   ├── test
│   │   ├── val
├── datasets
│   ├── esrgan
│   │   ├── __init__.py
│   │   ├── data_sampler.py
│   │   ├── data_util.py
│   │   ├── paired_image_dataset.py
│   │   ├── prefetch_dataloader.py
│   │   ├── transforms.py
│   ├── liif
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── image_folder.py
│   │   ├── wrappers.py
│   ├── msrn.py
│   ├── srcnn.py
├── deblurring
│   ├── config.py
│   ├── data_RGB.py
│   ├── dataset_RGB.py
│   ├── evaluate_RealBlur.py
│   ├── losses.py
│   ├── MPRNet.py
│   ├── test.py
│   ├── tarin.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── dataset_utils.py
│   │   ├── dir_utils.py
│   │   ├── image_utils.py
│   │   ├── model_utils.py
│   ├── warmup_scheduler.py
├── losses
│   ├── esrgan
│   │   ├── __init__.py
│   │   ├── loss_util.py
│   │   ├── losses.py
├── models
│   ├── esrgan
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── esrgan_model.py
│   │   ├── lr_scheduler.py
│   │   ├── RRDBNet_arch.py
│   │   ├── sr_model.py
│   │   ├── srgan_model.py
│   ├── liif
│   │   ├── __init__.py
│   │   ├── liif.py
│   │   ├── models.py
│   │   ├── edsr.py
│   │   ├── mlp.py
│   │   ├── misc.py
│   │   ├── rcan.py
│   │   ├── rdn.py
│   ├── model_fsrcnn.py
│   ├── msrn
├── ops
│   ├── esrgan
│   │   ├── __init__.py
│   │   ├── deform_conv.py
├── testing
│   ├── test_deblurring.py
│   ├── test_esrgan.py
│   ├── test_fsrcnn.py
│   ├── test_liif.py
│   ├── test_msrn.py
├── training
│   ├── training_esrgan.py
│   ├── training_fsrcnn.py
│   ├── training_liif.py
│   ├── training_msrn.py
├── utils
│   ├── esrgan
│   │   ├── __init__.py
│   │   ├── dist_util.py
│   │   ├── extract_subimages.py
│   │   ├── file_client.py
│   │   ├── img_util.py
│   │   ├── logger.py
│   │   ├── matlab_functions.py
│   │   ├── misc.py
│   │   ├── options.py
│   │   ├── registry.py
│   │   ├── test_paired_image_dataset.py
│   ├── utils_fsrcnn.py
│   ├── utils_liif.py
│   ├── utils_msrn.py
│   ├── utils.py
├── main.py


```

### Data

  1. Create a directory ```Data/nameDataset```. We use UC Merced Land Use Dataset that could be downloaded from: http://weegee.vision.ucmerced.edu/datasets/landuse.html
  2. Using the function ```train_val_test_split``` from ```utils/utils.py``` the Dataset will be split into 3 partitions: Training (60%), Validation (20%) and Test (20%).

### Run the code

1. Choose the configuration to execute, modify the file ```config/config.py```, where the method and the scale could be selected.
2. Run:
```bash
$ python main.py 
```

## Results

### Quantitative Results

<p align="center">
    <img src="https://github.com/claragarciamoll/Super-Resolution/blob/7d60774a10cbe5f90f4747ba2d2a13b8d96a45ef/metriques_psnr_ssim.png" width="700" align="center">
</p>

<p align="center">
    <img src="https://github.com/claragarciamoll/Super-Resolution/blob/7d60774a10cbe5f90f4747ba2d2a13b8d96a45ef/metriques_swd_fid.png" width="700" align="center">
</p>

### Qualitative Results

<p align="center">
    <img src="https://github.com/claragarciamoll/Super-Resolution/blob/1eda509c672082f170fc87e32f223a6b222e7b11/qualitative.png" width="700" align="center">
</p>

## Contact
If you have any questions, please email ```clara.garciamoll@gmail.com```

For more information about each method you can check:
* ESRGAN: https://github.com/xinntao/ESRGAN
* FSRCNN: https://github.com/yjn870/FSRCNN-pytorch
* LIIF: https://github.com/yinboc/liif
* MSRN: https://github.com/MIVRC/MSRN-PyTorch

