import math
import os
import cv2
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy import ndimage
import PIL.Image as pil_image
from PIL import ImageFilter
from scipy.ndimage import gaussian_filter


def train_val_test_split(df, train_percent=.6, val_percent=.2, seed=None):
    """
    Split dataset into 3 partitions: Train, Validation and Test
    :param df: dataset directory
    :param train_percent: Percentage images of the dataset that would be in the train partition
    :param val_percent: Percentage images of the dataset that would be in the validation and test partition
    """
    classes = sorted(glob.glob('{}/*'.format(df)))
    for data_class in classes:
        images_paths = sorted(glob.glob('{}/*'.format(data_class)))
        m = len(images_paths)
        train = random.sample(images_paths, int(m * train_percent))
        images_paths = [x for x in images_paths if x not in train]
        val = random.sample(images_paths, int(m * val_percent))
        test = [x for x in images_paths if x not in val]
        for train_path in train:
            if not os.path.exists(df + '/train'):
                os.makedirs(df + '/train')
            image = cv2.imread(train_path)
            name = train_path.split('/')[-1]
            cv2.imwrite(df + '/train/' + name, image)
        for val_path in val:
            if not os.path.exists(df + '/val'):
                os.makedirs(df + '/val')
            image = cv2.imread(val_path)
            name = val_path.split('/')[-1]
            cv2.imwrite(df + '/val/' + name, image)
        for test_path in test:
            if not os.path.exists(df + '/test'):
                os.makedirs(df + '/test')
            image = cv2.imread(test_path)
            name = test_path.split('/')[-1]
            cv2.imwrite(df + '/test/' + name, image)
        train = []
        val = []
        test = []


def downsampling_images(folder, scale):
    """
    Split Downsample images
    :param folder: image folder directory
    :param scale: scale to downsample the images
    """
    for image_name in sorted(glob.glob(folder + '/*')):
        image = pil_image.open(image_name).convert('RGB')
        img_width = int(image.width // scale)
        img_height = int(image.height // scale)
        down_image = image.resize((img_width, img_height), resample=pil_image.BICUBIC)
        if not os.path.exists(folder + '_downx' + str(scale)):
            os.makedirs(folder + '_downx' + str(scale))
        print(folder + '_downx' + str(scale))
        #print(folder + '_downx' + str(scale) + '/' + image_name.split('/')[-1])
        down_image.save(folder + '_downx' + str(scale) + '/' + image_name.split('/')[-1])
        
        if (image.width % 2) == 0:
            image = image.crop((0, 0, image.width-1, image.height-1))
            if not os.path.exists(folder + '_crop'):
                os.makedirs(folder + '_crop')
            image.save(folder + '_crop' + '/' + image_name.split('/')[-1])


def cropimg(folder):
    """
    Crop images
    :param folder: image folder directory
    """
    for image_name in sorted(glob.glob(folder + '/*')):
        image = pil_image.open(image_name).convert('RGB')
        if (image.width % 2) == 0:
            image = image.crop((0, 0, image.width-1, image.height-1))
            if not os.path.exists(folder + '_crop'):
                os.makedirs(folder + '_crop')
            image.save(folder + '_crop' + '/' + image_name.split('/')[-1])


def resampling_images(folder, scale):
    """
        Resampling images
        :param folder: image folder directory
        :param scale: scale to resample the images
        """
    for image_path in sorted(glob.glob(folder + '/*')):
        print(image_path)
        img = pil_image.open(image_path).convert('RGB')
        img_width = int(img.width // scale)
        img_height = int(img.height // scale)
        resampled_img = img.resize((img_width, img_height), resample=pil_image.BICUBIC)
        img_name = image_path.split('/')[-1]
        res_scale = scale * 0.3
        save_folder = folder + 'resampling_' + '07'#str(res_scale)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        resampled_img.save(save_folder + '/' + img_name)


def post_resampling(gt_path, est_path, scale):
    gt_img = pil_image.open(gt_path).convert('RGB') # image size 256 x 256 (native resolution)
    est_img = pil_image.open(est_path).convert('RGB')
    imgs_width = int(gt_img.width)
    imgs_height = int(gt_img.height)
    print(imgs_width)
    print(scale)
    print(int(imgs_width // scale))
    resampled_est_img = est_img.resize((int(imgs_width // scale), int(imgs_height // scale)), resample=pil_image.BICUBIC)
    print(est_path.split('/')[:-1])
    img_name = gt_path.split('/')[-1]
    folder = join(*est_path.split('/')[:-1])
    save_folder = folder + '_resampling_' + str(scale)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    resampled_est_img.save(save_folder + '/' + img_name)


'''def add_blurring(folder, kernel):
    for image_path in sorted(glob.glob(folder + '/*')):
        img = pil_image.open(image_path).convert('RGB')
        blur_img = img.filter(ImageFilter.GaussianBlur(kernel))
        save_folder = folder + '_blur_' + str(kernel)
        img_name = image_path.split('/')[-1]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        blur_img.save(save_folder + '/' + img_name)'''

def show_methods_differences(gt_path, img_m1_path, img_m2_path, thr, step, save_folder, methods):
    """
    Comparative between prediction from 2 SR methods and the GT
    :param gt_path: GT folder directory
    :param img_m1_path: Method 1 predictions folder directory
    :param img_m2_path: Method 2 predictions folder directory
    :param thr: threshold to decide which method predictions are the most similar to the GT
    :param step: step used to go throw the image
    :param save_folder: save folder directory
    :param methods: methods used in the comparison
    """
    name = gt_path.split('/')[-1]
    name = name.split('.')[0]
    if not os.path.exists(os.path.join(save_folder, methods)):
        os.makedirs(os.path.join(save_folder, methods))

    gt = cv2.imread(gt_path)
    gt = gaussian_filter(gt, sigma=1)
    img_m1 = cv2.imread(img_m1_path)
    img_m1 = gaussian_filter(img_m1, sigma=1)
    img_m2 = cv2.imread(img_m2_path)
    img_m2 = gaussian_filter(img_m2, sigma=1)
    r, c, z = gt.shape
    mask = np.zeros((r, c, 3))
    dif_gt_m1 = np.abs(gt - img_m1)
    cv2.imwrite(os.path.join(save_folder, methods, name + 'gt_m1.png'), dif_gt_m1)
    dif_gt_m2 = np.abs(gt - img_m2)
    cv2.imwrite(os.path.join(save_folder, methods, name + 'gt_m2.png'), dif_gt_m2)

    # GREEN WILL BE ASSIGNED TO THOSE PIXELS THAT HAS LOWER DIFF USING FIRST METHOD AND RED TO THE SECOND
    for i in range(step,r-step-1,step):
        for j in range(0,c-1,step):
            print(np.mean(dif_gt_m1[i, j, :]))
            print(np.mean(dif_gt_m2[i, j, :]))
            print('-------------------------')
            if int(np.mean(dif_gt_m1[i, j, :])) > thr and int(np.mean(dif_gt_m2[i, j, :])) > thr:
                if np.mean(dif_gt_m1[i, j, :]) > np.mean(dif_gt_m2[i, j, :]):
                    mask[i-step:i+step, j-step:j+step, :] = [0, 0, 1]
                else:
                    mask[i-step:i+step, j-step:j+step, :] = [0, 1, 0]
    mask = mask*255
    cv2.imwrite(os.path.join(save_folder, methods, name + '.png'), mask)


def spectrum2D(path):
    """
    Compute the Spectrum of the images
    :param path: image folder directory
    :return radialProfile_new: Radially Fourier Spectrum array
    """
    image = cv2.imread(path,0)
    fftOriginal = np.fft.fft2(image)
    shiftedFFT = np.fft.fftshift(fftOriginal)
    shiftedFFTMagnitude = np.abs(shiftedFFT)
    ## AVERAGE RADIAL
    rows = image.shape[0]
    cols = image.shape[1]
    midRow = rows/2+1
    midCol = cols/2+1
    maxRadius = math.ceil(np.sqrt((midRow+2)**2 + (midCol+2)**2))
    radialProfile = np.zeros((maxRadius, 1))
    count = np.zeros((maxRadius, 1))
    for i in range(cols):
        for j in range(rows):
            radius = np.sqrt((j-midRow)**2 + (i-midCol)**2)
            thisIndex = math.ceil(radius) + 1
            radialProfile[thisIndex] = radialProfile[thisIndex] + shiftedFFTMagnitude[j,i]
            count[thisIndex] = count[thisIndex] + 1
    radialProfile_new = radialProfile / count
    for k in range(radialProfile_new.shape[0]):
        if math.isnan(radialProfile_new[k]):
            radialProfile_new[k] = 0.
    return radialProfile_new

def tolerant_mean(arrs):
    """
    Compute the mean of the Radially Fourier Spectrum array
    :param arrs: Radially Fourier Spectrum arrays
    :return arr: Radially Averaged Fourier Spectrum
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l.flatten()
    return arr

def plotSpectrums2D(path_fsrcnn, path_liif, path_msrn, path_esrgan, path_bicubic, out_path):
    """
    Get the plots of the Radially Averaged Fourier Spectrum for the different SR methods
    :param path_fsrcnn: FSRCNN predictions folder directory
    :param path_liif: LIIF predictions folder directory
    :param path_msrn: MSRN predictions folder directory
    :param path_esrgan: ESRGAN predictions folder directory
    :param path_bicubic: Bicubic predictions folder directory
    :param out_path: output folder directory where the plots will be saved
    """
    radial_FSRCNN = []
    radial_LIIF = []
    radial_MSRN = []
    radial_ESRGAN = []
    radial_BICUBIC = []

    for FSRCNN_path, LIIF_path, MSRN_path, ESRGAN_path, bicubic_path in zip(sorted(glob.glob(path_fsrcnn)), sorted(glob.glob(path_liif)), sorted(glob.glob(path_msrn)), sorted(glob.glob(path_esrgan)), sorted(glob.glob(path_bicubic))):
        print(FSRCNN_path)
        radialProfile_FSRCNN = spectrum2D(FSRCNN_path)
        print(len(radialProfile_FSRCNN))
        radial_FSRCNN.append(radialProfile_FSRCNN)
        print(LIIF_path)
        radialProfile_LIIF = spectrum2D(LIIF_path)
        radial_LIIF.append(radialProfile_LIIF)
        print(MSRN_path)
        radialProfile_msrn = spectrum2D(MSRN_path)
        radial_MSRN.append(radialProfile_msrn)
        print(ESRGAN_path)
        radialProfile_esrgan = spectrum2D(ESRGAN_path)
        radial_ESRGAN.append(radialProfile_esrgan)
        print(bicubic_path)
        radialProfile_bicubic = spectrum2D(bicubic_path)
        radial_BICUBIC.append(radialProfile_bicubic)

        image_name = FSRCNN_path.split('/')[-1]
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        plt.figure()
        plt.plot(radialProfile_FSRCNN, label='FSRCNN')
        plt.plot(radialProfile_LIIF, label='LIIF')
        plt.plot(radialProfile_msrn, label='MSRN')
        plt.plot(radialProfile_esrgan, label='ESRGAN')
        plt.plot(radialProfile_bicubic, label='BICUBIC')
        plt.legend()
        plt.title(image_name)
        plt.yscale('log')
        plt.savefig(out_path + image_name)

    mean_FSRCNN = tolerant_mean(radial_FSRCNN)
    mean_LIIF = tolerant_mean(radial_LIIF)
    mean_MSRN = tolerant_mean(radial_MSRN)
    mean_ESRGAN = tolerant_mean(radial_ESRGAN)
    mean_BICUBIC = tolerant_mean(radial_BICUBIC)
    plt.figure()
    plt.plot(mean_FSRCNN, label='FSRCNN')
    plt.plot(mean_LIIF, label='LIIF')
    plt.plot(mean_MSRN, label='MSRN')
    plt.plot(mean_ESRGAN, label='ESRGAN')
    plt.plot(mean_BICUBIC, label='BICUBIC')
    plt.legend()
    plt.title('MEAN')
    plt.yscale('log')
    plt.savefig(out_path + 'mean')
    #return radialProfile_FSRCNN.shape, radialProfile_LIIF.shape, radialProfile_msrn.shape





