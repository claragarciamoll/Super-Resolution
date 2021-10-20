from os.path import join
import glob
import cv2

# IMPORT GENERALS
from utils.utils import *
from config.config import Config
from utils.esrgan.extract_subimages import *
from utils.esrgan.test_paired_image_dataset import *
# IMPORT CONFIGS
from config.config_liif import Config_liif
from config.config_srcnn import Config_srcnn
from config.config_msrn import Config_msrn
from config.config_esrgan import Config_esrgan
from config.config_deblurring import Config_deblurring
# IMPORT DATASETS
from datasets.srcnn import PrepareDataset
# IMPORT TRAINING
from training.training_fsrcnn import *
from training.training_liif import *
from training.training_msrn import *
from training.training_esrgan import *
from deblurring.train import *
# IMPORT TESTING
from testing.test_fsrcnn import *
from testing.test_liif import *
from testing.test_msrn import *
from testing.test_esrgan import *
from deblurring.test import *


def main(args):
    print(args)
    if args.split: #Split dataset into Training - Validation - Test partitions
        train_val_test_split(join(args.data_path, args.dataset))

    if args.resampling: # resampling dataset images (if needed)
        s = args.resample_scale/args.native_resolution
        resampling_images(join(args.data_path, args.dataset, 'train'), s)
        resampling_images(join(args.data_path, args.dataset, 'val'), s)
        resampling_images(join(args.data_path, args.dataset, 'test'), s)

    if args.add_blur: # add blur to simulate the PSF of the satellite
        add_blurring(join(args.data_path, args.dataset, 'trainresampling_0.33'), args.kernel_size)
        add_blurring(join(args.data_path, args.dataset, 'trainresampling_0.5'), args.kernel_size)
        add_blurring(join(args.data_path, args.dataset, 'valresampling_0.33'), args.kernel_size)
        add_blurring(join(args.data_path, args.dataset, 'valresampling_0.5'), args.kernel_size)
        add_blurring(join(args.data_path, args.dataset, 'testresampling_033'), args.kernel_size)
        add_blurring(join(args.data_path, args.dataset, 'testresampling_05'), args.kernel_size)

    if args.compare_img: # Compare output images of different methods vs GT
        show_methods_differences(gt_path='Data/UCMerced_LandUse/test/tenniscourt98.tif',
                                 img_m1_path='Data/UCMerced_LandUse/test/tenniscourt98_fsrcnn_x2.tif',
                                 img_m2_path='output/msrn/tenniscourt98.tif', thr=3, step=1, save_folder='output/compare', methods='FSRCNN_MSRN')

    elif args.post_resampling: # resample outputs if needed
        scale = args.resample_scale
        post_resampling(gt_folder='Data/UCMerced_LandUse/testresampling_05/*.tif', est_folder='output/msrn_1to05/*.tif', scale=scale)
    elif args.compare_img_spectrum: # get graphics of Radially Averaged Fourier Spectrum metric
        plotSpectrums2D('output/fsrcnn_1to033_x3_blur/x3_*.tif',
                        'output/liif_UCMerced_1to033_blur/*.tif',
                        'output/msrn_1to033/*.tif', 'output/esrgan_1to033_x3_blur/*', 'output/fsrcnn_1to033_x3_blur/bicubic_x3*', 'output/compare/spectrum_033_new/')
    elif args.deblurring: # ADD DEBLURRING PRE-STAGE
        if args.mode in 'train':
            deblurring_train()
        elif args.mode in 'test':
            args_deblurring = Config_deblurring().get_args()
            deblurring_test(args_deblurring)
    else:
        if args.method in 'liif': # Use LIIF as SR method
            args_liif = Config_liif().get_args()
            if args.mode in 'train':
                training_liif(args_liif)
            elif args.mode in 'test':
                testing_liif(args_liif)
        elif args.method in 'fsrcnn': # Use FSRCNN as SR method
            args_srcnn = Config_srcnn().get_args()
            args_srcnn.images_dir = join(args.data_path, args.dataset)
            if args.mode in 'train':
                PrepareDataset(args_srcnn)
                training_fsrcnn(args_srcnn)
            elif args.mode in 'test':
                test_fsrcnn(args_srcnn)
        elif args.method in 'msrn': # Use MSRN as SR method
            if args.mode in 'train':
                args_msrn = Config_msrn().get_args()
                training_msrn(args_msrn)
            elif args.mode in 'test':
                args_msrn = Config_msrn().get_args()
                testing_msrn(args_msrn)
        elif args.method in 'esrgan': # Use ESRGAN as SR method
            if args.downsample:  # Pre processat de imatges per ESRGAN
                '''if args.mode in 'train':
                    downsampling_images(join(args.data_path, args.dataset, 'trainresampling_0.5'), 2)
                    downsampling_images(join(args.data_path, args.dataset, 'valresampling_0.5'), 2)
                elif args.mode in 'test':
                    downsampling_images(join(args.data_path, args.dataset, 'testresampling_05'), 2)'''
            args_esrgan = Config_esrgan().get_args()
            if args.mode in 'train':
                training_esrgan(args_esrgan)
            elif args.mode in 'test':
                test_esrgan(args_esrgan)


if __name__ == "__main__":
    print('START')
    main(Config().get_args())