import argparse

class Config:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--method', type=str, default='liif', choices=['liif', 'msrn', 'fsrcnn', 'esrgan'], help='Method to Super Resolve images')
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'val'])
        parser.add_argument('--split', type=bool, default=False, help='Boolean to decide if dataset should be '
                                                                      'splitted or it is already')
        parser.add_argument('--native_resolution', type=float, default=0.3, help='Native resolution of the images used')
        parser.add_argument('--downsample', type=bool, default=False, help='Boolean to decide if dataset is downsampled or not')
        parser.add_argument('--compare_img', type=bool, default=False)
        parser.add_argument('--compare_img_spectrum', type=bool, default=False)
        parser.add_argument('--resampling', type=bool, default=False, help='Boolean to decide if the dataset should '
                                                                           'be resampled to do some experiments')
        parser.add_argument('--post_resampling', type=bool, default=False,
                            help='Boolean to decide if the dataset should be resampled to do some experiments')
        parser.add_argument('--resample_scale', type=float, default=0.9)
        parser.add_argument('--add_blur', type=bool, default=False, help='Boolean to decide if we add blur')
        parser.add_argument('--kernel_size', type=int, default=3)

        parser.add_argument('--deblurring', type=float, default=False)

        # ================================ INPUT ================================ #
        parser.add_argument('--data_path', type=str, default='Data', help='Path where data is saved')
        parser.add_argument('--dataset', type=str, default='UCMerced_LandUse', choices=['NWPU_RESISC45','UCMerced_LandUse','Satellogic'], help='Dataset used to train and evaluate the model')
        parser.add_argument('--extension', type=str, default="jpg", choices=['jpg','png'], help="Extension of the "
                                                                                                "frame files")

        # ================================ OUTPUT ================================ #
        parser.add_argument('--output_path', type=str, default='output', help='Output path to save results')

        return parser.parse_args()