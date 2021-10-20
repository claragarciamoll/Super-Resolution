import argparse

class Config_deblurring:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

        parser.add_argument('--input_dir', default='Data/UCMerced_LandUse/train_deblurring/input', type=str, help='Directory of validation images')
        parser.add_argument('--result_dir', default='output/', type=str, help='Directory for results')
        parser.add_argument('--weights', default='output/deblurring_05/Deblurring/models/MPRNet/model_best.pth', type=str,
                            help='Path to weights')
        parser.add_argument('--dataset', default='UCMerced_LandUse', type=str,
                            help='Test Dataset')  # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
        parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

        return parser.parse_args()