import argparse

class Config_srcnn:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()

        # ================================ DIRECTORIES ================================ #
        parser.add_argument('--images_dir', type=str, default='Data') #, required=True
        parser.add_argument('--output_path', type=str, default='output') #, required=True
        parser.add_argument('--train_file', type=str,default='output_train.hdf5')
        parser.add_argument('--eval_file', type=str, default='output_eval.hdf5')
        parser.add_argument('--weights_file', type=str, default='output/weights_fsrcnn_x2/best.pth')
        parser.add_argument('--save_path', type=str, default='output/fsrcnn_1to05_x2_blur')
        parser.add_argument('--image_file', type=str, default='Data/UCMerced_LandUse/testresampling_05/*')

        # ================================ PARAMETERS ================================ #
        parser.add_argument('--scale', type=int, default=2)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--epoch_save', type=int, default=10)

        parser.add_argument('--with_aug', type=bool, default=False) #, action='store_true'
        parser.add_argument('--eval', type=bool, default=False) #, action='store_true'


        return parser.parse_args()