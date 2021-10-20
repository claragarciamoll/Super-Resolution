import argparse

class Config_esrgan:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()

        # ================================ DIRECTORIES ================================ #
        parser.add_argument('--images_dir', type=str, default='Data') #, required=True
        parser.add_argument('--output_path', type=str, default='output') #, required=True
        parser.add_argument('--opt', type=str, default='config/train_esrgan.yaml', help='Path to option YAML file.')
        parser.add_argument('--model_path', type=str, default='../config/experiments/ESRGAN_x2_f64b16_UCML_1000k_B16G1_wandb/models/net_g_latest.pth')
        parser.add_argument('--save_path', type=str, default='output/esrgan')
        parser.add_argument('--image_file', type=str, default='Data/UCMerced_LandUse/testresampling_05_downx2/*')

        parser.add_argument('--auto_resume', type=bool, default=True)
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')

        return parser.parse_args()