import argparse


class Config_liif:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--config', default='config/train_liif.yaml')
        parser.add_argument('--name', default='liif_UCMerced')
        parser.add_argument('--tag', default=None)
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--model', default='output/liif_UCMerced_1to05_blur/epoch-best.pth')

        return parser.parse_args()
