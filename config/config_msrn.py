import argparse


class Config_msrn:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="PyTorch MSRN")

        # ============================== CNN PARAMETERS ============================== #
        parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
        parser.add_argument("--nEpochs", type=int, default=1500, help="number of epochs to train for")
        parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=1e-3")
        parser.add_argument("--colorjitter", default=False, help="Use colorjitter?")
        parser.add_argument("--add_noise", default=True, help="Use cuda?")
        parser.add_argument("--vgg_loss", default=True, help="Use perceptual loss?")
        parser.add_argument("--threads", type=int, default=1,
                            help="Number of threads for data loader to use, Default: 1")
        parser.add_argument("--start_epoch", type=int, default=1)
        parser.add_argument("--step", type=int, default=50,
                            help="Sets the learning rate to the initial LR decayed by momentum every n epochs, "
                                 "Default: n=500")
        parser.add_argument("--seed", default="1", type=str, help="random seed")
        parser.add_argument('--wind_size', type=int, default=64)

        # ============================ HARDWARE PARAMETERS ============================ #
        parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
        parser.add_argument("--resume", default=False, help="take last epoch available")
        parser.add_argument("--cuda", default=True, help="Use cuda?")
        parser.add_argument('--gpu_device', type=str, default='0')

        # ================================ DIRECTORIES ================================ #
        parser.add_argument("--path_out", default="output/msrn_1to05/", type=str,
                            help="path output")
        parser.add_argument('--path_to_model_weights', type=str, default='output/msrn_1to05/checkpoint/model_epoch_1500.pth')
        parser.add_argument('--filename', type=str, default='Data/UCMerced_LandUse/testresampling_05/*')

        return parser.parse_args()
