import cv2
import glob
import os
import sys
import time
import math
import numpy as np
import kornia
from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from utils.utils_msrn import (read_georaster_input,
                              save_tif,
                              get_mask,
                              convert_float_uintX,
                              convert_uintX_float)


class WindowsDataset_SR(data.Dataset):
    def __init__(self, nimg, wind_size=120, stride=480, scale=2):

        self.nimg = nimg
        H, W, C = nimg.shape
        self.H = H
        self.W = W
        self.C = C
        self.scale = scale

        self.H_out = H * scale
        self.W_out = W * scale

        self.stride = stride
        self.wind_size = wind_size

        # get all locations
        self.coordinates_input = []
        for j in range(0, self.H, self.stride):
            for i in range(0, self.W, self.stride):
                y0 = j
                y1 = min(j + self.wind_size, self.H)
                x0 = i
                x1 = min(i + self.wind_size, self.W)

                self.coordinates_input.append((y0, y1, x0, x1))

    def __len__(self):
        return len(self.coordinates_input)

    def __getitem__(self, index):
        y0, y1, x0, x1 = self.coordinates_input[index]
        w = x1 - x0
        h = y1 - y0
        crop = np.zeros((self.wind_size, self.wind_size, 3))

        crop[:h, :w] = self.nimg[y0:y1, x0:x1]

        # normalization
        crop = crop
        x_in = kornia.image_to_tensor(crop).float()

        ###### AFEGIT
        random_crop = kornia.augmentation.RandomCrop((crop.shape[0], crop.shape[1]), )
        sigma = 0.5 * self.scale
        kernel_size = math.ceil(sigma * 3 + 4)
        img_crop = random_crop(x_in)
        kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
        x_in = kornia.filter2D(img_crop, kernel_tensor[None])[0]
        print('X_in shape: ' + str(x_in.shape))
        ###### ######

        
        sample = {
            'x_in': x_in,
            'coordinates': np.array(self.coordinates_input[index]),
            'w': w,
            'h': h
        }
        return sample


def inference_model(model, nimg, wind_size=512, stride=480, scale=2,
                    batch_size=1, data_parallel=False, padding=5, manager=None):
    """
    Run sliding window on data using the sisr model.

    args:
        model
        data (H,W,C) in BGR format normalized between 0-1 (float)
        wind_size
        stride
    returns:
        super resolved image xscale. Numpy array (H,W,C) BGR 0-1 (float)
    """

    # get device
    for p in model.parameters():
        device = p.get_device()
        break
    if device == -1:
        device = 'cpu'
    elif data_parallel:
        print("Using multiple GPU!")
        model = nn.DataParallel(model).cuda()

    H, W, C = nimg.shape

    # init dataset
    dataset = WindowsDataset_SR(nimg, wind_size, stride, scale)
    print(dataset)

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    output = np.zeros((dataset.H_out, dataset.W_out, C)).astype(np.float)
    counts = np.zeros((dataset.H_out, dataset.W_out, C)).astype(np.float)

    psteps = tqdm(total=len(dataloader), desc='\tSI-AI inference', position=0)

    if manager is not None:
        psteps = manager.counter(total=len(dataloader), desc='\tSI-AI inference', unit='steps')

    for sample in dataloader:

        if not data_parallel:
            x_in = sample['x_in'].to(device)
        else:
            x_in = sample['x_in'].cuda()

        # add 5 pixel padding to avoid border effect
        x_in = F.pad(input=x_in, pad=(padding, padding, padding, padding), mode='reflect')

        pred_sate = model(x_in)
        pred_sate = pred_sate.detach().data.cpu().numpy()
        pred_sate = pred_sate[:, :,
                    scale * padding:-scale * padding,
                    scale * padding:-scale * padding]
        for ii in np.arange(pred_sate.shape[0]):
            pred_sample = pred_sate[ii].transpose((1, 2, 0))

            pred_sample = (np.clip((pred_sample), 0, 1))
            y0, y1, x0, x1 = sample['coordinates'][ii]
            h = sample['h'][ii].item()
            w = sample['w'][ii].item()
            Y0 = y0 * scale
            Y1 = y1 * scale
            X0 = x0 * scale
            X1 = x1 * scale
            hh, ww, cc = output[Y0:Y1, X0:X1].shape
            print('output: ', output[Y0:Y1, X0:X1].shape)
            if (hh < scale * wind_size) or (ww < scale * wind_size):
                Y1 = Y0 + hh
                X1 = X0 + ww
                pred_sample = pred_sample[:hh, :ww]
            print('OUTPUT SHAPE: ' + str(output.shape))
            print('PRED SAMPLE SHAPE: ' + str(pred_sample.shape))
            output[Y0:Y1, X0:X1] += pred_sample[...]
            counts[Y0:Y1, X0:X1, :] += 1
            psteps.update()

    res = np.divide(output, counts)
    return res


def load_msrn_model(weights_path=None, cuda='0'):
    """
    Load MSRN traied model on specific GPU for inference

    args:
        path to weights (default model_epoch_101.pth located in Nas) x2 scale
        cuda '0' or set to None if yoy want CPU usage

    return:
        pytorch MSRN
    """
    if cuda is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    model = MSRN_Upscale(n_scale=2)

    if weights_path is None:
        weights_path = "model.pth"

    if not torch.cuda.is_available():
        weights = torch.load(weights_path, map_location=torch.device('cpu'))
    else:
        weights = torch.load(weights_path)

    '''print(torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    print(torch.cuda.device_count())'''

    model.load_state_dict(weights['model'].state_dict())
    model.eval()

    if cuda is not None:
        model.cuda()

    print("Loaded MSRN ", weights_path)
    return model


# residual module
class MSRB(nn.Module):
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()
        self.n_feats = n_feats
        self.conv3_1 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=5, stride=1, padding=2)
        self.conv3_2 = nn.Conv2d(2 * self.n_feats, 2 * self.n_feats, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(2 * self.n_feats, 2 * self.n_feats, kernel_size=5, stride=1, padding=2)
        self.conv1_3 = nn.Conv2d(4 * self.n_feats, self.n_feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_input = x.clone()

        x3_1 = self.relu(self.conv3_1(x))
        x5_1 = self.relu(self.conv5_1(x))
        x1 = torch.cat([x3_1, x5_1], 1)

        x3_2 = self.relu(self.conv3_2(x1))
        x5_2 = self.relu(self.conv5_2(x1))
        x2 = torch.cat([x3_2, x5_2], 1)

        x_final = self.conv1_3(x2)
        return x_final + x_input


def ICNR(tensor, scale_factor=2, initializer=nn.init.kaiming_normal_):
    print('Tensor shape: ' + str(tensor.shape))
    OUT, IN, H, W = tensor.shape
    sub = torch.zeros(math.ceil(OUT / scale_factor ** 2), IN, H, W)
    sub = initializer(sub)
    print('Sub shape: ' + str(sub.shape))
    kernel = torch.zeros_like(tensor)
    for i in range(OUT):
        kernel[i] = sub[i // scale_factor ** 2]

    return kernel

# Full structure
class MSRN_Upscale(nn.Module):
    def __init__(self, n_input_channels=3, n_blocks=8, n_feats=64, n_scale=4):
        super(MSRN_Upscale, self).__init__()

        self.n_blocks = n_blocks
        self.n_feats = n_feats
        self.n_scale = n_scale
        self.n_input_channels = n_input_channels

        # input
        self.conv_input = nn.Conv2d(self.n_input_channels, self.n_feats, kernel_size=3, stride=1, padding=1)

        # body
        conv_blocks = []
        for i in range(self.n_blocks):
            conv_blocks.append(MSRB(self.n_feats))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.bottle_neck = nn.Conv2d((self.n_blocks + 1) * self.n_feats, self.n_feats, kernel_size=1, stride=1,
                                     padding=0)

        # tail
        self.conv_up = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1, bias=1)
        self.pixel_shuffle = nn.Upsample(scale_factor=self.n_scale, mode='nearest')
        self.conv_output = nn.Conv2d(self.n_feats, n_input_channels, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def _init_pixel_shuffle(self):
        kernel = ICNR(self.conv_up.weight, scale_factor=self.n_scale)
        self.conv_up.weight.data.copy_(kernel)

    def forward(self, x):
        x_input = x.clone()

        features = []

        # M0
        x = self.conv_input(x)
        features.append(x)

        # body
        for i in range(self.n_blocks):
            x = self.conv_blocks[i](x)
            features.append(x)

        x = torch.cat(features, 1)

        x = self.bottle_neck(x)

        x = self.conv_up(x)
        x = self.pixel_shuffle(x)
        x = self.conv_output(x)

        return x

    # def


def process_file(model, path_out, compress=True, res_output=0.7,
                 wind_size=512, stride=480, batch_size=1, padding=5, manager=None, filename=None):
    nimg = read_georaster_input(filename)
    dtype_input = 'uint8'
    channel_order_out = 'rgb'
    fout = None

    if nimg is not None:
        mask = get_mask(nimg)
        nimg[mask == 0] = 0

        nimg = cv2.copyMakeBorder(nimg, padding, padding, padding, padding,
                                  cv2.BORDER_REPLICATE)

        # inference
        result = inference_model(model, nimg,
                                 wind_size=wind_size, stride=stride,
                                 scale=2, batch_size=batch_size, manager=manager)

        result = result[2 * padding:-2 * padding, 2 * padding:-2 * padding]
        result = cv2.convertScaleAbs(result, alpha=np.iinfo(np.uint8).max)

        H, W = result.shape[:2]

        if mask is not None:
            # resize mask to fit SR
            mask = cv2.resize((mask * 255).astype(np.uint8), (W, H), cv2.INTER_NEAREST)
            result = result.transpose(2, 0, 1)
            for i in range(result.shape[0]):
                result[i] = np.where((result[i] == 0) & (mask > 0), 1, result[i])
            result = result.transpose(1, 2, 0)

        '''name = os.path.basename(filename).split('.')[0]
        print(path_out)
        print(filename)
        print(result)
        fout = save_tif(path_out, filename, result, target_resolution=res_output,
                        name=name, name_id='sr',
                        channel_order_out='rgb', compress=compress)'''
    return result


def testing_msrn(args):
    opt = args
    # Load model
    model = load_msrn_model(weights_path=opt.path_to_model_weights, cuda=opt.gpu_device)  # use first GPU available

    # Path out
    path_out = os.path.dirname(opt.filename)

    print(path_out)

    for image_file in sorted(glob.glob(opt.filename)):
        print('Image file: ' + image_file)

        t00 = time.time()
        fout = process_file(model, path_out, compress=True, res_output=0.7,
                            wind_size=opt.wind_size, stride=opt.wind_size - 10,
                            batch_size=1, padding=5, filename=image_file)

        name = os.path.basename(image_file).split('.')[0]
        W_t, H_t, _ = fout.shape

        res = cv2.resize(fout, (int(W_t/3), int(H_t/3)), cv2.INTER_AREA)
        cv2.imwrite(os.path.join(opt.path_out, name + '.tif'), res)

        t01 = time.time()
        print(t01 - t00, "fout: ", fout)
        print("DONE ", )
