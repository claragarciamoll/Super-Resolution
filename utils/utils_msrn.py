import os
import sys
import math
import torch
import numpy as np
import cv2
import rasterio


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


def save_tif(path_out_samples, fname, res, target_resolution=0.7,
             name=None, name_id='MSRN07',
             channel_order_out='rgb',
             compress=False, nodata=0):
    """
    input is float normalized between 0 an 1.
    by default it taked input and rescale values to targert dtype (uint8).

    """

    if channel_order_out == 'rgb':
        res_tmp = res.copy()
        res = res[:, :, ::-1]

    if name is None:
        name = os.path.basename(fname).split('.')[0]

    if len(name_id) > 0:
        file_out = os.path.join(path_out_samples, f"TMP_{name}_{name_id}.tif")
    else:
        file_out = os.path.join(path_out_samples, f"TMP_{name}.tif")

    cmd = f"gdalwarp -tr {target_resolution} {target_resolution} \"{fname}\" \"{file_out}\""  # -r lanczos
    os.system(cmd)
    with rasterio.open(file_out, "r") as src:

        H_t, W_t = src.shape
        res = cv2.resize(res, (W_t, H_t), cv2.INTER_AREA)

        print(res.shape, src.read().shape)
        meta = src.meta.copy()
        meta['dtype'] = res.dtype.name
        meta['count'] = f'{res.shape[-1]}'
        meta['nodata'] = nodata

        if compress:
            meta['compress'] = 'lzw'

        with rasterio.open(file_out.replace('TMP_', ''), "w", **meta) as dst:
            dst.write(res.transpose(2, 0, 1))

    # remove tmp file
    os.system(f"rm {file_out}")
    return file_out.replace('TMP_', '')


############################ RASTERS ##########################################


def read_georaster_input(fname):
    """
    args:
        fname:   path to .tif

    returns:
        nimg:    numpy array H, W, C normalized to 0-1

    """

    with rasterio.open(fname, 'r') as src:
        nimg = src.read()[:3]
    nimg = nimg[::-1]  # BGR for network input (cv2 style)

    drange = np.iinfo(nimg.dtype.name).max
    # normalize to 0-1 range
    nimg = nimg.astype(np.float) / drange

    # H,W,C dim order
    nimg = nimg.transpose(1, 2, 0)
    return nimg


def get_mask(nimg):
    image = nimg.transpose(2, 0, 1).copy()
    mask = []
    C = image.shape[0]
    for i in range(C):
        mask.append(1 * (image[i] > 0))
    mask = np.array(mask)
    mask = np.sum(mask, 0)
    mask[mask != C] = 0
    mask[mask > 0] = 1
    return mask


def convert_float_uintX(nimg, dtype=np.uint16):
    nimg = nimg * (np.iinfo(dtype).max)
    nimg = nimg.astype(dtype)
    return nimg


def convert_uintX_float(nimg):
    dtype = nimg.dtype
    nimg = nimg.astype(np.float)
    nimg = nimg / np.iinfo(dtype).max
    return nimg
