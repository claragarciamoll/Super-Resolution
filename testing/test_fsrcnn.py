import argparse
import glob
from os.path import join

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models.model_fsrcnn import FSRCNN
from utils.utils_fsrcnn import convert_ycbcr_to_rgb, preprocess, calc_psnr, calc_ssim

def test_fsrcnn(args):
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    psnrs = []
    ssims = []
    for image_file in sorted(glob.glob(args.image_file)):

        image = pil_image.open(image_file).convert('RGB')

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        #bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
        bicubic.save(join(args.save_path, 'bicubic_x{}_'.format(args.scale) + image_file.split('/')[-1]))

        lr, _ = preprocess(lr, device)
        hr, _ = preprocess(hr, device)
        _, ycbcr = preprocess(bicubic, device)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)

        psnr = calc_psnr(hr, preds)
        ssim = calc_ssim(hr, preds)
        psnrs.append(psnr.detach().cpu().numpy())
        ssims.append(ssim)
        print(image_file)
        print('PSNR: {:.2f}'.format(psnr))
        print('SSIM: {:.2f}'.format(ssim))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(join(args.save_path, 'x{}_'.format(args.scale) + image_file.split('/')[-1]))

    mean_psnr = np.mean(psnrs)
    print('PSNR Mean: {:.2f}'.format(mean_psnr))
    mean_ssim = np.mean(ssims)
    print('SSIM Mean: {:.2f}'.format(mean_ssim))
    #output.save(args.image_file.replace('.', '_fsrcnn_x{}.'.format(args.scale)))