import argparse
import os
import math
import cv2
import numpy as np
from functools import partial
from PIL import Image

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.liif import *
from models.liif import models
import utils.utils_liif as utils


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, test_plot=True,
              verbose=False):
    model.eval()
    print(eval_type)
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()
    val_ssim = utils.Averager() #test

    pbar = tqdm(loader, leave=False, desc='val')
    count = 0
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp,
                                   batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if test_plot is True:
            ih, iw = batch['inp'].shape[-2:]
            #print(ih)
            #print(iw)
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [round(ih * s), round(iw * s), 3]
            #print(shape)
            #print(pred.shape)
            pred_plot = pred.view(*shape)#.permute(0, 3, 1, 2).contiguous()
            img = pred_plot.cpu().detach().numpy()
            img = (img*255).astype(np.uint8)
            #print(img.shape)
            img = Image.fromarray(img)
            img.save('output/liif_UCMerced_1to05_blur/' + str(count) + '.tif')
            count = count + 1

        if eval_type is not None:  # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        #val_ssim = utils.calc_ssim(pred, batch['gt']) #train
        ssim = utils.calc_ssim(pred, batch['gt']) #test
        val_ssim.add(ssim.item(), inp.shape[0]) #test

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item(), val_ssim

def testing_liif(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res, ssim = eval_psnr(loader, model,
                    data_norm=config.get('data_norm'),
                    eval_type=config.get('eval_type'),
                    eval_bsize=config.get('eval_bsize'),
                    test_plot=config.get('test_plot'),
                    verbose=True)
    print('result_psnr: {:.4f}'.format(res))
    print('result_ssim: {:.4f}'.format(ssim))
