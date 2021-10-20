import argparse, os
import sys
import torch
import math, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

from utils.utils_msrn import PSNR
from utils.utils_msrn import SSIM
from models.msrn.msrn import MSRN_Upscale
from datasets.msrn import DatasetHR_LR
from torch.utils.data import DataLoader
from models.msrn.perceptual_loss import VGGPerceptualLoss


class noiseLayer_normal(nn.Module):
    def __init__(self, noise_percentage, mean=0, std=0.2):
        super(noiseLayer_normal, self).__init__()
        self.n_scale = noise_percentage
        self.mean=mean
        self.std=std

    def forward(self, x):
        if self.training:
            noise_tensor = torch.normal(self.mean, self.std, size=x.size()).to(x.get_device()) 
            x = x + noise_tensor * self.n_scale
        
            mask_high = (x > 1.0)
            mask_neg = (x < 0.0)
            x[mask_high] = 1
            x[mask_neg] = 0

        return x

def training_msrn(args):

    global opt, model
    opt = args
    os.makedirs(opt.path_out, exist_ok=True)
    writer = SummaryWriter(opt.path_out)

    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)


    print("===> Loading datasets")
    train_set = DatasetHR_LR("training", apply_color_jitter=opt.colorjitter)
    validation_set = DatasetHR_LR("validation")

    dataloaders ={
        'training': DataLoader(dataset=train_set, num_workers=opt.threads, \
                        batch_size=opt.batchSize, shuffle=True),
        'validation': DataLoader(dataset=validation_set, num_workers=opt.threads, \
            batch_size=opt.batchSize, shuffle=False)}

    print("===> Building model")
    model = MSRN_Upscale(n_scale=2)
    # pretrained
    criterion = nn.L1Loss(reduction='none')
    
    print("INIT PIXEL SHUFFLE!!")
    model._init_pixel_shuffle()
    
    if opt.vgg_loss:
        global perceptual_loss

        perceptual_loss = VGGPerceptualLoss()
        perceptual_loss.eval()
        perceptual_loss.cuda()
        print("Using perceptual loss")
    
    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        
    # optionally resume from a checkpoint
    if opt.resume:
        path_checkpoints = os.path.join(opt.path_out, "checkpoint/")
        list_epochs = [int(f.split('.')[0].split('_')[-1]) for f in os.listdir(path_checkpoints)]
        list_epochs.sort()
        last_epoch = list_epochs[-1]
        print(" resume from ", last_epoch)
        weights = torch.load(os.path.join(path_checkpoints, f"model_epoch_{last_epoch}.pth"))
        model.load_state_dict(weights["model"].state_dict())
        opt.start_epoch = weights["epoch"] + 1
        
    
#     for name,param in model.named_parameters():
#         param.requires_grad = False
#         if 'conv_up' in name:
#             param.requires_grad = True
#         if 'conv_output' in name:
#             param.requires_grad = True
            
    for name,param in model.named_parameters():
        print(name, param.requires_grad)
    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        for mode in ['training', 'validation']:
            train(mode, dataloaders, optimizer, model, criterion, epoch, writer)
            if mode=='training':
                save_checkpoint(model, epoch, opt.path_out)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(mode, dataloader, optimizer, model, criterion, epoch, writer):
    
    metric_psnr = PSNR()
    metric_ssim = SSIM()

#     lr = adjust_learning_rate(optimizer, epoch-1)
#     print("learning rate", lr)
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr

    print("{}\t Epoch={}, lr={}".format(mode, epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    if mode=='validation':
        model.eval()

    for iteration, batch in enumerate(dataloader[mode], 1):

        img_lr, img_hr = batch
        
        if opt.cuda:
            img_lr = img_lr.cuda()
            img_hr = img_hr.cuda()
            
        if opt.add_noise:
            scale_noise = np.random.choice(np.arange(0.05, 0.2, 0.01))
            add_noise = noiseLayer_normal(scale_noise, mean=0, std=0.2)
            img_lr = add_noise(img_lr)

        output = model(img_lr)
        loss_spatial = criterion(img_hr, output)
        loss = torch.mean(loss_spatial)
        
        if opt.vgg_loss:
            vgg_loss,_ = perceptual_loss(output, img_hr)
            loss = loss + 10*vgg_loss

        psnr = metric_psnr(img_hr, output)
        ssim = metric_ssim(img_hr, output)
        
        if mode=='training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        grid_lr = torchvision.utils.make_grid(img_lr)[[2, 1, 0],...]
        grid_hr = torchvision.utils.make_grid(img_hr)[[2, 1, 0],...]
        grid_pred = torchvision.utils.make_grid(output)[[2, 1, 0],...]

        if iteration%10 == 0:
            print("===>{}\tEpoch[{}]({}/{}): Loss: {:.5} \t PSNR: {:.5} \t SSIM: {:.5}".format(mode, epoch, iteration, len(dataloader[mode]), loss.item(), psnr.item(), ssim.item()))
            writer.add_scalar(f'{mode}/LOSS/', loss.item(), epoch*len(dataloader[mode])+iteration)
            writer.add_scalar(f'{mode}/PSNR', psnr.item(), epoch*len(dataloader[mode])+iteration)
            writer.add_scalar(f'{mode}/SSIM', ssim.item(), epoch * len(dataloader[mode]) + iteration)
            writer.add_image(f'{mode}/lr', grid_lr, iteration)
            writer.add_image(f'{mode}/hr', grid_hr, iteration)
            writer.add_image(f'{mode}/pred', grid_pred, iteration)
            
    

def save_checkpoint(model, epoch, path_out):
    path_out = os.path.join(path_out, "checkpoint/")         
    os.makedirs(path_out, exist_ok=True)
    model_out_path = path_out + f"model_epoch_{epoch}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

