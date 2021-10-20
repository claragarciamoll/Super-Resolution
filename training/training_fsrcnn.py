import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models.model_fsrcnn import FSRCNN
from datasets.srcnn import TrainDataset, EvalDataset
from utils.utils_fsrcnn import *

def training_fsrcnn(args):
    global log, writer

    args.output_path = os.path.join(args.output_path, 'x{}'.format(args.scale))
    log, writer = set_save_path(args.save_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = FSRCNN(scale_factor=args.scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': args.learning_rate * 0.1}
    ], lr=args.learning_rate)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    epoch_save = args.epoch_save
    timer = Timer()
    epoch_max = args.epochs

    for epoch in range(args.epochs):
        t_epoch_start = timer.t()
        print(epoch)
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.epochs - 1))
            log_info = ['epoch {}/{}'.format(epoch, args.epochs - 1)]

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                log_info.append('train: loss={:.4f}'.format(epoch_losses.avg))
                writer.add_scalars('loss', {'train': epoch_losses.avg}, epoch)

        torch.save(model.state_dict(), os.path.join(args.output_path, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(calc_ssim(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        log_info.append('val: psnr={:.4f}'.format(epoch_psnr.avg))
        writer.add_scalars('psnr', {'val': epoch_psnr.avg}, epoch)

        print('eval ssim: {:.2f}'.format(epoch_ssim.avg))
        log_info.append('val: ssim={:.4f}'.format(epoch_ssim.avg))
        writer.add_scalars('ssim', {'val': epoch_ssim.avg}, epoch)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        t = timer.t()
        prog = (epoch - t_epoch_start + 1) / (epoch_max - t_epoch_start + 1)
        t_epoch = time_text(t - t_epoch_start)
        t_elapsed, t_all = time_text(t), time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.output_path, 'best.pth'))