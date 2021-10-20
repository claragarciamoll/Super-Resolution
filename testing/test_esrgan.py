import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
from models.esrgan import RRDBNet_arch as arch
#from archs.esrgan import rrdbnet_arch as arch


def test_esrgan(args):

    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    # device = torch.device('cpu')

    test_img_folder = args.image_file

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    #model.load_state_dict(torch.load(args.model_path), strict=True)
    #print(model)
    weights = torch.load(args.model_path)
    #print(weights)
    model.load_state_dict(weights['params'])
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(args.model_path))

    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        if not osp.exists(args.save_path):
            os.makedirs(args.save_path)
        cv2.imwrite(osp.join(args.save_path, '{:s}_rlt.png'.format(base)), output)
