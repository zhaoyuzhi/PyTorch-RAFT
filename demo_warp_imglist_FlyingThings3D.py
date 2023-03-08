import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def create_raft(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    return model

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


def demo_warp(img1, flo):
    img2_warp = warp(img1, flo)
    img2_warp = img2_warp[0].permute(1,2,0).cpu().numpy()
    return img2_warp


def demo(args, model, imfile1, imfile2):
    with torch.no_grad():
        
        readpath1 = os.path.join(args.read_path, imfile1)
        readpath2 = os.path.join(args.read_path, imfile2)
        savepath = os.path.join(args.save_path, imfile1.replace('.', '_warp.'))
        image1 = load_image(readpath1)
        image2 = load_image(readpath2)
        assert image1.shape == image2.shape
        _, _, h, w = image1.shape

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        image2_warp = demo_warp(image1, flow_up)

        # convert to opencv format for saving
        image2_warp = (image2_warp).astype(np.uint8)
        image2_warp = cv2.resize(image2_warp, (w, h))
        image2_warp = cv2.cvtColor(image2_warp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepath, image2_warp)


# read a txt expect EOF
def text_readlines(filename):
    # try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content


# multi-layer folder
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--txt_path', default='txt/FlyingThings3D_subset_train_split.txt', help="image name list")
    parser.add_argument('--read_path', default='F:\\FlyingThings3D_subset\\train\\image_clean\\left', help="path including all input frames")
    parser.add_argument('--save_path', default='F:\\FlyingThings3D_subset\\train\\image_clean\\left_warp', help="path for saving frames")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # build pathlist for RAFT to compute optical flows and warp
    split_image_list = text_readlines(args.txt_path)
    pathlist = []
    for i in range(len(split_image_list)):
        cur_split = split_image_list[i]
        cur_split = cur_split.split(' ')
        print(cur_split)
        for j in range(len(cur_split)-1):
            temp_list = []
            prev = cur_split[j]
            next = cur_split[j+1]
            temp_list.append(prev)
            temp_list.append(next)
            pathlist.append(temp_list)

    check_path(args.save_path)

    # create model and perform warpping
    model = create_raft(args)

    for source_path, target_path in pathlist:
        print(source_path, target_path)
        demo(args, model, source_path, target_path)
    