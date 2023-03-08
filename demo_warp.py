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

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def warp(x, flo):
    """
    warp an image/tensor according to the optical flow
    x: [B, C, H, W] image/tensor
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


def warp_cv2(x, flo):
    """
    warp an image/tensor according to the optical flow
    x: [H, W, C] image/tensor
    flo: [H, W, 2] flow
    """
    # calculate mat
    w = int(x.shape[1])
    h = int(x.shape[0])
    flo = np.float32(flo)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    pixel_map = coords + flo
    output = cv2.remap(x, pixel_map, None, cv2.INTER_LINEAR)
    return output


def viz_warp(img1, img2, flo):
    img2_warp = warp(img1, flo)

    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    img2_warp = img2_warp[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_concat1 = np.concatenate([img1, flo], axis=0)
    img_concat2 = np.concatenate([img2, img2_warp], axis=0)
    img_concat3 = np.concatenate([(img1 + img2) / 2, img2_warp], axis=0)
    img_concat = np.concatenate([img_concat1, img_concat2, img_concat3], axis=1)

    import matplotlib.pyplot as plt
    plt.imshow(img_concat / 255.0)
    plt.show()

    # cv2.imshow('image', img_concat[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    return img2_warp


def viz_warp_cv2(img1, img2, flo):

    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    img2_warp = warp_cv2(img1, flo)
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_concat1 = np.concatenate([img1, flo], axis=0)
    img_concat2 = np.concatenate([img2, img2_warp], axis=0)
    img_concat3 = np.concatenate([(img1 + img2) / 2, img2_warp], axis=0)
    img_concat = np.concatenate([img_concat1, img_concat2, img_concat3], axis=1)

    import matplotlib.pyplot as plt
    plt.imshow(img_concat / 255.0)
    plt.show()

    # cv2.imshow('image', img_concat[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    return img2_warp


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        imfile1 = args.path1
        imfile2 = args.path2

        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        assert image1.shape == image2.shape
        _, _, h, w = image1.shape

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        image2_warp = viz_warp(image1, image2, flow_up)
        #image2_warp = viz_warp_cv2(image1, image2, flow_up)

        '''
        # convert to opencv format for saving
        image2_warp = (image2_warp).astype(np.uint8)
        image2_warp = cv2.resize(image2_warp, (w, h))
        image2_warp = cv2.cvtColor(image2_warp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imfile1.replace('.', '_warp.'), image2_warp)
        '''


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--path1', default='demo-frames/frame_0016.png', help="dataset for evaluation")
    parser.add_argument('--path2', default='demo-frames/frame_0017.png', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
    