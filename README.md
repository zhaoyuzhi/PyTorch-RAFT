# PyTorch-RAFT

This repository contains the source code for:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## Requirements

- pytorch>=1.6
- torchvision
- cudatoolkit
- matplotlib
- tensorboard
- scipy
- opencv-python

## Pre-trained models

Pretrained models can be downloaded by running:
```Shell
./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

## Demos

You can demo a trained model on a sequence of frames:
```Shell
python demo.py --model=models/raft-things.pth --path=demo-frames
```

The default mode is **forward warping**, which is implemented by the following code:
```bash
python demo_warp.py --model=models/raft-things.pth --path1=demo-frames/frame_0016.png --path2=demo-frames/frame_0017.png
```
(i.e., it warps the PATH1 image to the position of PATH2 image)

BTW, if we want to implement **backward warping**, simply changing PATH1 and PATH2 as:
```bash
python demo_warp.py --model=models/raft-things.pth --path1=demo-frames/frame_0017.png --path2=demo-frames/frame_0016.png
```
(i.e., the source image is from PATH1)

If you have a bunch of image pairs to warp, run the following code:
```bash
python demo_warp_imglist.py
```

## Required Data

To evaluate/train RAFT, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)

By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Evaluation

You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training

We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```

If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation

You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.
