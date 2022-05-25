# DisARM: Displacement Aware Relation Module for 3D Detection

## Abstract

We introduce Displacement Aware Relation Module(DisARM), a novel neural network module for enhancing the performance of 3D object detection in point cloud scenes. The core idea is extracting the most principal contextual information is critical for detection while the target is incomplete or featureless. We find that relations between proposals provide a good representation to describe
the context. However, adopting relations between all the object or patch proposals for detection is inefficient, and an imbalanced combination of local and global relations brings extra noise that could mislead the training. Rather than working with all relations, we find that training with relations only between the most representative ones, or anchors, can significantly boost the detection performance. Good anchors should be semantic-aware with no ambiguity and able to describe the whole layout of a scene with no redundancy. To find the anchors, we first perform a preliminary relation anchor module with an objectness-aware sampling approach and then devise a displacement based
module for weighing the relation importance for better utilization of contextual information. This light-weight relation module leads to significantly higher accuracy of object instance detection when being plugged into the state-of-the-art detectors. Evaluations on the public benchmarks of
real-world scenes show that our method achieves the state-of-the-art performance on both SUN RGB-D and ScanNet V2.

![teaser](resources/teaser_disarm.jpg)

## Introduction

This repo is the official implementation of ["DisARM: Displacement Aware Relation Module for 3D Detection"](https://arxiv.org/abs/2203.01152).

Authors: [Yao Duan](https://yaraduan.github.io), [Chenyang Zhu](http://www.zhuchenyang.net/), [Yuqing Lan](), [Renjiao Yi](https://renjiaoyi.github.io/), [Xinwang Liu](https://xinwangliu.github.io/), [Kai Xu](http://kevinkaixu.net/index.html)*.

In this repository, we provide model implementation (with MMDetection3D V0.17.1) as well as training scripts on ScanNet and SUN RGB-D.

**Note:**

We also will fork the `MMDetection3D` project and merge the DisAMR module to the master branch. If you want to follow the newest version, please look forward to the offical repository of MMDetection3D in the coming weeks.
<!-- refer to the [offical repository](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/disarm). -->


## Results and models

### ScanNet V2

|Method | mAP@0.25 | mAP@0.5 |
|:---|:---:|:---:|
|[VoteNet](https://arxiv.org/abs/1904.09664)       | 58.6 | 33.5 | 
|[VoteNet](https://arxiv.org/abs/1904.09664)+DisARM| 66.1 | 49.7 | 
|[BRNet](https://arxiv.org/abs/1904.09664)         | 66.1 | 50.9 | 
|[BRNet](https://arxiv.org/abs/1904.09664)+DisARM  | 66.7 | 42.3 |
|[H3DNet*](https://arxiv.org/abs/2006.05682)       | 66.4 | 48.0 | 
|[H3DNet*](https://arxiv.org/abs/2006.05682)+DisARM | 66.8 | 48.8 | 
|[GroupFree3D*](https://arxiv.org/abs/2006.05682)(L6,0256) | 66.3 | 47.8 | 
|[GroupFree3D*](https://arxiv.org/abs/2006.05682)(L6,0256)+DisARM | 67.0 | 50.7 | 
|[GroupFree3D*](https://arxiv.org/abs/2006.05682)(L12,0256) | 66.6 | 48.2 | 
|[GroupFree3D*](https://arxiv.org/abs/2006.05682)(L12,0256)+DisARM | 67.2 | 52.5 | 
|[GroupFree3D*](https://arxiv.org/abs/2006.05682)(w2×,L12,0256) | 68.2 | 52.6 |
|[GroupFree3D*](https://arxiv.org/abs/2006.05682)(w2×,L12,0256)+DisARM | 69.3 | 53.6 | 


### SUN RGB-D

|Method | mAP@0.25 | mAP@0.5 |
|:---|:---:|:---:|
|[VoteNet](https://arxiv.org/abs/1904.09664)       | 57.7 | 35.8 |
|[VoteNet](https://arxiv.org/abs/1904.09664)+DisARM| 61.5 | 41.3 | 
|[imVoteNet](https://arxiv.org/abs/2001.10692)*| 64.0 | - |  
|[imVoteNet](https://arxiv.org/abs/2001.10692)*+DisARM| 65.3 | - | 
**Notes:**

-  We use one NVIDIA GeForce RTX 3090 GPU for training GroupFree3D+DisARM and one NVIDIA TITAN V GPU for others. 
-  We report the best results on validation set during each training. 
-  \* denotes that the model is implemented on MMDetection3D.

## Install

This repo is built based on [MMDetection3D]()(V0.17.1), please follow the [getting_started.md](docs/getting_started.md) for installation.

The code is tested under the following environment:

- `Ubuntu 16.04 LTS`
- `Anaconda` with `python=3.7.10`
- `pytorch 1.9.0`
- `cuda 11.1`
- `GCC 5.4`

**Notes:**

If you want to test on `BRNet+DisARM`, please follow the [getting_started.md](./BRNet+DisARM/docs/getting_started.md) to install the dependences under `./BRNet+DisARM` for the reason that BRNet is implemented on MMDetection3D V0.11.0. 

## Data preparation

For SUN RGB-D, follow the [README](./mmdetection/data/sunrgbd/README.md) under the `/data/sunrgbd` folder.

For ScanNet, follow the [README](./mmdetection/data/scannet/README.md) under the `/data/scannet` folder.

**Notes:**

For `BRNet+DisARM`, please follow the instruction under `./BRNet+DisARM/data/` to process the data for training and testing. 

## Usage

Using DisARM for your own detectors, please follow the steps:

1.copy `./mmdetection3d/mmdet3d/models/model_utils/disarm.py` to your project

2.import `DisARM` module and input the proposal features and locations to the module

3.add relation_anchor_loss in your file

4.config DisARM as bellow:

```bash
disarm_module_cfg=dict(
    sample_approach='OS-FFPS', 
    num_anchors=15,  
    num_candidate_anchors=64, 
    in_channels=YOUR_PROPOSAL_FEATURE_DIM,  
),
relation_anchor_loss=dict(
    type='VarifocalLoss',
    use_sigmoid=True,
    reduction='sum',
    loss_weight=1.0
),
```

5.add the returned relation features to your proposal features

## Training 

### ScanNet

For `VoteNet+DisARM` training, please go to the `mmdetection` dir and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/disarm/votenet_disarm_scannet.py
```

For `BRNet+DisARM` training, please go to the `brnet_diarm` dir and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/disarm/brnet_disarm_scannet.py --seed 42
```

For `H3DNet+DisARM` training, please go to the `mmdetection` dir and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/disarm/h3dnet_disarm_scannet.py
```

For `GroupFree3D+DisARM` training, please go to the `mmdetection` dir and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/disarm/groupfree3d-L6-O256_disarm_scannet.py
```

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/disarm/groupfree3d-L12-O256_disarm_scannet.py
```

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/disarm/groupfree3d-L12-O256_disarm_scannet.py
```

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/disarm/groupfree3d-L12-O256_disarm_scannet.py
```

#### SUN RGB-D

For `VoteNet+DisARM` training, please go to the `mmdetection` dir and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/disarm/votenet_disarm_sunrgbd.py
```

For `imVoteNet+DisARM` training, please go to the `mmdetection` dir and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/disarm/imvotenet_disarm_sunrgbd.py
```

## Citation

```
@article{duan2022disarm,
      title={DisARM: Displacement Aware Relation Module for 3D Detection}, 
      author={Yao Duan and Chenyang Zhu and Yuqing Lan and Renjiao Yi and Xinwang Liu and Kai Xu},
      year={2022},
      eprint={2203.01152},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

We thank a lot for the flexible codebase of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)and [BRNet](https://github.com/cheng052/BRNet).

## License

The code is released under MIT License (see LICENSE file for details).