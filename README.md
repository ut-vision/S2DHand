Single-to-Dual-View Adaptation for Egocentric 3D Hand Pose Estimation
---
[![arXiv](https://img.shields.io/badge/arXiv-2403.04381-DodgerBlue.svg?style=plastic)](https://arxiv.org/pdf/2403.04381.pdf)
![Python 3.10](https://img.shields.io/badge/python-3.10-DodgerBlue.svg?style=plastic)
![Pytorch 1.13.0](https://img.shields.io/badge/pytorch-1.13.0-DodgerBlue.svg?style=plastic)
![CUDA 11.7](https://img.shields.io/badge/cuda-11.7-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

Our paper is accepted by **CVPR-2024**

<div align=center>  <img src="figures/teaser.png" alt="Teaser" width="500" align="bottom" /> </div>

**Picture:**  *Overview of the proposed Unsupervised 1-to-2 Views Adaption framework for adapting a single-view estimator to flexible dual views.*

<div align=center>  <img src="./figures/overview.png" alt="Main image" width="800" align="center" /> </div>

**Picture:**  *The proposed architecture.*

---

**Results**

<div align=center>  <img src="figures/results.png" alt="Teaser" width="800" align="bottom" /> </div>



This repository contains the official PyTorch implementation of the following paper:

> **Single-to-Dual-View Adaptation for Egocentric 3D Hand Pose Estimation**<br>
Ruicong Liu, Takehiko Ohkawa, Mingfang Zhang, and Yoichi Sato<br> <!-- >  https://arxiv.org/abs/  -->
> 
>**Abstract:**   The pursuit of accurate 3D hand pose estimation stands as a keystone for understanding human activity in the realm of egocentric vision. The majority of existing estimation methods still rely on single-view images as input, leading to potential limitations, e.g., limited field-of-view and ambiguity in depth. To address these problems, adding another camera to better capture the shape of hands is a practical direction. However, existing multi-view hand pose estimation methods suffer from two main drawbacks: 1) Requiring multi-view annotations for training, which are expensive. 2) During testing, the model becomes inapplicable if camera parameters/layout are not the same as those used in training. In this paper, we propose a novel Single-to-Dual-view adaptation (S2DHand) solution that adapts a pre-trained single-view estimator to dual views. Compared with existing multi-view training methods, 1) our adaptation process is unsupervised, eliminating the need for multi-view annotation. 2) Moreover, our method can handle arbitrary dual-view pairs with unknown camera parameters, making the model applicable to diverse camera settings. Specifically, S2DHand is built on certain stereo constraints, including pair-wise cross-view consensus and invariance of transformation between both views. These two stereo constraints are used in a complementary manner to generate pseudo-labels, allowing reliable adaptation. Evaluation results reveal that S2DHand achieves significant improvements on arbitrary camera pairs under both in-dataset and cross-dataset settings, and outperforms existing adaptation methods with leading performance.

## Resources

Material related to our paper is available via the following links:

- Paper: https://arxiv.org/pdf/2403.04381.pdf
- Code: https://github.com/ut-vision/S2DHand
- AssemblyHands dataset: https://assemblyhands.github.io/
- Pre-trained models: 

## System requirements

* Only Linux is tested.
* 64-bit Python 3.10 installation. 

## Playing with pre-trained networks and training

### Data preparation

Please download the pre-trained models and AssemblyHands dataset first. Assuming the pre-trained models and dataset are stored under ${DATA_DIR}. The structure of ${DATA_DIR} follows

```
- ${DATA_DIR}
    - AsseblyHands
        - annotations
        - images      
    - S2DHand-pretrain 
        - ckp_detnet_37.pth
        - ckp_detnet_68.pth
```
Please run the following command to register the pre-trained model and data.
```
cp ${DATA_DIR}/S2DHand-pretrain/* pretrain
mkdir data
mkdir data/assemblyhands
ln -s ${DATA_DIR}/AssemblyHands/* data/assemblyhands
```

### Single-to-Dual Views Adaptation (Training)

`run.sh` provides a complete procedure of training and testing.

We provide two optional arguments, `--stb` and `--pre`. They repersent two different network components, which could be found in our paper.

`--source` and `--target` represent the datasets used as the pre-training set and the dataset adapting to. It is recommended to use `gaze360, eth-mv-train` as `--source` and use `eth-mv` as `--target`. Please see `config.yaml` for the dataset configuration.

`--pairID` represents the index of dual-camera pair to adapt, ranging from 0 to 8.

`--i` represents the index of person which is used as the testing set. It is recommended to set it as -1 for using all the person as the training set.

`--pic` represents the number of image pairs for adaptation.

We also provide other arguments for adjusting the hyperparameters in our UVAGaze architecture, which could be found in our paper.

For example, run code like:

```bash
python3 adapt.py --i -1 --cams 18 --pic 256 --bs 32  --pairID 0 --savepath eth2eth --source eth-mv-train --target eth-mv --gpu 0 --stb --pre
```

### Test

`--i, --savepath, --target, --pairID` are the same as training. In addition to `eth-mv`, using `eth-mv100k` (a subset of ETH-MV) as `--target` is recommended for a faster testing.

For example, run code like:

```bash
python3 test_pair.py --pairID 0 --savepath eth2eth --target eth-mv100k --gpu 0
```

**Note**: the result printed by `test_pair.py` is **NOT** the final result on the specific dual-camera pair. It contains evaluation results on the **FULL** testing set.

Running `calc_metric.py` is needed to get four metrics on the pair we adapt to. These four metrics are the final results, which are described in our paper.

```bash
python3 calc_metric.py --pairID 0 --savepath eth2eth --source eth-mv-train --target eth-mv100k
```
We have provided the evaluation result of baseline model. Its result can be seen after running the above code, "base pair: ...". The result should be like:

```
base pair: Mono err: xx; Dual-S err: xx; Dual-A err: xx; HP err: xx
1: ...
2: ...
...
10: ...
```

The improvements brought by our method can be seen by comparing with the baseline results.

Please refer to `run.sh` for a complete procedure from training to testing.

## Citation

If this work or code is helpful in your research, please cite:

```latex
@inproceedings{liu2024uvagaze,
  title={UVAGaze: Unsupervised 1-to-2 Views Adaptation for Gaze Estimation},
  author={Liu, Ruicong and Lu, Feng},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  year={2024}
}
```
If you are using our ETH-MV dataset, please also cite the original paper of ETH-XGaze:

```latex
@inproceedings{zhang2020eth,
  title={Eth-xgaze: A large scale dataset for gaze estimation under extreme head pose and gaze variation},
  author={Zhang, Xucong and Park, Seonwook and Beeler, Thabo and Bradley, Derek and Tang, Siyu and Hilliges, Otmar},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part V 16},
  pages={365--381},
  year={2020},
  organization={Springer}
}
```

## Contact

For any questions, including algorithms and datasets, feel free to contact me by email: `liuruicong(at)buaa.edu.cn`
