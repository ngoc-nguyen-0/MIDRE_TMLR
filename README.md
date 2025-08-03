# Random Erasing vs. Model Inversion: A Promising Defense or a False Hope? (TMLR 2025)
  <center>
  <img src="images/figure2.jpg" alt="PPA Examples"  height=260>
  </center>

> **Abstract:**
> *Model Inversion (MI) attacks pose a significant privacy threat by reconstructing private training data from machine learning models. While existing defenses primarily concentrate on model-centric approaches, the impact of data on MI robustness remains largely unexplored. In this work, we explore Random Erasing (RE), a technique traditionally used for improving model generalization under occlusion, and uncover its surprising effectiveness as a defense against MI attacks. Specifically, our novel feature space analysis shows that models trained with RE-images introduce a significant discrepancy between the features of MI-reconstructed images and those of the private data. At the same time, features of private images remain distinct from other classes and well-separated from different classification regions. These effects collectively degrade MI reconstruction quality and attack accuracy while maintaining reasonable natural accuracy. Furthermore, we explore two critical properties of RE including Partial Erasure and Random Location. Partial Erasure prevents the model from observing entire objects during training. We find this has a significant impact on MI, which aims to reconstruct the entire objects. Random Location of erasure plays a crucial role in achieving a strong privacy-utility trade-off. Our findings highlight RE as a simple yet effective defense mechanism that can be easily integrated with existing privacy-preserving techniques. Extensive experiments across 37 setups demonstrate that our method achieves state-of-the-art (SOTA) performance in the privacy-utility trade-off. The results consistently demonstrate the superiority of our defense over existing methods across different MI attacks, network architectures, and attack configurations. For the first time, we achieve a significant degradation in attack accuracy without a decrease in utility for some configurations.*  
[Full Paper (PDF)](https://arxiv.org/abs/2409.01062)

## Model Inversion Defense
  <center>
  <img src="images/MIDRE_2.png" alt="Defense Pipeline"  height=1000>
  </center>

# Setup and Run Attacks
## Install
```
conda create --name midre python=3.10 -y
conda activate midre

pip install -r requirements.txt

```

## Setup StyleGAN2
Clone the official [StyleGAN2-ADA-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch):

```
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
rm -r --force stylegan2-ada-pytorch/.git/
rm -r --force stylegan2-ada-pytorch/.github/
rm --force stylegan2-ada-pytorch/.gitignore

wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -P stylegan2-ada-pytorch/

```

## Prepare Datasets
We support [FaceScrub](http://vintage.winklerbros.net/facescrub.html), [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) as datasets to train the target models. Please follow the instructions on the websites to download the datasets. 
Replace the ```root``` of dataset in  ```datasets/facescrub.py```
Make sure that the following structure is kept:

  
    ├── celeba
        ├── img_align_celeba
        ├── identity_CelebA.txt
        ├── list_attr_celeba.txt
        ├── list_bbox_celeba.txt
        ├── list_eval_partition.txt
        ├── list_landmarks_align_celeba.txt
        └── list_landmarks_celeba.txt
    ├── facescrub
        ├── actors
            ├── faces
            └── images
        └── actresses
            ├── faces
            └── images
    ├── stanford_dogs
        ├── Annotation
        ├── Images
        ├── file_list.mat
        ├── test_data.mat
        ├── test_list.mat
        ├── train_data.mat
        └── train_list.mat

## Train Target Models

To define the model and training configuration, you need to create a configuration file. We provide a example configuration with explanations at ```configs/training/targets/ResNet18_FaceScrub_midre.yaml```. To train the target models accordingly to our paper, we provide a training configuration for each dataset. You only need to adjust the architecture used and the Weights & Biases configuration - all other parameters are identical for each target dataset. Only the batch size has to be adjusted for larger models.

After a training configuration file has been created, run the following command to start the training with the specified configuration:
```bash
python train_model.py -c=configs/training/targets/ResNet18_FaceScrub_midre.yaml
```
After the optimization is performed, the results are automatically evaluated. All results together with the initial, optimized, and selected latent vectors are stored at WandB.

## Perform Attacks
To perform our attacks, prepare an attack configuration file including the WandB run paths of the target and evaluation models. We provide an example configuration with explanations at ```configs/attacking/FaceScrub_FFHQ_resnet18_midre.yml```. We further provide configuration files to reproduce the various attack results stated in our paper. You only need to adjust the run paths for each dataset combination, and possibly the batch size.

After an attack configuration file has been created, run the following command to start the attack with the specified configuration:
```bash
python attack.py -c=configs/attacking/FaceScrub_FFHQ_resnet18_midre.yml

```

All results including the metrics will be stored at WandB for easy tracking and comparison.

## Citation
If you build upon our work, please don't forget to cite us.
```
@article{
Tran2025random,
title={Random Erasing vs. Model Inversion: A Promising Defense or a False Hope?},
author={Viet-Hung Tran, Ngoc-Bao Nguyen, Son T. Mai, Hans Vandierendonck, Ira Assent, Alex Kot, Ngai-Man Cheung},
journal={Accepted in Transactions on Machine Learning Research (TMLR)},
year={2025},
url={https://openreview.net/forum?id=S9CwKnPHaO}
}

```

## Implementation Credits
Some of our implementations rely on other repos. We want to thank the authors for making their code publicly available. 
For license details, refer to the corresponding files in our repo. For more details on the specific functionality, please visit the corresponding repos.
- PPA: https://github.com/LukasStruppek/Plug-and-Play-Attacks/tree/master
- FID Score: https://github.com/mseitzer/pytorch-fid
- Stanford Dogs Dataset Class: https://github.com/zrsmithson/Stanford-dogs
- FaceNet: https://github.com/timesler/facenet-pytorch
- Negative Label Smoothing: https://github.com/UCSC-REAL/negative-label-smoothing
