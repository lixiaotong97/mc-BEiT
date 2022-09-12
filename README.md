# mc-BEiT: Multi-choice Discretization for Image BERT Pre-training
Official pytorch implementation of "mc-BEiT: Multi-choice Discretization for Image BERT Pre-training" in European Conference on Computer Vision (__ECCV__) 2022. 

By Xiaotong Li, Yixiao Ge, Kun Yi, Zixuan Hu, Ying Shan, Ling-Yu Duan. 


## Introduction

> In this work, we introduce an improved BERT-style image pre-training method, namely mc-BEiT, which performs MIM proxy tasks towards eased and refined multi-choice training objectives. Specifically, the multi-choice supervision for the masked image patches is formed by the soft probability vectors of the discrete token ids, which are predicted by the off-the-shelf image ``tokenizer'' and further refined by high-level inter-patch perceptions resorting to the observation that similar patches should share their choices. 

![Overview](./overview.png)

## Model Zoo
+ We provide the models and logs **pre-trained** and **fine-tuned** on ImageNet1k. 
+ You can download the model weights and finetuned on your customized downstream tasks. 

| Arch | Params | Epochs | Acc@1 | Pre-trained model | Fine-tuned model |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-Base| 86M | 800 | 84.1| [model](https://drive.google.com/file/d/1rh9ccxbJRBwhI69p97YkrOBkuwIOLh6l/view?usp=sharing)/[log](https://drive.google.com/file/d/1GXgwhbJLDATNyJQ96bNDeUX9wUKFH2-U/view?usp=sharing) | [model](https://drive.google.com/file/d/17Ffx2V1YRzFC_gOg8A-HSCwK2sQfnnJa/view?usp=sharing)/[log](https://drive.google.com/file/d/101iJEKNqFsHJR2XYlLmNwuihGX3Cd0ty/view?usp=sharing) |
| ViT-Large| 307M | 800 | 85.6| [model](https://drive.google.com/file/d/1_pbH5G4Pbf7LM2YRr-svMseqMi6toeBn/view?usp=sharing)/[log](https://drive.google.com/file/d/1JHSsyuZxNJJ1CcakGX1vTNDp-iM3BPWs/view?usp=sharing) | [model](https://drive.google.com/file/d/1oxcR4zJlKVJIzNOD_bZiTrCWA7Ny7RcH/view?usp=sharing)/[log](https://drive.google.com/file/d/1kkiMB0fjin9e7Ot8w9ehM3NH8ZAf0fYE/view?usp=sharing) |

## Setup
Clone the github repo and install the required packages.
```
git clone git@github.com:lixiaotong97/mc-BEiT.git
pip install -r requirements.txt
```
For mixed-precision training, please install [apex](https://github.com/NVIDIA/apex)

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
## Data Preparation
+ We use standard ImageNet-1K dataset (http://image-net.org/) for pre-training
+ Read from train and val list (download in this [link](https://drive.google.com/drive/folders/1Kmu3VHw1Ssqh6jwrWaUL1ihVx9KakKZv?usp=sharing)) to boost the speed of reading images from massive small files:
```
/dataset
└── imagenet1k
    ├── train
    ├── val
    ├── train_map.txt
    └── val_map.txt
```
+ `train_map.txt`,`val_map.txt` : which store the relative path in the corresponding zip file and ground truth label, and can be downloaded in this [link](https://drive.google.com/drive/folders/1Kmu3VHw1Ssqh6jwrWaUL1ihVx9KakKZv?usp=sharing).
## Pre-training on ImageNet-1K
+ Download the off-the-shelf tokenizer and config in this [link](https://drive.google.com/drive/folders/101qHTHO5YiS3RLe7g3j_Uku_JdR7GKge?usp=sharing) and place it under `./weight/tokenzier/`.

+ We pre-train the ViT-Base model with 16 NVIDIA A100/V100 GPUs on ImageNet-1K as follows:

```
OUTPUT_DIR="./output/mcbeit_pretrained"
DATA_PATH="./dataset/imagenet1k"
TOKENIZER_PATH='./weight/tokenzier'
mkdir -p $OUTPUT_DIR

python -m torch.distributed.launch --nproc_per_node=16 run_mcbeit_pretraining.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} \
        --model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
        --batch_size 128 --lr 1.5e-3 --warmup_epochs 10 --epochs 800 \
        --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1 \
        --mask_type 'random' \
        --imagenet_default_mean_and_std \
        --omega 0.2 \
        --temp 4.0 \
        --temp_warmup 4.0 \
```

## Fine-tuning on ImageNet-1K Classification
+ We finetune the pre-trained ViT-Base model with 8 NVIDIA A100/V100 GPUs as follows: 
```
CKP="./output/mcbeit_pretrained/checkpoint-799.pth"
OUTPUT_DIR="./output/mcbeit_finetuned/"
DATA_PATH="/dataset/imagenet1k/"
mkdir -p $OUTPUT_DIR

python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model beit_base_patch16_224 --data_path ${DATA_PATH}\
    --finetune ${CKP} \
    --output_dir ${OUTPUT_DIR} --batch_size 128 --lr 4e-3 --update_freq 1 \
    --warmup_epochs 20 --epochs 100 --layer_decay 0.65 --drop_path 0.1 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 --nb_classes 1000 --enable_deepspeed \
    --imagenet_default_mean_and_std
```
## Fine-tuning on ADE20K Semantic Segmentation
+ Follow the guidence in [BEiT](https://github.com/microsoft/unilm/tree/master/beit) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) to install library and required packages.
```
cd semantic_segmentation
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```
+ For example, we finetune a ViT-base backbone with UperNet using 8 NVIDIA A100/V100 GPUs:
```
CKP='/mcbeit_pretrained/checkpoint-799.pth'
bash tools/dist_train.sh \
    configs/mcbeit/upernet/upernet_mcbeit_base_12_512_slide_160k_ade20k_pt 8 \
    --work-dir /path/to/save --seed 0  --deterministic \
    --options model.pretrained=${CKP}
```

## Fine-tuning on COCO Detection and Segmentation

+ The experiment is built on [MIMDet](https://github.com/hustvl/MIMDet). Follow the instruction to install the required packages.
+ Prepare the COCO Dataset.
+ Place the pre-trained model path in the training config file.

+ For example, we finetune a ViT-base backbone with MaskRCNN and the model is trained for 25 epochs.

```
cd mimdet
python lazyconfig_train_net.py --config-file configs/benchmarking/benchmarking_mask_rcnn_base_FPN_25ep_LSJ_mc-beit.py --num-gpus 8 --num-machines 4 --dist-url tcp://127.0.0.1:21633 $3 $4 $5
```
## Acknowledgement

This repository is built using the [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository, the [timm](https://github.com/rwightman/pytorch-image-models) library, the [DeiT](https://github.com/facebookresearch/deit) repository, and the [MIMDet](https://github.com/hustvl/MIMDet) repository. Thanks for their excellent projects!

## Citation
If you find our work is useful for your research, please kindly cite our paper.
```
@inproceedings{li2022mc,
  title={mc-BEiT: Multi-choice Discretization for Image BERT Pre-training},
  author={Li, Xiaotong and Ge, Yixiao and Yi, Kun and Hu, Zixuan and Shan, Ying and Duan, Ling-Yu},
  booktitle={European conference on computer vision},
  year={2022},
}
```
## Contact
If you have any questions, you can contact me from the email: lixiaotong@stu.pku.edu.cn