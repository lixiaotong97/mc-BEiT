# mc-BEiT: Multi-choice Discretization for Image BERT Pre-training
Official pytorch implementation of "mc-BEiT: Multi-choice Discretization for Image BERT Pre-training" in European Conference on Computer Vision (__ECCV__) 2022. 

By Xiaotong Li, Yixiao Ge, Kun Yi, Zixuan Hu, Ying Shan, Ling-Yu Duan. 

## Updates

The pretrained and finetuned models are available! The full-version code is cleaning and coming soon.
## Introduction

> In this work, we introduce an improved BERT-style image pre-training method, namely mc-BEiT, which performs MIM proxy tasks towards eased and refined multi-choice training objectives. Specifically, the multi-choice supervision for the masked image patches is formed by the soft probability vectors of the discrete token ids, which are predicted by the off-the-shelf image ``tokenizer'' and further refined by high-level inter-patch perceptions resorting to the observation that similar patches should share their choices. 

![Overview](./overview.png)

## Model Zoo
Models and logs **pre-trained** and **fine-tuned** on ImageNet1k. You can download the model weights and finetuned on your customized downstream tasks. 

| Arch | Params | Epochs | Acc@1 | Pre-trained model | Fine-tuned model |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-Base| 86M | 800 | 84.1| [model](https://drive.google.com/file/d/1rh9ccxbJRBwhI69p97YkrOBkuwIOLh6l/view?usp=sharing)/[log](https://drive.google.com/file/d/1GXgwhbJLDATNyJQ96bNDeUX9wUKFH2-U/view?usp=sharing) | [model](https://drive.google.com/file/d/17Ffx2V1YRzFC_gOg8A-HSCwK2sQfnnJa/view?usp=sharing)/[log](https://drive.google.com/file/d/101iJEKNqFsHJR2XYlLmNwuihGX3Cd0ty/view?usp=sharing) |
| ViT-Large| 307M | 800 | 85.6| [model](https://drive.google.com/file/d/1_pbH5G4Pbf7LM2YRr-svMseqMi6toeBn/view?usp=sharing)/[log](https://drive.google.com/file/d/1JHSsyuZxNJJ1CcakGX1vTNDp-iM3BPWs/view?usp=sharing) | [model](https://drive.google.com/file/d/1oxcR4zJlKVJIzNOD_bZiTrCWA7Ny7RcH/view?usp=sharing)/[log](https://drive.google.com/file/d/1kkiMB0fjin9e7Ot8w9ehM3NH8ZAf0fYE/view?usp=sharing) |



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
