**BiADT**: Bidirectional Alignment for Domain Adaptive Detection with Transformers
========


This repository is an official implementation of the [BiADT](https://www.amazon.science/publications/bidirectional-alignment-for-domain-adaptive-detection-with-transformers) (accepted to **ICCV 2023**). Code will be avaliable after the internal approval.
[[paper link](https://www.amazon.science/publications/bidirectional-alignment-for-domain-adaptive-detection-with-transformers)]

## News

[2023/8]Code will be released soon!

[2023/7]BiADT is accepted by ***ICCV 2023***

## Introduction


**Abstract**: 
We propose a Bidirectional Alignment for domain adaptive Detection with Transformers (BiADT) to improve cross domain object detection performance. Existing adversarial learning based methods use gradient reverse layer (GRL) to reduce the domain gap between the source and target domains in feature representations. Since different image parts and objects may exhibit various degrees of domain-specific characteristics, directly applying GRL on a global image or object representation may not be suitable. Our proposed BiADT explicitly estimates token-wise domain-invariant and domain-specific features in the image and object token sequences. BiADT has a novel deformable attention and self-attention, aimed at bi-directional domain alignment and mutual information minimization. These two objectives reduce the domain gap in domain-invariant representations, and simultaneously increase the distinc- tiveness of domain-specific features. Our experiments show that BiADT achieves very competitive performance to SOTA consistently on Cityscapes-to-FoggyCityscapes, Sim10K-to-Citiscapes and Cityscapes-to-BDD100K, out- performing the strong baseline, AQT, by 2.0, 2.1, and 2.4 in mAP50, respectively.


## Model Weights
We provide the model weights for C2FC domain shift. The model weights for BIADT w/o and w/ AQT can be found here: [c2fc_biadt.pth](https://drive.google.com/file/d/1XsItqdHkoO0zAcdXkZWQwEcnXp-avop-/view?usp=drive_link) (AP=49.6) and [c2fc_aqt_biadt.pth](https://drive.google.com/file/d/1Fl4Kzkto6CN8xPyEdDzNmKyHj7LkHqJg/view?usp=drive_link)(AP=50.1).

For the inference, see test.sh and test2.sh. For training, see train.sh.

## Links
Our work is based on **DAB-DETR** (DN-DETR version) and **AQT**. We sincerely appreciate their great work! 
- **AQT: Adversarial Query Transformers for Domain Adaptive Object Detection**.     
Wei-Jie Huang, Yu-Lin Lu, Shih-Yao Lin, Yusheng Xie, and Yen-Yu Lin.  
[IJCAI-ECAI 2022 ](https://ijcai-22.org/)  
[[paper]](http://vllab.cs.nctu.edu.tw/images/paper/ijcai-huang22.pdf) [[code]](https://github.com/weii41392/AQT/tree/master).  

- **DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR**.  
Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, Lei Zhang.    
International Conference on Learning Representations (ICLR) 2022.  
[[Paper]](https://arxiv.org/abs/2201.12329) [[Code]](https://github.com/SlongLiu/DAB-DETR).     

- **DN-DETR: Accelerate DETR Training by Introducing Query DeNoising**.  
Feng Li, Hao Zhang*, Shilong Liu, Jian Guo, Lionel M.Ni, and Lei Zhang.    
IEEE/CVF conference on computer vision and pattern recognition (CVPR oral) 2022.  
[[Paper]](https://arxiv.org/pdf/2203.01305.pdf)   [[Code]](https://github.com/IDEA-Research/DN-DETR)



## LICNESE
BiADT is released under the Apache 2.0 license. Please see the [LICENSE](LICNESE) file for more information.

<!-- Copyright (c) OSU and Amazon. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. -->

## Bibtex
If you find our work helpful for your research, please consider citing the following BibTeX entry.   
```
@article{he2023bidirectional,
  title={Bidirectional alignment for domain adaptive detection with transformers},
  author={He, Liqiang and Wang, Wei and Chen, Albert and Sun, Min and Kuo, Cheng-Hao and Todorovic, Sinisa},
  year={2023}
}
```
