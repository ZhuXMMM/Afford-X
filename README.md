# A Generalizable and Slim Affordance Reasoning Framework for Task-oriented Manipulation

This repository is an official implementation of AffordX:

A Generalizable and Slim Affordance Reasoning Framework for Task-oriented Manipulation
[Xiaomeng Zhu]( )<sup>†</sup>, [Yuyang Li]( )<sup>†</sup>, [Leiyao Cui]( ), [Pengfei Li](), [Huan-ang Gao](), [Yixin Zhu](https://yzhu.io/)<sup>✉</sup>, [Hao Zhao](https://sites.google.com/view/fromandto)<sup>✉</sup>

<small><sup>†</sup> Equal contribution, <sup>✉</sup> Corresponding author.</small>

## Introduction
Object affordance reasoning, the ability to infer object functionalities based on physical properties, is fundamental for task-oriented planning and activities in both humans and \ac{ai}. This capability, required for planning and executing daily activities in a task-oriented manner, relies on commonsense knowledge of object physics and functionalities, extending beyond simple object recognition. Current computational models for affordance reasoning from perception lack generalizability, limiting their applicability in novel scenarios. Meanwhile, comprehensive \acp{llm} with emerging reasoning capabilities are challenging to deploy on local devices for task-oriented manipulations. Here, we introduce \affordanceDatasetNamelvis{}, a large-scale dataset comprising 1,496 tasks and 897k images, designed to enhance the generalizability of affordance reasoning from perception. Utilizing this dataset, we develop \affordanceModelName{}, an end-to-end trainable affordance reasoning model that incorporates Verb Attention and Bi-Fusion modules to improve multi-modal understanding. This model achieves up to a 25.5\% performance improvement on unseen categories and tasks, while maintaining a compact 187M parameter size and inferring nearly 50 times faster than the GPT-4V API. Our work demonstrates the potential for efficient, generalizable affordance reasoning models that can be deployed on local devices for task-oriented manipulations. We showcase \affordanceModelName{}'s effectiveness in enabling task-oriented manipulations for robots across various tasks and environments, underscoring its efficiency and broad implications for advancing robotics and \ac{ai} systems in real-world applications.

<p align="center"><img src="media/teaser.png" width="600" /></p>


<!-- If you find our code or paper useful, please consider citing: -->
<!-- ```bibtex
@article{li2022toist,
  title={Afford-X: Task Oriented Instance Segmentation Transformer with Noun-Pronoun Distillation},
  author={Li, Pengfei and Tian, Beiwen and Shi, Yongliang and Chen, Xiaoxue and Zhao, Hao and Zhou, Guyue and Zhang, Ya-Qin},
  journal={arXiv preprint arXiv:2210.10775},
  year={2022}
}
``` -->

This repository is a PyTorch implementation.

## Datasets
Please follow the instructions in [the official website](https://github.com/coco-tasks/dataset) to download the COCO-Tasks dataset.

You can organize the 'data' folder as follows:
```
data/
  ├── id2name.json
  ├── images/
  │    ├── train2014/
  │    └── val2014/
  └── coco-tasks/
       └── annotations/
            ├── task_1_train.json
            ├── task_1_test.json
            ...
            ├── task_14_train.json
            └── task_14_test.json
```
Then set the arguments `coco_path`, `refexp_ann_path` and `catid2name_path` in file `configs/tdod.json` to be the path of `data/images/`, `data/coco-tasks/annotations/` and `data/id2name.json`, respectively.

## Installation
Make sure that you have all dependencies in place. The simplest way to do so is to use anaconda.

Make a new conda env and activate it:
```
conda create --name AffordX python=3.8
conda activate AffordX
```

Install the the packages in the requirements.txt:
```
pip install -r requirements.txt
```

## Running

All training and evaluation scripts are organized in the `scripts` directory. The directory structure is as follows:

```
scripts/
  ├── noun/                 # Scripts for training teacher model (noun input)
  │    ├── noun_cocotasks.sh        # Training on COCO-Tasks
  │    ├── noun_mask_cocotasks.sh   # Training segmentation on COCO-Tasks
  │    ├── eval_noun.sh             # Evaluation for detection
  │    ├── eval_noun_mask.sh        # Evaluation for segmentation
  │    └── ...                      # Scripts for other datasets (COCO, LVIS)
  │
  ├── pronoun/              # Scripts for training student model (pronoun input)
  │    ├── pronoun_cocotasks.sh     # Training on COCO-Tasks
  │    ├── pronoun_mask_cocotasks.sh# Training segmentation on COCO-Tasks
  │    ├── eval_pronoun.sh          # Evaluation for detection
  │    ├── eval_pronoun_mask.sh     # Evaluation for segmentation
  │    └── ...                      # Scripts for other datasets (COCO, LVIS)
  │
  └── distillation/        # Scripts for training with distillation framework
       ├── distill_cocotasks.sh     # Detection distillation on COCO-Tasks
       ├── distill_mask_cocotasks.sh # Segmentation distillation on COCO-Tasks
       ├── eval_distill.sh          # Evaluation for distilled detection
       ├── eval_distill_mask.sh     # Evaluation for distilled segmentation
       └── ...                      # Scripts for other datasets (COCO, LVIS)
```

### Training and Evaluation

1. **For Teacher Model (Noun Input)**:
```bash
# Train detection model on COCO-Tasks
bash scripts/noun/noun_cocotasks.sh

# Train segmentation model on COCO-Tasks
bash scripts/noun/noun_mask_cocotasks.sh

# Evaluate detection model
bash scripts/noun/eval_noun.sh

# Evaluate segmentation model
bash scripts/noun/eval_noun_mask.sh
```

2. **For Student Model (Pronoun Input)**:
```bash
# Train detection model on COCO-Tasks
bash scripts/pronoun/pronoun_cocotasks.sh

# Train segmentation model on COCO-Tasks
bash scripts/pronoun/pronoun_mask_cocotasks.sh

# Evaluate detection model
bash scripts/pronoun/eval_pronoun.sh

# Evaluate segmentation model
bash scripts/pronoun/eval_pronoun_mask.sh
```

3. **For Distillation Framework**:
```bash
# Train detection with distillation on COCO-Tasks
bash scripts/distillation/distill_cocotasks.sh

# Train segmentation with distillation on COCO-Tasks
bash scripts/distillation/distill_mask_cocotasks.sh

# Evaluate distilled detection model
bash scripts/distillation/eval_distill.sh

# Evaluate distilled segmentation model
bash scripts/distillation/eval_distill_mask.sh
```

Note: Each script contains predefined hyperparameters and configurations. Please check the script content before running and modify paths and parameters as needed. For different datasets, please modify the following parameters:

- For COCO-Tasks: Set `--dataset_config configs/tdod.json`
- For COCO-Aff: Set `--dataset_config configs/tdod_coco.json` and add `--load_word_full`
- For LVIS-Aff: Set `--dataset_config configs/tdod_lvis.json` and add `--load_full --load_word_full`

## Pre-trained Models

We provide pre-trained weights for different datasets:

- [COCO-Tasks Model](https://drive.google.com/file/d/1XMy2eBrW6SOrLMI3NpGdDztxRbFvlkpY/view?usp=sharing)
- [COCO-Aff Model](https://drive.google.com/file/d/1hTmZheupK3IY8LcOj0mwESEvXDkkh7Us/view?usp=drive_link)
- [LVIS-Aff Model](https://drive.google.com/file/d/11BOEOh7UNKwp3EwcURij44vQpFyRgJ60/view?usp=drive_link)

After downloading the weights, you can evaluate the models using the provided script. Please modify the model path and other parameters in the script before running:

```bash
bash scripts/distillation/eval_distill_mask.sh
```

## Datasets

### COCO-Tasks
The COCO-Tasks dataset can be found at [the official website](https://coco-tasks.github.io/). Follow the instructions there to download the dataset.

### COCO-Aff & LVIS-Aff
These datasets will be made publicly available soon. Stay tuned!

#### Data Organization
After downloading, organize your data directory as follows:
```
data/
  ├── coco-tasks/           # COCO-Tasks dataset
  │    └── annotations/     
  │         ├── task_1_train.json
  │         ├── task_1_test.json
  │         └── ...
  │
  ├── coco-aff/            # COCO-Aff dataset (coming soon)
  │    └── annotations/
  │         └── ...
  │
  ├── lvis-aff/           # LVIS-Aff dataset (coming soon)
  │    └── annotations/
  │         └── ...
  │
  ├── images/             # Shared image directory
  │    ├── train2014/    # For COCO-Tasks and COCO-Aff
  │    ├── val2014/      # For COCO-Tasks and COCO-Aff
  │    ├── train2017/    # For LVIS-Aff
  │    └── val2017/      # For LVIS-Aff
  │
  ├── id2name.json       # Category mapping file
  └── id2name_lvis.json  # Category mapping file
```

## License

Afford-X is released under the MIT License.


## Acknowledgment

We would like to thank the open-source data and code of [COCO-Tasks](https://coco-tasks.github.io/), [Microsoft COCO](https://cocodataset.org/#home), [GGNN](https://github.com/yassersouri/task-driven-object-detection), [MDETR](https://github.com/ashkamath/mdetr), [DETR](https://github.com/facebookresearch/detr), [Detectron2](https://github.com/facebookresearch/detectron2) and [TOIST](https://github.com/AIR-DISCOVER/TOIST).




