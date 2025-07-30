# SAT-UIR: Self-Assessment Training for Semi-Supervised Underwater Image Restoration

## Introduction
This repository provides the official code for **SAT-UIR**, a self-assessment training framework for semi-supervised underwater image restoration.

## Dependencies

- Ubuntu==22.04.5 LTS
- Pytorch==2.1.0  
- CUDA==12.1

Other dependencies are listed in `requirements.txt`

## Data Preparation

Run `data_split.py` to randomly split your paired datasets into training, validation and testing set.

Run `estimate_illumination.py` to get illumination map of the corresponding image.

Finally, the structure of  `data`  are aligned as follows:

```
data
├── labeled
│   ├── input
│   └── GT
│   └── LA
├── unlabeled
│   ├── input
│   └── LA
│   └── candidate
└── val
    ├── input
    └── GT
    └── LA
└── test
    ├── benchmarkA
        ├── input
        └── LA
```

You can download the training set and test sets from benchmarks [UIEB](https://li-chongyi.github.io/proj_benchmark.html), [EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset), [UWCNN](https://li-chongyi.github.io/proj_underwater_image_synthesis.html), [Sea-thru](http://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html), [RUIE](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark). 

## Test

Put your test benchmark under `data/test` folder, run `estimate_illumination.py` to get its illumination map.

Run `test.py` and you can find results from folder `result`.

## Train

To train the framework, run `create_candiate.py` to initialize reliable bank. Hyper-parameters can be modified in `trainer.py`.

Run `train.py` to start training.

## Acknowledgement
The training code architecture is based on the [Semi-UIR](https://github.com/Huang-ShiRui/Semi-UIR), thanks for their work. 

We also thank for the following repositories: [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch), [UIEB](https://li-chongyi.github.io/proj_benchmark.html), [EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset), [UWCNN](https://li-chongyi.github.io/proj_underwater_image_synthesis.html), [Sea-thru](http://csms.haifa.ac.il/profiles/tTreibitz/datasets/sea_thru/index.html), [RUIE](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark),[GDCP](https://github.com/ytpeng-aimlab/Generalization-of-the-Dark-Channel-Prior-for-Single-Image-Restoration/tree/main), [MMLE](https://github.com/Li-Chongyi/MMLE_code), [PCDE](https://github.com/Li-Chongyi/PCDE), [WaterNet](https://github.com/Li-Chongyi/Water-Net_Code), [Ucolor](https://github.com/Li-Chongyi/Ucolor), [FUnIE-GAN](https://github.com/xahidbuffon/FUnIE-GAN), [CWR](https://github.com/JunlinHan/CWR), [U-shape](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement), [Water-CDM](https://github.com/HKandWJJ/Water-CDM), [GH-UIR](https://github.com/CXH-Research/GuidedHybSensUIR) and [SuirSIR](https://github.com/jacezhang66/OctopusAI-suirSIR-network).
