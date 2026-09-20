<h1 align="center">Synergizing Feature Smoothing and Cross-sample Mixing to Promote Adversarial Transferability in Vision Transformers</h1>

## Paper
📄 [IEEE TDSC 2026 - Synergizing Feature Smoothing and Cross-sample Mixing to Promote Adversarial Transferability in Vision Transformers](https://ieeexplore.ieee.org/abstract/document/11691579)

## Overview
![Overview](./figs/overview.png)

## Requirements
- Python==3.9.23
- torch==1.12.1+cu116
- torchvision==0.13.1+cu116
- timm==0.9.12
- numpy==1.24.4

```bash
pip install -r requirements.txt
```

## Attack and Evaluation
```bash
python main.py
```

## Citation
If you find this work useful, please cite:

```bibtex
@article{ZhuHG:Adversarial:TDSC26,
  author    = {Hegui Zhu and Jingyan Tian and Xingwei Wang and Chengqing Li and Yaguan Qian},
  title     = {Synergizing Feature Smoothing and Cross-sample Mixing to Promote Adversarial Transferability in Vision Transformers},
  journal   = {IEEE Transactions on Dependable and Secure Computing},
  year      = {2027},
  volume    = {24},
  number    = {1},
  pages     = {},
  doi       = {10.1109/TDSC.2026.3734212},
  source    = {https://github.com/FourMouthWhale/SFM},
}
```