[![License](https://img.shields.io/badge/License-BSD%202--Clause-red.svg)](LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-orange.svg)](https://pytorch.org/)

# SDformer: A Dual-Branch Sparse-Differential Transformer for Multivariate Time Series Anomaly Detection

This repository contains the implementation of **SDformer**, a dual-branch Transformer architecture for multivariate time series anomaly detection.

> **Note**: This work is currently under review. Please do not distribute without permission.

## Features

- Dual-branch architecture combining Sparse Transformer and Differential Transformer
- Support for multiple benchmark datasets: SMAP, SWaT, WADI, SMD, MBA
- Includes 11 baseline models for comparison

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd DTAAD-main-org1

# Install PyTorch (GPU version)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Place your datasets in the `data/` folder, then run preprocessing:

```bash
python preprocess.py SMAP SWaT WADI SMD MBA
```

This will:
- Normalize the time series data
- Create sliding windows
- Save processed data to `processed/` folder

## Training

To train SDformer on a dataset:

```bash
python main.py --model SDformer --dataset SMAP --retrain
```

### Arguments

- `--model`: Model name (default: `DTAAD`)
  - Available: `SDformer`, `DTAAD`, `DAGMM`, `OmniAnomaly`, `USAD`, `MSCRED`, `CAE_M`, `MTAD_GAT`, `GDN`, `TranAD`, `MAD_GAN`
- `--dataset`: Dataset name (default: `SMD`)
- `--Device`: `cuda` or `cpu` (default: `cuda`)
- `--retrain`: Train from scratch
- `--test`: Test only (skip training)
- `--less`: Use 25% training data

### Examples

```bash
# Train SDformer on SWaT dataset
python main.py --model SDformer --dataset SWaT --retrain

# Test a trained model
python main.py --model SDformer --dataset SMAP --test

# Train with limited data
python main.py --model SDformer --dataset WADI --retrain --less
```

## Evaluation

The training script automatically evaluates the model and reports:
- Precision, Recall, F1-score
- Training time and inference time
- Anomaly scores for each test sample

Results and plots will be saved to the `plots/` folder.

## Supported Models

| Model | Description |
|-------|-------------|
| SDformer | Dual-branch sparse-differential Transformer (Ours) |
| DTAAD | Dual TCN + Transformer |
| TranAD | Self-conditioning Transformer |
| GDN | Graph Deviation Network |
| MTAD_GAT | Multi-scale GAT |
| USAD | Adversarial Autoencoder |
| OmniAnomaly | VAE with Stochastic RNN |
| MSCRED | Multi-scale ConvLSTM |
| MAD_GAN | GAN-based Detection |
| DAGMM | Deep Autoencoding GMM |
| CAE_M | Convolutional Autoencoder |

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.

```
BSD 2-Clause License

Copyright (c) 2025, Zhang Shihuan

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Citation

If you find this work useful, please consider citing (preprint):

```bibtex
@article{sdformer2025,
  title={SDformer: A Dual-Branch Sparse-Differential Transformer for Multivariate Time Series Anomaly Detection},
  author={Zhang, Shihuan and Others},
  journal={Under Review},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
