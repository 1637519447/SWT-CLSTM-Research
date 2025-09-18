# SWT-CLSTM: Stationary Wavelet Transform Enhanced Convolutional LSTM for Cloud Resource Utilization Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## 📖 Overview

SWT-CLSTM is a novel deep learning framework that combines **Stationary Wavelet Transform (SWT)**, **Convolutional Neural Networks (CNN)**, and **Long Short-Term Memory (LSTM)** networks for accurate cloud resource utilization prediction. The model integrates multi-scale temporal feature extraction with contrastive learning mechanisms to achieve superior performance in CPU and memory utilization forecasting.

### Key Features

- 🌊 **Multi-scale Decomposition**: Utilizes Stationary Wavelet Transform for comprehensive temporal pattern analysis
- 🧠 **Hybrid Architecture**: Combines CNN for spatial feature extraction and LSTM for temporal dependency modeling
- 🔄 **Contrastive Learning**: Implements advanced data augmentation and contrastive loss for robust feature learning
- 📊 **Multi-dataset Support**: Validated on real-world datasets from Alibaba and Google cloud traces
- 🎯 **Statistical Validation**: Comprehensive statistical significance testing with t-tests and Wilcoxon signed-rank tests
- ⚡ **GPU Acceleration**: Full CUDA support for efficient training and inference

## 🏗️ Architecture

The SWT-CLSTM model consists of three main components:

1. **SWT Decomposition Layer**: Decomposes time series into multiple frequency components
2. **CNN-LSTM Hybrid Network**: 
   - Convolutional layers for local pattern extraction
   - Five-layer LSTM with decreasing hidden units (200→160→130→100→70)
3. **Contrastive Learning Module**: Enhances model robustness through data augmentation and InfoNCE loss

## 📊 Performance

Our extensive experiments demonstrate significant improvements over baseline models:

### CPU Utilization Prediction (Alibaba 30s dataset)
- **SWT-CLSTM**: 2.315×10⁻⁵ MSE
- **Performance Improvement**: 300%-2845% over baseline models (LSTM, ARIMA, TFC, PatchTST, TimeMixerPlusPlus)

### Memory Utilization Prediction (Alibaba 30s dataset)
- **SWT-CLSTM**: 4.721×10⁻⁷ MSE
- **Performance Improvement**: 233%-12026% over baseline models

All improvements are statistically significant (p < 0.001) with large effect sizes (Cohen's d > 0.7).

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.9.0
CUDA >= 11.0 (optional, for GPU acceleration)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/1637519447/SWT-CLSTM-Research.git
cd SWT-CLSTM-Research
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scipy scikit-learn
pip install pywavelets joblib tqdm thop
```

### Usage

#### Basic Training

```bash
# Train on CPU utilization data
python CPUrate_SWT_CLSTM_improved.py

# Train on memory utilization data
python Mem_SWT_CLSTM_improved.py
```

#### Ablation Studies

```bash
# Run ablation experiments
python CPUrate_SWT-CLSTM-C.py    # Without CNN
python CPUrate_SWT-CLSTM-S.py    # Without SWT
python CPUrate_SWT-CLSTM-SG.py   # Without SG filter
python CPUrate_SWT-CLSTM-SC.py   # Without SWT and CNN
python CPUrate_SWT-CLSTM-SSC.py  # Without SWT, SG filter, and CNN
```

#### Baseline Comparisons

```bash
# Compare with baseline models
python CPUrate_compared_PatchTST.py
python CPUrate_compared_TFC.py
python CPUrate_compared_TimeMixer.py
python CPUrate_compared_SG-LSTM_improved.py
```

#### Statistical Analysis

```bash
# Run statistical significance tests
python statistical_significance_test.py

# Generate robustness analysis
python robustness_check_SWT_CLSTM.py
```

## 📁 Project Structure

```
SWT-CLSTM-Research/
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore file
├── 
├── Core Models/
│   ├── CPUrate_SWT_CLSTM_improved.py  # Main SWT-CLSTM implementation
│   ├── Mem_SWT_CLSTM_improved.py      # Memory prediction variant
│   └── Multivariate_SWT_CLSTM.py      # Multivariate version
├── 
├── Ablation Studies/
│   ├── Ablation/                       # Ablation experiment results
│   ├── CPUrate_SWT-CLSTM-*.py         # Ablation variants
│   └── Mem_SWT-CLSTM-*.py             # Memory ablation variants
├── 
├── Baseline Models/
│   ├── CPUrate_compared_*.py          # Baseline model implementations
│   ├── Memrate_compared_*.py          # Memory baseline models
│   └── Arima_cor.py                   # ARIMA baseline
├── 
├── Data Processing/
│   ├── alibaba_data_processor.py      # Alibaba dataset processor
│   ├── google_cluster_data_processor.py # Google dataset processor
│   ├── azure_data_processor.py        # Azure dataset processor
│   └── csv_data_cleaner.py            # Data cleaning utilities
├── 
├── Analysis & Visualization/
│   ├── statistical_significance_test.py # Statistical testing
│   ├── robustness_check_SWT_CLSTM.py   # Robustness analysis
│   ├── visualize_distribution_shift_results.py # Distribution analysis
│   ├── *_plot.py                       # Visualization scripts
│   └── images/                         # Generated plots and figures
├── 
├── Datasets/
│   ├── Alibaba_*.csv                   # Alibaba cloud traces
│   ├── Google_*.csv                    # Google cluster traces
│   ├── Azure_*.csv                     # Azure cloud traces
│   ├── Pre_data/                       # Preprocessed data
│   └── Compared_data/                  # Comparison results
├── 
└── Results/
    ├── *.json                          # Experimental results
    ├── *.txt                           # Analysis reports
    └── *.csv                           # Statistical test results
```

## 🔬 Experimental Setup

### Datasets

1. **Alibaba Cloud Traces (30s intervals)**
   - Real-world production data from Alibaba cloud infrastructure
   - CPU and memory utilization metrics

2. **Google Cluster Traces (5m intervals)**
   - Large-scale cluster workload data from Google
   - Resource usage patterns from production clusters

3. **Azure Cloud Traces (5m intervals)**
   - Microsoft Azure cloud infrastructure data
   - Used for distribution shift analysis

### Evaluation Metrics

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**
- **Step-wise RMSE** (for multi-step prediction analysis)

### Statistical Validation

- **Paired t-tests** for mean difference significance
- **Wilcoxon signed-rank tests** for non-parametric validation
- **Cohen's d** for effect size measurement
- **Confidence intervals** at 95% level

## 📈 Results and Analysis

### Model Complexity
- **Parameters**: ~2.1M trainable parameters
- **MACs**: ~847K multiply-accumulate operations
- **Training Time**: ~19s per epoch (GPU)
- **Inference Time**: <1ms per sample

### Ablation Study Results
| Variant | Component Removed | Performance Drop |
|---------|------------------|------------------|
| SWT-CLSTM-C | CNN | 15-25% |
| SWT-CLSTM-S | SWT | 20-35% |
| SWT-CLSTM-SG | SG Filter | 10-18% |
| SWT-CLSTM-SC | SWT + CNN | 35-50% |
| SWT-CLSTM-SSC | SWT + SG + CNN | 45-60% |

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

<!-- ### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request -->



<!-- ## 📚 Citation

If you use this code in your research, please cite our work:

```bibtex
@article{swt_clstm_2024,
  title={SWT-CLSTM: Stationary Wavelet Transform Enhanced Convolutional LSTM for Cloud Resource Utilization Prediction},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]}
}
``` -->

## 🙏 Acknowledgments

- Alibaba Group for providing the cloud trace datasets
- Google for the cluster usage traces
- Microsoft Azure for the cloud infrastructure data
- The PyTorch team for the excellent deep learning framework

## 📞 Contact

For questions and support, please open an issue in this repository or contact:

- **Email**: [1637519447@qq.com]
- **GitHub**: [@1637519447](https://github.com/1637519447)

---

**Note**: This research is conducted for academic purposes. Please ensure compliance with data usage policies when working with cloud trace datasets.