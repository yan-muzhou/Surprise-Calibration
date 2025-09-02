# Surprise Calibration for Better In-Context Learning

This repository contains the code implementation for the paper "[Surprise Calibration for Better In-Context Learning](https://arxiv.org/abs/2506.12796)".

## Overview

In-context learning (ICL) has emerged as a powerful paradigm for adapting large language models (LLMs) to specific tasks using only input-output examples in prompts, without parameter updates. However, LLMs often suffer from miscalibration in ICL scenarios, producing overconfident predictions even with incorrect predictions. 

This project introduces Surprise Calibration (SC), a novel calibration method that leverages surprise signals from prior knowledge probing to enhance calibration in ICL while maintaining or improving predictive performance.

## Features

- Implementation of Surprise Calibration (SC) method for ICL
- Support for various sampling strategies (BM25, GTE, random)
- Multiple ranking strategies (U-curve, increasing, decreasing, random)
- Flexible configuration for different datasets and models
- Extensive evaluation framework

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd ICL_calibration_git

# Install required packages
pip install -r requirements.txt
```

## Project Structure

```
ICL_calibration_git/
├── config.py          # Configuration settings
├── main.py            # Main execution script
├── model.py           # Model interface and generation functions
├── sample.py          # Sampling strategies implementation
├── utils.py           # Utility functions
├── SC.ipynb           # Analysis notebook
└── README.md          # This file
```

## Usage

### Basic Command

```bash
python main.py --dataset <dataset_name> --model <model_name> --num_sample <sample_count>
```

### Parameters

- `--dataset`: Dataset name (e.g., SST-2, MRPC, RTE)
- `--model`: Model name (e.g., qwen_7b, llama_8b)
- `--num_sample`: Number of samples to use for prompt generation
- `--sample_strategy`: Sampling strategy (top_bm25, top_gte, random)
- `--rank_strategy`: Ranking strategy (Ucurve, increase, decrease, random)
- `--do_probe`: Whether to perform probe
- `--do_predict`: Whether to perform prediction

### Example

```bash
# Run prediction with Qwen-7B model on SST-2 dataset
python main.py --dataset SST-2 --model qwen_7b --num_sample 60 --sample_strategy top_bm25 --rank_strategy Ucurve --do_predict

# Run with probe
python main.py --dataset SST-2 --model qwen_7b --num_sample 60 --sample_strategy top_bm25 --rank_strategy Ucurve --do_probe --do_predict
```

## Supported Models

- Qwen series (0.5B, 1.5B, 3B, 7B, 14B, 72B)
- LLaMA 3 8B

## Supported Datasets

- SST-2 (Sentiment Analysis)
- MRPC (Paraphrase Detection)
- RTE (Textual Entailment)
- QNLI (Question Natural Language Inference)
- WiC (Word-in-Context)
- MNLI (Multi-Genre Natural Language Inference)
- AG-news (Text Classification)
- YouTube (Spam Detection)
- AI-GA (AI-generated Text Detection)

## Configuration

The [config.py](config.py) file contains the base configuration including:
- Model mappings
- Dataset-specific verb mappings
- Special token configurations
- File paths and output directories

## Methodology

The Surprise Calibration method works through the following steps:

1. **Prior Knowledge Probing**: Probe the model with prior knowledge examples to obtain surprise signals
2. **Confidence Estimation**: Calculate confidence scores for different labels
3. **Linear Calibration**: Apply linear transformation to calibrate predictions
4. **Prediction**: Generate final calibrated predictions

## SC Core Implementation

The core implementation of the Surprise Calibration method is in [SC.ipynb](SC.ipynb), which includes:

### Key Components

1. **Transformer-based Encoder Model**:
   - Uses positional encoding to capture sequence information
   - Implements Transformer encoder layers for processing support examples
   - Aggregates information to predict calibration adjustments

2. **RNN-based Models**:
   - LSTM and GRU models for sequential processing of support examples
   - Layer normalization for stable training
   - Orthogonal initialization for better convergence

3. **Calibration Pipeline**:
   - Data preprocessing and batch processing
   - Training and evaluation loops
   - Memory and time monitoring

### Core Functions

- `multiclass_pointwise_surprise()`: Computes surprise signals from prior knowledge probing
- `data_reader()`: Processes and prepares data for training and testing
- `train_model()` and `test_model()`: Training and evaluation functions
- `GRUModel`, `LSTMModel`, `Transformer_encoder`: Neural network architectures for calibration

### Supported Datasets and Evaluation

The implementation supports multiple datasets and provides comprehensive evaluation:
- MNLI, SST-2, MRPC, QNLI, RTE, WiC, YouTube, AI-GA_1-1
- Multi-seed experiments for robust evaluation
- GPU memory and inference time monitoring

## Results

Our method achieves superior calibration performance while maintaining or improving prediction accuracy across various benchmarks and model sizes.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{tan2025surprise,
  title={Surprise Calibration for Better In-Context Learning},
  author={Tan, Zhihang and Hou, Jingrui and Wang, Ping and Hu, Qibiao and Zhu, Peng},
  journal={arXiv preprint arXiv:2506.12796},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.