# ImageNet Training Pipeline with FFCV

A high-performance distributed training pipeline for ImageNet using FFCV, optimized for AWS g6.12xlarge instances with A10G GPUs.

## Overview

This project implements a distributed training pipeline for ResNet-50 on ImageNet, with the following key features:

- Optimized for AWS g6.12xlarge instances (4x NVIDIA A10G GPUs)
- Uses FFCV for high-speed data loading
- Implements mixed-precision training (FP16)
- Includes automatic learning rate finding
- Supports distributed training across multiple GPUs
- Integrates with S3 for data and model storage

## Prerequisites

- Ubuntu 20.04 or later
- NVIDIA drivers supporting CUDA 12.1
- Python 3.8 or later
- AWS credentials configured
- At least 500GB storage space

## Installation

1. Clone the repository:

   git clone `<repository-url>`

   cd `<repository-name>`
2. Run the setup script:

   cd training

   chmod +x setup.sh

   ./setup.sh

## Project Structure

training/

├── check_cuda.py # CUDA availability checker

├── config.py # Configuration settings

├── lr_finder.py # Learning rate range test implementation

├── requirements.txt # Python dependencies

├── setup.sh # Environment setup script

└── train_s3.py # Main training script

## Configuration

Edit `config.py` to set:

- AWS credentials and S3 bucket details
- Training hyperparameters
- Hardware-specific optimizations

  ```
  AWS_CONFIG = {
  'bucket_name': 'your-bucket',
  'region_name': 'your-region',
  'access_key_id': 'your-access-key',
  'secret_access_key': 'your-secret-key'
  }
  TRAINING_CONFIG = {
  'batch_size': 1024, # 256 per GPU for 4 GPUs
  'num_epochs': 40,
  'num_workers': 12,
  'learning_rate': 0.1,
  'warmup_epochs': 5
  }
  ```

## Training Pipeline Features

1. **Automatic Learning Rate Finding**

   - Implements the learning rate range test
   - Generates visualization of optimal learning rate
   - Automatically sets learning rate for training
2. **Optimized Data Loading**

   - FFCV for high-speed data loading
   - Parallel data loading with optimal worker allocation
   - Mixed-precision data processing
3. **Distributed Training**

   - Multi-GPU training using DistributedDataParallel
   - Automatic process group initialization
   - Efficient gradient synchronization
4. **Performance Optimizations**

   - Mixed precision training (FP16)
   - CUDA graphs support (configurable)
   - Gradient clipping
   - Memory-efficient data handling
5. **Monitoring and Checkpointing**

   - Regular model checkpointing
   - S3 integration for artifact storage
   - Training metrics logging
   - GPU memory usage tracking

## Usage

1. **Prepare Data**

   - Upload ImageNet dataset to S3
   - Configure S3 bucket details in `config.py`
2. **Start Training**

```bash
python train_s3.py
```

3. **Monitor Training**
   - Check logs in `training_*.log`
   - Monitor GPU usage with `nvidia-smi`
   - View learning rate finder plot in `lr_finder_plot.png`

## Performance Tips

1. **GPU Optimization**

   - Set GPU power limit: `sudo nvidia-smi -pl 250`
   - Enable persistence mode: `sudo nvidia-smi -pm 1`
   - Set optimal GPU clocks: `sudo nvidia-smi -ac 1215,1410`
2. **System Optimization**

   - Adjust system parameters in `setup.sh`
   - Optimize number of workers based on CPU cores
   - Monitor memory usage and swap

## Troubleshooting

1. **CUDA Out of Memory**

   - Reduce batch size
   - Enable gradient accumulation
   - Check GPU memory usage patterns
2. **Slow Data Loading**

   - Verify FFCV installation
   - Adjust number of workers
   - Check disk I/O performance
3. **S3 Issues**

   - Verify AWS credentials
   - Check S3 bucket permissions
   - Monitor network connectivity
