import os
import torch
import torch.nn as nn
import boto3
import zipfile
import ffcv
import cupy as cp
from sklearn.model_selection import train_test_split
from ffcv.fields import RGBImageField, IntField
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder, IntDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, RandomHorizontalFlip
from ffcv.transforms import NormalizeImage, RandomResizedCrop
from ffcv.writer import DatasetWriter
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models
from tqdm import tqdm
import shutil
import numpy as np
import sys
from config import AWS_CONFIG, TRAINING_CONFIG
from ffcv.transforms import Convert
from ffcv.transforms import ToTorchImage
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
import platform
import requests
import subprocess

print(f'Path: {sys.path}\t System version:{sys.version}\t Torch Version: {torch.__version__} \n ffcv version: {ffcv.__version__}')

class S3Handler:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_CONFIG['access_key_id'],
            aws_secret_access_key=AWS_CONFIG['secret_access_key'],
            region_name=AWS_CONFIG['region_name']
        )
        self.bucket = AWS_CONFIG['bucket_name']
    
    def download_dataset(self, s3_key, local_path):
        print(f"Downloading from S3: s3://{self.bucket}/{s3_key}")
        self.s3.download_file(self.bucket, s3_key, local_path)
    
    def upload_model(self, local_path, s3_key):
        """Upload model artifacts to S3 with organized structure"""
        s3_path = f'model_artifacts/{s3_key}'  # Organize in model_artifacts folder
        print(f"Uploading to S3: s3://{self.bucket}/{s3_path}")
        self.s3.upload_file(local_path, self.bucket, s3_path)

def prepare_data(zip_path, data_dir):
    """Unzip and split data into train/val"""
    # Unzip dataset
    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm(files, desc="Extracting files"):
            zip_ref.extract(file, data_dir)
    
    # Get all image paths and labels
    dataset = ImageFolder(os.path.join(data_dir, os.listdir(data_dir)[0]))
    images = [x[0] for x in dataset.samples]
    labels = [x[1] for x in dataset.samples]
    
    # Count samples per class
    from collections import Counter
    label_counts = Counter(labels)
    print(f"Class distribution: {label_counts}")
    
    # Filter out classes with less than 2 samples
    valid_classes = [cls for cls, count in label_counts.items() if count >= 2]
    print(f"Number of valid classes (with >=2 samples): {len(valid_classes)}")
    
    # Filter dataset to only include valid classes
    valid_indices = [i for i, label in enumerate(labels) if label in valid_classes]
    filtered_images = [images[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    
    # Remap labels to be consecutive integers starting from 0
    label_map = {old: new for new, old in enumerate(sorted(valid_classes))}
    filtered_labels = [label_map[label] for label in filtered_labels]
    
    print(f"Total samples after filtering: {len(filtered_images)}")
    
    # Split into train/val
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        filtered_images, filtered_labels, test_size=0.2, 
        stratify=filtered_labels, random_state=42
    )
    
    # Create train/val directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Move files to respective directories with progress bars
    print("Preparing training data...")
    for img, label in tqdm(zip(train_imgs, train_labels), total=len(train_imgs), desc="Copying training files"):
        target_dir = os.path.join(train_dir, str(label))
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(img, target_dir)
    
    print("Preparing validation data...")
    for img, label in tqdm(zip(val_imgs, val_labels), total=len(val_imgs), desc="Copying validation files"):
        target_dir = os.path.join(val_dir, str(label))
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(img, target_dir)
    
    # Save label mapping for reference
    import json
    with open(os.path.join(data_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_map, f, indent=2)
    
    return train_dir, val_dir

def write_ffcv_dataset(data_dir, write_path):
    """Convert dataset to FFCV format"""
    print(f"Converting {data_dir} to FFCV format...")
    dataset = ImageFolder(data_dir)
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(
            max_resolution=256,
            jpeg_quality=95
        ),
        'label': IntField()
    }, num_workers=8)
    
    # Convert PIL Images to numpy arrays with progress bar
    total = len(dataset)
    with tqdm(total=total, desc="Writing FFCV dataset") as pbar:
        writer.from_indexed_dataset(dataset)
        pbar.update(total)

def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def create_ffcv_loaders(train_path, val_path, batch_size=512, rank=0, world_size=1):
    """Create FFCV data loaders with DDP support"""
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255.0
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255.0
    
    # Update device assignment to use specific GPU
    device = torch.device(f'cuda:{rank}')
    
    # Modified pipelines to use specific GPU
    train_image_pipeline = [
        RandomResizedCropRGBImageDecoder(output_size=(224,224)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(channels_last=True),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]

    val_image_pipeline = [
        CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.8),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(channels_last=True),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        Convert(torch.long)
    ]
    
    # Adjust batch size per GPU
    per_gpu_batch_size = batch_size // world_size if world_size > 1 else batch_size
    
    train_loader = Loader(
        train_path,
        batch_size=per_gpu_batch_size,
        num_workers=4,
        order=OrderOption.RANDOM,
        drop_last=True,
        pipelines={
            'image': train_image_pipeline,
            'label': label_pipeline
        },
        os_cache=False
    )
    
    val_loader = Loader(
        val_path,
        batch_size=per_gpu_batch_size,
        num_workers=4,
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines={
            'image': val_image_pipeline,
            'label': label_pipeline
        },
        os_cache=False
    )
    
    return train_loader, val_loader

def train_model(rank, world_size, train_loader, val_loader, s3_handler, num_epochs=10, resume_path=None, save_freq=2, patience=3):
    """Train model with DDP support"""
    setup_ddp(rank, world_size)
    
    # Rest of the setup remains similar, but move model to specific GPU
    model = models.resnet50(weights=None)
    model = model.to(rank)
    model = model.to(memory_format=torch.channels_last)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    scaler = torch.amp.GradScaler(device=f'cuda:{rank}')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # Initialize variables for training
    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0
    best_loss = float('inf')
    
    # Resume training if checkpoint provided
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from epoch {start_epoch} with best accuracy: {best_acc:.2f}%")
    
    scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=num_epochs,
                          steps_per_epoch=len(train_loader))
    
    model.train()
    model.requires_grad_(True)
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for i, (images, labels) in enumerate(pbar):
            if i % 10 == 0:
                torch.cuda.empty_cache()
            
            labels = labels.squeeze()
                
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss/(i+1), 
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, labels in val_pbar:
                labels = labels.squeeze()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': val_loss/(total/labels.size(0)), 
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        val_acc = 100.*correct/total
        avg_val_loss = val_loss/len(val_loader)
        print(f'Validation Accuracy: {val_acc:.2f}% | Loss: {avg_val_loss:.4f}')
        
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_acc': val_acc,
            'best_acc': best_acc
        }
        
        # Save checkpoint based on save_freq
        if (epoch + 1) % save_freq == 0:
            torch.save(checkpoint, f'checkpoint_epoch{epoch+1}.pth')
            s3_handler.upload_model(f'checkpoint_epoch{epoch+1}.pth', 
                                  f'models/checkpoints/checkpoint_epoch{epoch+1}.pth')
        
        # Save best model and check for early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(checkpoint, 'best_model.pth')
            s3_handler.upload_model('best_model.pth', 'models/best_model.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
    
    # Save final model
    torch.save(checkpoint, 'final_model.pth')
    s3_handler.upload_model('final_model.pth', 'models/final_model.pth')

    dist.destroy_process_group()

def print_system_info():
    print("\nSystem Information:")
    
    # Get AWS instance information
    try:
        # Get instance type from AWS metadata
        instance_response = requests.get("http://169.254.169.254/latest/meta-data/instance-type", timeout=2)
        instance_type = instance_response.text
        print(f"AWS Instance Type: {instance_type}")
    except:
        print("Could not determine AWS instance type")
    
    # Get GPU information using nvidia-smi
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
        print("\nNVIDIA-SMI Output:")
        print(nvidia_smi.decode('utf-8'))
    except:
        print("nvidia-smi command failed")

    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"FFCV Version: {ffcv.__version__}")
    print(f"CPU Cores: {multiprocessing.cpu_count()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print("\nPyTorch GPU Detection:")
        print(f"Number of GPUs (torch.cuda.device_count): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"System Path: {sys.path}\n")

def main():
    try:
        # Print system information
        print_system_info()

        # Check CUDA availability first
        if not torch.cuda.is_available():
            print("CUDA is not available. Running on CPU only.")
            return

        # Get GPU information
        gpu_count = torch.cuda.device_count()
        print(f"GPU Information:")
        print(f"Number of GPUs detected: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Print worker information
        num_workers = min(multiprocessing.cpu_count(), 8)  # Usually good to limit to 8
        print(f"\nWorker Information:")
        print(f"CPU Cores Available: {multiprocessing.cpu_count()}")
        print(f"Number of workers to be used: {num_workers}")
        print()

        s3_handler = S3Handler()
        
        if gpu_count > 1:
            print(f"Multiple GPUs detected! Using DistributedDataParallel for training across {gpu_count} GPUs.")
            
            # Download and prepare data first
            s3_handler.download_dataset('train_archive.zip', 'dataset.zip')
            train_dir, val_dir = prepare_data('dataset.zip', 'data')
            write_ffcv_dataset(train_dir, 'train.beton')
            write_ffcv_dataset(val_dir, 'val.beton')
            
            # Launch distributed training
            mp.spawn(
                train_model,
                args=(
                    gpu_count,  # world_size = number of GPUs
                    'train.beton',
                    'val.beton',
                    s3_handler,
                    TRAINING_CONFIG['num_epochs'],
                    None,  # resume_path
                    2,     # save_freq
                    3      # patience
                ),
                nprocs=gpu_count,
                join=True
            )
        else:
            print("Single GPU detected. Running without distributed training.")
            # Single GPU training path
            s3_handler.download_dataset('train_archive.zip', 'dataset.zip')
            train_dir, val_dir = prepare_data('dataset.zip', 'data')
            write_ffcv_dataset(train_dir, 'train.beton')
            write_ffcv_dataset(val_dir, 'val.beton')
            
            # Create data loaders for single GPU
            train_loader, val_loader = create_ffcv_loaders(
                'train.beton', 
                'val.beton', 
                batch_size=TRAINING_CONFIG['batch_size']
            )
            
            # Train on single GPU
            train_model(
                0,              # rank
                1,              # world_size
                train_loader,
                val_loader,
                s3_handler,
                num_epochs=TRAINING_CONFIG['num_epochs'],
                save_freq=2,
                patience=3
            )

    finally:
        # Cleanup temporary files
        for file in ['dataset.zip', 'train.beton', 'val.beton']:
            if os.path.exists(file):
                os.remove(file)
        if os.path.exists('data'):
            shutil.rmtree('data')

if __name__ == '__main__':
    main() 