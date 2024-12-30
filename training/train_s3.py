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
from torch.amp import autocast, GradScaler
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
from lr_finder import LRFinder
import logging
from datetime import datetime
import math
import time
from botocore.exceptions import ClientError

print(f'Path: {sys.path}\t System version:{sys.version}\t Torch Version: {torch.__version__} \n ffcv version: {ffcv.__version__}')

class S3Handler:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_CONFIG['access_key_id'],
            aws_secret_access_key=AWS_CONFIG['secret_access_key'],
            region_name=AWS_CONFIG['region_name'],
            config=boto3.session.Config(
                max_pool_connections=50,
                retries={'max_attempts': 5},
                connect_timeout=60,
                read_timeout=60
            )
        )
        self.bucket = AWS_CONFIG['bucket_name']
    
    def download_dataset(self, s3_key, local_path):
        """Download a file from S3"""
        print(f"Downloading from S3: s3://{self.bucket}/{s3_key}")
        try:
            # Create directory if local_path contains a directory structure
            directory = os.path.dirname(local_path)
            if directory:  # Only create directory if there's a path
                os.makedirs(directory, exist_ok=True)
                
            self.s3.download_file(self.bucket, s3_key, local_path)
        except Exception as e:
            print(f"Error downloading from S3: {str(e)}")
            print(f"Attempted to download from: s3://{self.bucket}/{s3_key}")
            raise
    
    def upload_model(self, local_path, s3_key):
        """Upload artifacts to S3 with better error handling"""
        try:
            print(f"Uploading to S3: s3://{self.bucket}/{s3_key}")
            
            # Verify file exists and is readable
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            # Get file size
            file_size = os.path.getsize(local_path)
            
            # For large files, use multipart upload
            if file_size > 100 * 1024 * 1024:  # 100MB
                self._multipart_upload(local_path, s3_key)
            else:
                # For smaller files, use single upload with ExtraArgs
                with open(local_path, 'rb') as data:
                    self.s3.upload_fileobj(
                        data, 
                        self.bucket, 
                        s3_key,
                        ExtraArgs={
                            'StorageClass': 'STANDARD',
                            'ContentType': 'application/octet-stream'
                        }
                    )
            
            print(f"Successfully uploaded {local_path} to s3://{self.bucket}/{s3_key}")
            
        except Exception as e:
            print(f"Error uploading to S3: {str(e)}")
            # Don't raise the exception to prevent training interruption
            return False
        return True
    
    def _multipart_upload(self, local_path, s3_key):
        """Handle multipart upload for large files"""
        # Initialize multipart upload
        mpu = self.s3.create_multipart_upload(
            Bucket=self.bucket,
            Key=s3_key,
            StorageClass='STANDARD',
            ContentType='application/octet-stream'
        )
        
        try:
            # Chunk size of 100MB
            chunk_size = 100 * 1024 * 1024
            file_size = os.path.getsize(local_path)
            chunks = int(math.ceil(file_size / chunk_size))
            
            parts = []
            
            with open(local_path, 'rb') as f:
                for i in range(chunks):
                    data = f.read(chunk_size)
                    part = self.s3.upload_part(
                        Bucket=self.bucket,
                        Key=s3_key,
                        PartNumber=i + 1,
                        UploadId=mpu['UploadId'],
                        Body=data
                    )
                    
                    parts.append({
                        'PartNumber': i + 1,
                        'ETag': part['ETag']
                    })
            
            # Complete multipart upload
            self.s3.complete_multipart_upload(
                Bucket=self.bucket,
                Key=s3_key,
                UploadId=mpu['UploadId'],
                MultipartUpload={'Parts': parts}
            )
            
        except Exception as e:
            print(f"Error in multipart upload: {str(e)}")
            # Abort the multipart upload
            self.s3.abort_multipart_upload(
                Bucket=self.bucket,
                Key=s3_key,
                UploadId=mpu['UploadId']
            )
            return False
        
        return True

def extract_chunk(args):
    """Extract a chunk of files from zip archive"""
    zip_path, chunk_files, temp_dir = args
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in chunk_files:
            zip_ref.extract(file, temp_dir)

def prepare_data(zip_path, data_dir, s3_handler=None):
    """Unzip and prepare ImageNet data with caching support"""
    cache_dir = os.path.join(data_dir, 'processed_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if processed data exists in cache
    train_cache = os.path.join(cache_dir, 'train')
    val_cache = os.path.join(cache_dir, 'val')
    
    # Try to download cached data from S3 first
    if s3_handler:
        try:
            print("Checking for cached data in S3...")
            s3_handler.download_dataset('processed_data/train.tar', f'{train_cache}.tar')
            s3_handler.download_dataset('processed_data/val.tar', f'{val_cache}.tar')
            
            # Extract cached data
            print("Extracting cached data...")
            subprocess.run(['tar', '-xf', f'{train_cache}.tar', '-C', cache_dir])
            subprocess.run(['tar', '-xf', f'{val_dir}.tar', '-C', cache_dir])
            
            print("Using cached data from S3")
            return train_cache, val_cache
            
        except Exception as e:
            print(f"No cached data found in S3 or error downloading: {e}")
            print("Processing from scratch...")
    
    # If no cached data, process from scratch
    temp_dir = os.path.join(data_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Set up paths based on your dataset structure
    data_root = os.path.join(temp_dir, 'Data', 'CLS-LOC')
    train_path = os.path.join(data_root, 'train')
    val_path = os.path.join(data_root, 'val')
    
    # Load class mapping
    synset_mapping = {}
    mapping_file = os.path.join(temp_dir, 'LOC_synset_mapping.txt')
    with open(mapping_file, 'r') as f:
        for line in f:
            synset, label = line.strip().split(' ', 1)
            synset_mapping[synset] = label
    
    # Create destination directories
    train_dir = os.path.join(cache_dir, 'train')
    val_dir = os.path.join(cache_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    for synset in tqdm(os.listdir(train_path)):
        src_dir = os.path.join(train_path, synset)
        dst_dir = os.path.join(train_dir, synset)
        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    
    # Process validation data
    print("Processing validation data...")
    val_annotations = {}
    
    # Read validation ground truth
    val_ground_truth = os.path.join(temp_dir, 'LOC_val_solution.csv')
    if os.path.exists(val_ground_truth):
        print("Using validation ground truth file...")
        with open(val_ground_truth, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    img_name = parts[0]
                    # Extract synset from the annotation format
                    synset = parts[1].split()[0].strip()
                    if synset.startswith('n'):  # Verify it's a valid synset ID
                        val_annotations[img_name] = synset
    else:
        print("Validation ground truth file not found!")
        # Try to find val_loc.txt
        val_loc = os.path.join(temp_dir, 'ImageSets', 'CLS-LOC', 'val_loc.txt')
        if os.path.exists(val_loc):
            print("Using val_loc.txt...")
            with open(val_loc, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name = parts[0]
                        synset = parts[1]
                        val_annotations[img_name] = synset
        else:
            raise FileNotFoundError("Could not find validation annotations!")

    print(f"Found {len(val_annotations)} validation annotations")

    # Create validation directory structure
    print("Organizing validation images into class folders...")
    val_path = os.path.join(data_root, 'val')
    
    # Create validation class directories
    unique_synsets = set(val_annotations.values())
    for synset in unique_synsets:
        os.makedirs(os.path.join(val_dir, synset), exist_ok=True)

    # Copy validation images to their respective class folders
    missing_images = []
    for img_name, synset in tqdm(val_annotations.items()):
        # Try different possible image extensions
        found = False
        for ext in ['.JPEG', '.jpeg', '.jpg']:
            src_path = os.path.join(val_path, img_name + ext)
            if os.path.exists(src_path):
                dst_dir = os.path.join(val_dir, synset)
                dst_path = os.path.join(dst_dir, img_name + ext)
                try:
                    shutil.copy2(src_path, dst_path)
                    found = True
                    break
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
        
        if not found:
            missing_images.append(img_name)

    if missing_images:
        print(f"Warning: Could not find {len(missing_images)} validation images")
        print("First few missing images:", missing_images[:5])

    # Verify validation data structure
    print("\nVerifying validation data structure...")
    val_classes = os.listdir(val_dir)
    print(f"Number of validation classes: {len(val_classes)}")
    
    class_counts = {}
    for class_dir in val_classes:
        class_path = os.path.join(val_dir, class_dir)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts[class_dir] = count
    
    total_val_images = sum(class_counts.values())
    print(f"Total validation images: {total_val_images}")
    
    # Print distribution statistics
    counts = list(class_counts.values())
    if counts:
        print(f"Images per class - Min: {min(counts)}, Max: {max(counts)}, "
              f"Average: {sum(counts)/len(counts):.1f}")
        
        # Print classes with very few images
        small_classes = {k: v for k, v in class_counts.items() if v < 10}
        if small_classes:
            print("\nWarning: Classes with less than 10 images:")
            for cls, count in small_classes.items():
                print(f"{cls}: {count} images")

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of training classes: {len(os.listdir(train_dir))}")
    print(f"Number of validation classes: {len(os.listdir(val_dir))}")
    
    train_samples = sum(len(os.listdir(os.path.join(train_dir, d))) 
                       for d in os.listdir(train_dir))
    val_samples = sum(len(os.listdir(os.path.join(val_dir, d))) 
                     for d in os.listdir(val_dir))
    
    print(f"Total training samples: {train_samples}")
    print(f"Total validation samples: {val_samples}")
    
    # Create tar archives of processed data
    print("\nCreating tar archives of processed data...")
    subprocess.run(['tar', '-cf', f'{train_dir}.tar', '-C', cache_dir, 'train'])
    subprocess.run(['tar', '-cf', f'{val_dir}.tar', '-C', cache_dir, 'val'])
    
    # Upload processed data to S3 if handler provided
    if s3_handler:
        print("Uploading processed data to S3...")
        s3_handler.upload_model(f'{train_dir}.tar', 'processed_data/train.tar')
        s3_handler.upload_model(f'{val_dir}.tar', 'processed_data/val.tar')
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    print("\nDataset preparation completed!")
    
    return train_dir, val_dir

def write_ffcv_dataset(data_dir, write_path):
    """Convert dataset to FFCV format with optimizations for ImageNet"""
    print(f"Converting {data_dir} to FFCV format...")
    
    # Verify directory structure
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    # Check for class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {data_dir}")
    
    print(f"Found {len(class_dirs)} classes")
    
    # Verify each class has images
    empty_classes = []
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        if not os.listdir(class_path):
            empty_classes.append(class_dir)
    
    if empty_classes:
        raise ValueError(f"Found {len(empty_classes)} empty classes: {empty_classes}")
    
    # Create ImageFolder dataset
    try:
        dataset = ImageFolder(data_dir)
    except Exception as e:
        raise ValueError(f"Error creating ImageFolder dataset: {str(e)}")
    
    print(f"Dataset size: {len(dataset)} images")
    
    # Configure FFCV writer
    writer = DatasetWriter(
        write_path, 
        {
            'image': RGBImageField(
                max_resolution=256,
                jpeg_quality=95
            ),
            'label': IntField()
        },
        num_workers=min(32, os.cpu_count())
    )
    
    # Write dataset with progress bar and error handling
    total = len(dataset)
    try:
        with tqdm(total=total, desc="Writing FFCV dataset") as pbar:
            writer.from_indexed_dataset(
                dataset,
                chunksize=100,
                shuffle_indices=True
            )
            pbar.update(total)
    except Exception as e:
        # Clean up partial write
        if os.path.exists(write_path):
            os.remove(write_path)
        raise ValueError(f"Error writing FFCV dataset: {str(e)}")

def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)
    
    # Ensure all processes are ready
    dist.barrier()

def create_ffcv_loaders(train_path, val_path, batch_size=1024, rank=0, world_size=1):
    """Create FFCV data loaders optimized for g6.12xlarge"""
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
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),  # Normalize in fp32 first
        Convert(torch.float16)  # Convert to fp16 after normalization
    ]

    val_image_pipeline = [
        CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.8),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(channels_last=True),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),  # Normalize in fp32 first
        Convert(torch.float16)  # Convert to fp16 after normalization
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        Convert(torch.long)
    ]
    
    # Optimize batch size per GPU (1024/4 = 256 per GPU)
    per_gpu_batch_size = batch_size // world_size
    
    # Optimize workers for g6.12xlarge (48 vCPUs / 4 GPUs = 12 workers per GPU)
    num_workers = 12
    
    train_loader = Loader(
        train_path,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
        order=OrderOption.RANDOM,
        drop_last=True,
        pipelines={
            'image': train_image_pipeline,
            'label': label_pipeline
        },
        os_cache=True,
        distributed=world_size > 1,
        seed=42
    )
    
    val_loader = Loader(
        val_path,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines={
            'image': val_image_pipeline,
            'label': label_pipeline
        },
        os_cache=True
    )
    
    return train_loader, val_loader

def train_model(rank, world_size, train_path, val_path, num_epochs=40, resume_path=None, save_freq=2, patience=3):
    """Train model with optimizations for g6.12xlarge"""
    # Initialize process group first
    setup_ddp(rank, world_size)
    
    # Create S3 handler inside the process
    s3_handler = S3Handler()
    
    # Create data loaders after DDP setup
    train_loader, val_loader = create_ffcv_loaders(
        train_path, 
        val_path, 
        batch_size=TRAINING_CONFIG['batch_size'],
        rank=rank,
        world_size=world_size
    )
    
    logger = logging.getLogger('training')
    
    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Create model
    model = models.resnet50(weights=None)
    model = model.to(rank)
    model = model.to(memory_format=torch.channels_last)
    
    # Create criterion here
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    if world_size > 1:
        # Configure DDP for better performance
        model = DDP(model, 
                   device_ids=[rank],
                   find_unused_parameters=False,  # Disable unused parameter detection
                   broadcast_buffers=False)  # Disable buffer broadcasting
    
    # Update GradScaler initialization
    scaler = GradScaler(
        enabled=True,
        init_scale=2**16,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000,
        device='cuda'  # Specify device explicitly
    )
    
    # Use SGD with Nesterov momentum for better convergence
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                 TRAINING_CONFIG['gradient_clip_val'])
    
    # Use OneCycleLR with warmup
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=TRAINING_CONFIG['learning_rate'],
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=TRAINING_CONFIG['warmup_epochs'] / num_epochs,
        div_factor=25,
        final_div_factor=1e4
    )
    
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
    
    model.train()
    model.requires_grad_(True)
    
    # Log training configuration
    logger.info(f"Training Configuration:")
    logger.info(f"Mixed Precision: Enabled (AMP with float16)")
    logger.info(f"Number of GPUs: {world_size}")
    logger.info(f"Batch size per GPU: {TRAINING_CONFIG['batch_size'] // world_size}")
    logger.info(f"Total batch size: {TRAINING_CONFIG['batch_size']}")
    logger.info(f"Number of workers per GPU: {TRAINING_CONFIG['num_workers']}")
    
    # Track AMP scaling stats
    amp_stats = {
        'overflow_count': 0,
        'scale': scaler.get_scale()
    }
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for i, (images, labels) in enumerate(pbar):
            if i % 10 == 0:
                torch.cuda.empty_cache()
                current_scale = scaler.get_scale()
                if current_scale != amp_stats['scale']:
                    logger.info(f"AMP scale changed: {amp_stats['scale']} → {current_scale}")
                    amp_stats['scale'] = current_scale
            
            labels = labels.squeeze()
            
            # Update autocast usage
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Rest of training loop remains the same
            scaler.scale(loss).backward()
            
            if not scaler.step(optimizer):
                amp_stats['overflow_count'] += 1
                # logger.warning(f"Gradient overflow detected (count: {amp_stats['overflow_count']})")
            
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
        
        # Log epoch statistics
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': 100. * correct / total,
            'amp_scale': scaler.get_scale(),
            'amp_overflow_count': amp_stats['overflow_count'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        logger.info(
            f"Epoch {epoch_stats['epoch']}/{num_epochs} - "
            f"Train Loss: {epoch_stats['train_loss']:.4f} - "
            f"Train Acc: {epoch_stats['train_acc']:.2f}% - "
            f"LR: {epoch_stats['learning_rate']:.6f} - "
            f"AMP Scale: {epoch_stats['amp_scale']} - "
            f"Overflow Count: {epoch_stats['amp_overflow_count']}"
        )
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
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
            
            # Save model locally first
            save_path = 'best_model.pth'
            try:
                torch.save(checkpoint, save_path)
                # Only upload if save was successful
                if os.path.exists(save_path):
                    if not s3_handler.upload_model(save_path, 'models/best_model.pth'):
                        print("Warning: Failed to upload model to S3, but continuing training")
            except Exception as e:
                print(f"Error saving model: {e}")
                # Continue training even if save fails
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
        
        # Upload logs to S3 periodically
        if (epoch + 1) % save_freq == 0:
            log_files = [f for f in os.listdir('.') if f.endswith('.log')]
            for log_file in log_files:
                s3_handler.upload_model(
                    log_file, 
                    f'training_logs/{log_file}'
                )
        
        # Log GPU memory usage
        if rank == 0:  # Only log from main process
            log_gpu_memory_stats(logger, rank)
    
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

def find_lr(model, train_loader, criterion, device):
    """Performs the learning rate range test"""
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.9, weight_decay=1e-4)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    
    print("Running learning rate range test...")
    lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100, step_mode="exp")
    
    # Plot the results
    lr_finder.plot(skip_start=10, skip_end=5)
    
    # Get the suggested learning rate
    suggested_lr = lr_finder.get_suggested_lr()
    
    # Reset the model and optimizer to their initial states
    lr_finder.reset()
    
    return suggested_lr

def setup_logging():
    """Configure logging to both file and console"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'training_{timestamp}.log'
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_gpu_memory_stats(logger, rank):
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(rank) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(rank) / 1024**2
        max_memory_allocated = torch.cuda.max_memory_allocated(rank) / 1024**2
        
        logger.info(
            f"GPU:{rank} Memory (MB) - "
            f"Allocated: {memory_allocated:.1f} - "
            f"Reserved: {memory_reserved:.1f} - "
            f"Peak: {max_memory_allocated:.1f}"
        )

def main():
    try:
        # Print system information
        print_system_info()
        s3_handler = S3Handler()
        
        # Create data directory
        os.makedirs('data', exist_ok=True)

        # Download FFCV files from S3
        print("Downloading FFCV files from S3...")
        try:
            # Download from the correct S3 paths
            s3_handler.download_dataset('processed_data/train.beton', 'train.beton')
            s3_handler.download_dataset('processed_data/val.beton', 'val.beton')
            print("Successfully downloaded FFCV files from S3")
        except Exception as e:
            print(f"Error downloading FFCV files from S3: {e}")
            raise  # Stop execution if we can't get the required files

        # Start training
        if not torch.cuda.is_available():
            print("CUDA is not available. Running on CPU only.")
            return

        gpu_count = torch.cuda.device_count()
        print(f"\nFound {gpu_count} GPUs!")

        if gpu_count > 1:
            print(f"Using DistributedDataParallel across {gpu_count} GPUs")
            try:
                mp.spawn(
                    train_model,
                    args=(
                        gpu_count,
                        'train.beton',
                        'val.beton',
                        TRAINING_CONFIG['num_epochs'],
                        None,  # resume_path
                        2,     # save_freq
                        3      # patience
                    ),
                    nprocs=gpu_count,
                    join=True
                )
            except Exception as e:
                print(f"Error in DDP training: {str(e)}")
                # Clean up in case of error
                if dist.is_initialized():
                    dist.destroy_process_group()
                raise
        else:
            print("Running on single GPU")
            
            # Create model and move to GPU
            model = models.resnet50(weights=None)
            model = model.to(0)
            model = model.to(memory_format=torch.channels_last)
            
            # Create data loaders
            train_loader, val_loader = create_ffcv_loaders(
                'train.beton', 
                'val.beton', 
                batch_size=TRAINING_CONFIG['batch_size']
            )
            
            # Add LR finder before training
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            suggested_lr = find_lr(model, train_loader, criterion, device=torch.device('cuda:0'))
            print(f"Suggested learning rate: {suggested_lr}")
            
            s3_handler = S3Handler()  # Create S3 handler here for single GPU case
            
            # Upload the LR finder plot to S3
            if os.path.exists('lr_finder_plot.png'):
                s3_handler.upload_model('lr_finder_plot.png', 'training_artifacts/lr_finder_plot.png')
            
            # Update the learning rate in your training configuration
            TRAINING_CONFIG['learning_rate'] = suggested_lr
            
            # Start training
            train_model(
                0,              # rank
                1,              # world_size
                'train.beton',
                'val.beton',
                num_epochs=TRAINING_CONFIG['num_epochs'],
                resume_path=None,
                save_freq=2,
                patience=3
            )

    finally:
        # Cleanup temporary files but keep the FFCV files
        for file in ['dataset.zip', 'downloaded_checkpoint.pth', 'lr_finder_plot.png']:
            if os.path.exists(file):
                os.remove(file)
        # Only remove the data directory if FFCV files were successfully created
        if os.path.exists('train.beton') and os.path.exists('val.beton') and os.path.exists('data'):
            shutil.rmtree('data')

if __name__ == '__main__':
    logger = setup_logging()
    main() 