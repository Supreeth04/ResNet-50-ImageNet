(pytorch) ubuntu@ip-172-31-35-183:/opt/dlami/nvme/nvme/training$ python train_s3.py
Path: ['/opt/dlami/nvme/nvme/training', '/opt/conda/envs/pytorch/lib/python311.zip', '/opt/conda/envs/pytorch/lib/python3.11', '/opt/conda/envs/pytorch/lib/python3.11/lib-dynload', '/opt/conda/envs/pytorch/lib/python3.11/site-packages', '/opt/dlami/nvme/nvme/training/ffcv', '/tmp/tmpweo3sl88']   System version:3.11.11 | packaged by conda-forge | (main, Dec  5 2024, 14:17:24) [GCC 13.3.0]   Torch Version: 2.5.1
 ffcv version: 1.0.2

System Information:
AWS Instance Type:

NVIDIA-SMI Output:
Sun Dec 29 16:26:32 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L4                      On  |   00000000:38:00.0 Off |                    0 |
| N/A   59C    P8             17W /   72W |       4MiB /  23034MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA L4                      On  |   00000000:3A:00.0 Off |                    0 |
| N/A   54C    P8             17W /   72W |       4MiB /  23034MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA L4                      On  |   00000000:3C:00.0 Off |                    0 |
| N/A   59C    P8             17W /   72W |       4MiB /  23034MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA L4                      On  |   00000000:3E:00.0 Off |                    0 |
| N/A   54C    P8             17W /   72W |       4MiB /  23034MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

Python Version: 3.11.11
PyTorch Version: 2.5.1
FFCV Version: 1.0.2
CPU Cores: 48
CUDA Available: True
CUDA Version: 12.4
cuDNN Version: 90100

PyTorch GPU Detection:
Number of GPUs (torch.cuda.device_count): 4
GPU 0: NVIDIA L4
GPU 1: NVIDIA L4
GPU 2: NVIDIA L4
GPU 3: NVIDIA L4
System Path: ['/opt/dlami/nvme/nvme/training', '/opt/conda/envs/pytorch/lib/python311.zip', '/opt/conda/envs/pytorch/lib/python3.11', '/opt/conda/envs/pytorch/lib/python3.11/lib-dynload', '/opt/conda/envs/pytorch/lib/python3.11/site-packages', '/opt/dlami/nvme/nvme/training/ffcv', '/tmp/tmpweo3sl88']

Found 4 GPUs!
Using DistributedDataParallel across 4 GPUs
Path: ['/opt/dlami/nvme/nvme/training', '/opt/conda/envs/pytorch/lib/python311.zip', '/opt/conda/envs/pytorch/lib/python3.11', '/opt/conda/envs/pytorch/lib/python3.11/lib-dynload', '/opt/conda/envs/pytorch/lib/python3.11/site-packages', '/opt/dlami/nvme/nvme/training/ffcv', '/tmp/tmpweo3sl88', '/tmp/tmpor69onf5']       System version:3.11.11 | packaged by conda-forge | (main, Dec  5 2024, 14:17:24) [GCC 13.3.0]   Torch Version: 2.5.1
 ffcv version: 1.0.2
Path: ['/opt/dlami/nvme/nvme/training', '/opt/conda/envs/pytorch/lib/python311.zip', '/opt/conda/envs/pytorch/lib/python3.11', '/opt/conda/envs/pytorch/lib/python3.11/lib-dynload', '/opt/conda/envs/pytorch/lib/python3.11/site-packages', '/opt/dlami/nvme/nvme/training/ffcv', '/tmp/tmpweo3sl88', '/tmp/tmp0s9x_o8m']       System version:3.11.11 | packaged by conda-forge | (main, Dec  5 2024, 14:17:24) [GCC 13.3.0]   Torch Version: 2.5.1
 ffcv version: 1.0.2
Path: ['/opt/dlami/nvme/nvme/training', '/opt/conda/envs/pytorch/lib/python311.zip', '/opt/conda/envs/pytorch/lib/python3.11', '/opt/conda/envs/pytorch/lib/python3.11/lib-dynload', '/opt/conda/envs/pytorch/lib/python3.11/site-packages', '/opt/dlami/nvme/nvme/training/ffcv', '/tmp/tmpweo3sl88', '/tmp/tmptuyln36q']       System version:3.11.11 | packaged by conda-forge | (main, Dec  5 2024, 14:17:24) [GCC 13.3.0]   Torch Version: 2.5.1
 ffcv version: 1.0.2
Path: ['/opt/dlami/nvme/nvme/training', '/opt/conda/envs/pytorch/lib/python311.zip', '/opt/conda/envs/pytorch/lib/python3.11', '/opt/conda/envs/pytorch/lib/python3.11/lib-dynload', '/opt/conda/envs/pytorch/lib/python3.11/site-packages', '/opt/dlami/nvme/nvme/training/ffcv', '/tmp/tmpweo3sl88', '/tmp/tmp8l7jg72m']       System version:3.11.11 | packaged by conda-forge | (main, Dec  5 2024, 14:17:24) [GCC 13.3.0]   Torch Version: 2.5.1
 ffcv version: 1.0.2
[rank0]:[W1229 16:26:42.707809428 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
[rank1]:[W1229 16:26:42.715568090 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 1]  using GPU 1 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
[rank3]:[W1229 16:26:42.724143476 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 3]  using GPU 3 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
[rank2]:[W1229 16:26:42.724628757 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 2]  using GPU 2 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
Epoch 1/100 [Train]:   0%|                                                                                             | 0/1251 [00:00<?, ?it/s]/opt/conda/envs/pytorch/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:224: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn(
/opt/conda/envs/pytorch/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:224: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn(
/opt/conda/envs/pytorch/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:224: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn(
/opt/conda/envs/pytorch/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:224: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn(
Epoch 1/100 [Train]: 100%|████████████████████████████████████████████████████████████| 1251/1251 [11:57<00:00,  1.74it/s, loss=6.45, acc=1.90%]
Epoch 1/100 [Train]: 100%|████████████████████████████████████████████████████████████| 1251/1251 [11:57<00:00,  1.74it/s, loss=6.46, acc=1.92%]
Epoch 1/100 [Train]: 100%|████████████████████████████████████████████████████████████| 1251/1251 [11:57<00:00,  1.74it/s, loss=6.45, acc=1.91%]

Epoch 1/100 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.10it/s, loss=1.82, acc=5.77%]
Validation Accuracy: 5.77% | Loss: 5.7888
Epoch 1/100 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.04it/s, loss=1.82, acc=5.74%]
Validation Accuracy: 5.74% | Loss: 5.7906
Epoch 1/100 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:39<00:00,  4.98it/s, loss=1.82, acc=5.76%]
Validation Accuracy: 5.76% | Loss: 5.7922
Epoch 1/100 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:39<00:00,  4.96it/s, loss=1.82, acc=5.71%]
Validation Accuracy: 5.71% | Loss: 5.7947
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 2/100 [Train]: 100%|███████████████████████████████████████████████████████████| 1251/1251 [11:39<00:00,  1.79it/s, loss=5.38, acc=10.22%]

Epoch 2/100 [Train]: 100%|███████████████████████████████████████████████████████████| 1251/1251 [11:40<00:00,  1.79it/s, loss=5.38, acc=10.16%]

Epoch 2/100 [Val]: 100%|███████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.40it/s, loss=1.53, acc=15.63%]
Validation Accuracy: 15.63% | Loss: 4.8910
Epoch 2/100 [Val]: 100%|███████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.32it/s, loss=1.53, acc=15.55%]
Validation Accuracy: 15.55% | Loss: 4.8935
Epoch 2/100 [Val]: 100%|███████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=1.54, acc=15.42%]
Validation Accuracy: 15.42% | Loss: 4.9061
Epoch 2/100 [Val]: 100%|███████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.21it/s, loss=1.54, acc=15.56%]
Validation Accuracy: 15.56% | Loss: 4.8959
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch2.pth
Successfully uploaded checkpoint_epoch2.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch2.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded checkpoint_epoch2.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch2.pth
Epoch 3/100 [Train]: 100%|████████████████████████████████████████████████████████████| 1251/1251 [11:50<00:00,  1.76it/s, loss=4.6, acc=20.75%]^[[C
Epoch 3/100 [Train]: 100%|████████████████████████████████████████████████████████████| 1251/1251 [11:53<00:00,  1.75it/s, loss=4.6, acc=20.83%]
Epoch 3/100 [Train]: 100%|████████████████████████████████████████████████████████████| 1251/1251 [11:50<00:00,  1.76it/s, loss=4.6, acc=20.93%]
Epoch 3/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.38it/s, loss=1.32, acc=26.68%]
Validation Accuracy: 26.68% | Loss: 4.2015
Epoch 3/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.32it/s, loss=1.32, acc=26.69%]
Validation Accuracy: 26.69% | Loss: 4.2035
Epoch 3/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=1.32, acc=26.62%]
Validation Accuracy: 26.62% | Loss: 4.2056
Error in multipart upload: An error occurred (MalformedXML) when calling the CompleteMultipartUpload operation:
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 3/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.21it/s, loss=1.32, acc=26.66%]
Validation Accuracy: 26.66% | Loss: 4.2005
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 4/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [14:23<00:00,  1.45it/s, loss=4.08, acc=29.63%]
Epoch 4/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [14:22<00:00,  1.45it/s, loss=4.09, acc=29.45%]
Epoch 4/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.40it/s, loss=1.18, acc=34.29%]
Validation Accuracy: 34.29% | Loss: 3.7718
Epoch 4/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.36it/s, loss=1.18, acc=34.27%]
Validation Accuracy: 34.27% | Loss: 3.7718
Epoch 4/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.29it/s, loss=1.19, acc=34.06%]
Validation Accuracy: 34.06% | Loss: 3.7802
Epoch 4/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.23it/s, loss=1.19, acc=34.10%]
Validation Accuracy: 34.10% | Loss: 3.7802
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch4.pth
Successfully uploaded training_20241229_110903.log to s3://imagenet-data-170/training_logs/training_20241229_110903.log
Epoch 5/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [14:51<00:00,  1.40it/s, loss=3.75, acc=35.74%]
Epoch 5/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [14:53<00:00,  1.40it/s, loss=3.76, acc=35.73%]
Epoch 5/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [14:50<00:00,  1.41it/s, loss=3.75, acc=35.83%]

Epoch 5/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.42it/s, loss=1.09, acc=40.16%]
Validation Accuracy: 40.16% | Loss: 3.4912
Epoch 5/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.37it/s, loss=1.1, acc=40.07%]
Validation Accuracy: 40.07% | Loss: 3.4961
Epoch 5/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.34it/s, loss=1.1, acc=39.92%]
Validation Accuracy: 39.92% | Loss: 3.4977
Epoch 5/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=1.1, acc=39.99%]
Validation Accuracy: 39.99% | Loss: 3.4998
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 6/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=3.52, acc=40.52%]
Epoch 6/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=3.53, acc=40.34%]
Epoch 6/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=3.52, acc=40.55%]
Epoch 6/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.40it/s, loss=1.06, acc=42.20%]
Validation Accuracy: 42.20% | Loss: 3.3880
Epoch 6/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.33it/s, loss=1.06, acc=42.13%]
Validation Accuracy: 42.13% | Loss: 3.3939
Epoch 6/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=1.06, acc=42.15%]
Validation Accuracy: 42.15% | Loss: 3.3928
Epoch 6/100 [Val]: 100%|█████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.22it/s, loss=1.07, acc=41.92%]
Validation Accuracy: 41.92% | Loss: 3.4020
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch6.pth
Successfully uploaded checkpoint_epoch6.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch6.pth
Epoch 7/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [11:40<00:00,  1.79it/s, loss=3.36, acc=43.71%]
Epoch 7/100 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=3.36, acc=43.70%]
Epoch 7/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.40it/s, loss=1, acc=46.29%]
Validation Accuracy: 46.29% | Loss: 3.2029
Epoch 7/100 [Val]: 100%|███████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.33it/s, loss=1.01, acc=46.26%]
Validation Accuracy: 46.26% | Loss: 3.2103
Epoch 7/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.30it/s, loss=1, acc=46.40%]
Validation Accuracy: 46.40% | Loss: 3.1986
Epoch 7/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.23it/s, loss=1, acc=46.37%]
Validation Accuracy: 46.37% | Loss: 3.2003
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 8/100 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=3.25, acc=46.39%]
Epoch 8/100 [Train]: 100%|███████████████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=3.24, acc=46.46%]
Epoch 8/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.38it/s, loss=0.956, acc=49.24%]
Validation Accuracy: 49.24% | Loss: 3.0490
Epoch 8/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████| 196/196 [00:36<00:00,  5.33it/s, loss=0.957, acc=49.26%]
Validation Accuracy: 49.26% | Loss: 3.0505
Epoch 8/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.955, acc=49.35%]
Validation Accuracy: 49.35% | Loss: 3.0465
Epoch 8/100 [Val]: 100%|██████████████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.22it/s, loss=0.955, acc=49.35%]
Validation Accuracy: 49.35% | Loss: 3.0442
Successfully uploaded checkpoint_epoch8.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch8.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch8.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 9/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=3.14, acc=48.49%]

Epoch 9/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.78it/s, loss=3.14, acc=48.54%]
Epoch 9/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.78it/s, loss=3.14, acc=48.42%]
Epoch 9/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.956, acc=49.67%]
Validation Accuracy: 49.67% | Loss: 3.0495
Epoch 9/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.24it/s, loss=0.953, acc=49.93%]
Validation Accuracy: 49.93% | Loss: 3.0385
Epoch 9/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.18it/s, loss=0.955, acc=49.81%]
Validation Accuracy: 49.81% | Loss: 3.0452
Epoch 9/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.14it/s, loss=0.954, acc=49.86%]
Validation Accuracy: 49.86% | Loss: 3.0414
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 10/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 10/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 10/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 10/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=3.06, acc=50.19%]
Epoch 10/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=3.06, acc=50.11%]
Epoch 10/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=3.07, acc=50.16%]

Epoch 10/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.24it/s, loss=0.955, acc=49.07%]
Validation Accuracy: 49.07% | Loss: 3.0454
Epoch 10/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.22it/s, loss=0.951, acc=49.37%]
Validation Accuracy: 49.37% | Loss: 3.0328
Epoch 10/40 [Val]:  99%|████████████████████████████████████████████████████████████████▎| 194/196 [00:37<00:00,  5.17it/s, loss=3.04, acc=49.27%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch10.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch10.pth
Epoch 10/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.16it/s, loss=0.953, acc=49.27%]
Validation Accuracy: 49.27% | Loss: 3.0404
Epoch 10/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.14it/s, loss=0.951, acc=49.40%]
Validation Accuracy: 49.40% | Loss: 3.0323
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch10.pth
Successfully uploaded checkpoint_epoch10.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch10.pth
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 11/40 [Train]: 100%|████████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=3, acc=51.55%]
Epoch 11/40 [Train]: 100%|████████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=3, acc=51.53%]
Epoch 11/40 [Train]: 100%|████████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=3, acc=51.47%]
Epoch 11/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.906, acc=52.90%]
Validation Accuracy: 52.90% | Loss: 2.8897
Epoch 11/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.907, acc=52.81%]
Validation Accuracy: 52.81% | Loss: 2.8923
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 11/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.21it/s, loss=0.906, acc=52.92%]
Validation Accuracy: 52.92% | Loss: 2.8879
Epoch 11/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.908, acc=52.72%]
Validation Accuracy: 52.72% | Loss: 2.8968
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 12/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.69it/s, loss=2.95, acc=52.64%]
Epoch 12/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.95, acc=52.69%]
Epoch 12/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.78it/s, loss=2.95, acc=52.68%]
Epoch 12/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.78it/s, loss=2.95, acc=52.64%]
Epoch 12/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.24it/s, loss=0.906, acc=53.27%]
Validation Accuracy: 53.27% | Loss: 2.8902
Epoch 12/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.24it/s, loss=0.907, acc=53.21%]
Validation Accuracy: 53.21% | Loss: 2.8926
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch12.pth
Epoch 12/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.15it/s, loss=0.907, acc=53.21%]
Validation Accuracy: 53.21% | Loss: 2.8933
Epoch 12/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.14it/s, loss=0.908, acc=53.10%]
Validation Accuracy: 53.10% | Loss: 2.8947
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch12.pth
Epoch 13/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.9, acc=53.85%]
Epoch 13/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.9, acc=53.66%]
Epoch 13/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.78it/s, loss=2.91, acc=53.60%]
Epoch 13/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.886, acc=54.31%]
Validation Accuracy: 54.31% | Loss: 2.8265
Epoch 13/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.24it/s, loss=0.887, acc=54.25%]
Validation Accuracy: 54.25% | Loss: 2.8292
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 13/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.16it/s, loss=0.886, acc=54.41%]
Validation Accuracy: 54.41% | Loss: 2.8246
Epoch 13/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.14it/s, loss=0.885, acc=54.35%]
Validation Accuracy: 54.35% | Loss: 2.8218
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
 Epoch 14/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.86, acc=54.68%]
Epoch 14/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.87, acc=54.53%]
Epoch 14/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.88, acc=54.90%]
Validation Accuracy: 54.90% | Loss: 2.8072
Epoch 14/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.88, acc=54.88%]
Validation Accuracy: 54.88% | Loss: 2.8076
Epoch 14/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.20it/s, loss=0.88, acc=54.91%]
Validation Accuracy: 54.91% | Loss: 2.8074
Epoch 14/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.88, acc=54.95%]
Validation Accuracy: 54.95% | Loss: 2.8056
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch14.pth
Successfully uploaded checkpoint_epoch14.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch14.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 15/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.83, acc=55.54%]
Epoch 15/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.83, acc=55.44%]
Epoch 15/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.83, acc=55.52%]
Epoch 15/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.853, acc=56.76%]
Validation Accuracy: 56.76% | Loss: 2.7187
Epoch 15/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.853, acc=56.73%]
Validation Accuracy: 56.73% | Loss: 2.7198
Epoch 15/40 [Val]:  98%|████████████████████████████████████████████████████████████████ | 193/196 [00:37<00:00,  5.14it/s, loss=2.72, acc=56.66%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 15/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.853, acc=56.75%]
Validation Accuracy: 56.75% | Loss: 2.7200
Epoch 15/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.14it/s, loss=0.854, acc=56.66%]
Validation Accuracy: 56.66% | Loss: 2.7222
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 16/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.8, acc=56.12%]
Epoch 16/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.79, acc=56.09%]
Epoch 16/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.844, acc=57.12%]
Validation Accuracy: 57.12% | Loss: 2.6928
Epoch 16/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.844, acc=57.21%]
Validation Accuracy: 57.21% | Loss: 2.6909
Epoch 16/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.844, acc=57.23%]
Validation Accuracy: 57.23% | Loss: 2.6908
Epoch 16/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.15it/s, loss=0.843, acc=57.24%]
Validation Accuracy: 57.24% | Loss: 2.6882
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch16.pth
Successfully uploaded checkpoint_epoch16.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch16.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 17/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.76, acc=56.80%]
Epoch 17/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.77, acc=56.80%]
Epoch 17/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.77, acc=56.87%]
Epoch 17/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.823, acc=59.05%]
Validation Accuracy: 59.05% | Loss: 2.6231
Epoch 17/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.23it/s, loss=0.823, acc=59.04%]
Validation Accuracy: 59.04% | Loss: 2.6242
Epoch 17/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.821, acc=59.20%]
Validation Accuracy: 59.20% | Loss: 2.6190
Epoch 17/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.14it/s, loss=0.821, acc=59.15%]
Validation Accuracy: 59.15% | Loss: 2.6184
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 18/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.73, acc=57.59%]
Epoch 18/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.74, acc=57.39%]
Epoch 18/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.74, acc=57.49%]
Epoch 18/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.73, acc=57.52%]
Epoch 18/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.84, acc=57.54%]
Validation Accuracy: 57.54% | Loss: 2.6790
Epoch 18/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.842, acc=57.43%]
Validation Accuracy: 57.43% | Loss: 2.6841
Epoch 18/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.842, acc=57.40%]
Validation Accuracy: 57.40% | Loss: 2.6863
Epoch 18/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.15it/s, loss=0.841, acc=57.57%]
Validation Accuracy: 57.57% | Loss: 2.6811
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch18.pth
Successfully uploaded checkpoint_epoch18.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch18.pth
Epoch 19/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.71, acc=58.10%]
Epoch 19/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.71, acc=57.99%]
Epoch 19/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.822, acc=59.10%]
Validation Accuracy: 59.10% | Loss: 2.6203
Epoch 19/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.821, acc=59.20%]
Validation Accuracy: 59.20% | Loss: 2.6185
Epoch 19/40 [Val]:  98%|████████████████████████████████████████████████████████████████ | 193/196 [00:37<00:00,  5.16it/s, loss=2.62, acc=59.11%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 19/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.20it/s, loss=0.82, acc=59.30%]
Validation Accuracy: 59.30% | Loss: 2.6158
Epoch 19/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.16it/s, loss=0.822, acc=59.16%]
Validation Accuracy: 59.16% | Loss: 2.6216
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 20/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 20/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 20/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 20/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.68, acc=58.75%]
Epoch 20/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.68, acc=58.69%]
Epoch 20/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.68, acc=58.71%]

Epoch 20/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.81, acc=59.82%]
Validation Accuracy: 59.82% | Loss: 2.5833
Epoch 20/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.23it/s, loss=0.809, acc=59.89%]
Validation Accuracy: 59.89% | Loss: 2.5797
Epoch 20/40 [Val]:  98%|███████████████████████████████████████████████████████████████▋ | 192/196 [00:37<00:00,  5.14it/s, loss=2.58, acc=59.93%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch20.pth
Epoch 20/40 [Val]:  99%|████████████████████████████████████████████████████████████████▋| 195/196 [00:37<00:00,  5.24it/s, loss=2.58, acc=59.91%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch20.pth
Epoch 20/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.809, acc=59.92%]
Validation Accuracy: 59.92% | Loss: 2.5787
Error in multipart upload: An error occurred (MalformedXML) when calling the CompleteMultipartUpload operation: The XML you provided was not well-formed or did not validate against our published schema
Successfully uploaded checkpoint_epoch20.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch20.pth
Epoch 20/40 [Val]:  99%|████████████████████████████████████████████████████████████████▎| 194/196 [00:37<00:00,  5.14it/s, loss=2.58, acc=59.95%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch20.pth
Epoch 20/40 [Val]:  99%|████████████████████████████████████████████████████████████████▋| 195/196 [00:38<00:00,  5.15it/s, loss=2.58, acc=59.94%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 20/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.13it/s, loss=0.809, acc=59.95%]
Validation Accuracy: 59.95% | Loss: 2.5782
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch20.pth
Successfully uploaded checkpoint_epoch20.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch20.pth
Successfully uploaded checkpoint_epoch20.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch20.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 21/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded checkpoint_epoch20.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch20.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 21/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 21/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 21/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.65, acc=59.29%]
Epoch 21/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.65, acc=59.28%]
Epoch 21/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.78it/s, loss=2.66, acc=59.18%]

Epoch 21/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.24it/s, loss=0.81, acc=59.59%]
Validation Accuracy: 59.59% | Loss: 2.5843
Epoch 21/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.23it/s, loss=0.81, acc=59.65%]
Epoch 21/40 [Val]:  98%|███████████████████████████████████████████████████████████████▋ | 192/196 [00:37<00:00,  5.15it/s, loss=2.59, acc=59.55%]Validation Accuracy: 59.65% | Loss: 2.5844
Epoch 21/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.812, acc=59.47%]
Validation Accuracy: 59.47% | Loss: 2.5907
Epoch 21/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.15it/s, loss=0.811, acc=59.59%]
Validation Accuracy: 59.59% | Loss: 2.5866
Epoch 22/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.63, acc=59.80%]
Epoch 22/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.63, acc=59.90%]
Epoch 22/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.63, acc=59.74%]

Epoch 22/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.789, acc=61.42%]
Validation Accuracy: 61.42% | Loss: 2.5154
Epoch 22/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.787, acc=61.55%]
Validation Accuracy: 61.55% | Loss: 2.5086
Epoch 22/40 [Val]:  98%|████████████████████████████████████████████████████████████████ | 193/196 [00:37<00:00,  5.16it/s, loss=2.51, acc=61.46%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch22.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch22.pth
Epoch 22/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.20it/s, loss=0.788, acc=61.49%]
Validation Accuracy: 61.49% | Loss: 2.5125
Epoch 22/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.16it/s, loss=0.788, acc=61.46%]
Validation Accuracy: 61.46% | Loss: 2.5138
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch22.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch22.pth
Successfully uploaded checkpoint_epoch22.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch22.pth
Successfully uploaded checkpoint_epoch22.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch22.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded checkpoint_epoch22.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch22.pth
Successfully uploaded checkpoint_epoch22.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch22.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 23/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 23/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 23/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 23/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.61, acc=60.35%]

Epoch 23/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.61, acc=60.30%]
Epoch 23/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.6, acc=60.45%]
Epoch 23/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.793, acc=61.13%]
Validation Accuracy: 61.13% | Loss: 2.5300
Epoch 23/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.24it/s, loss=0.793, acc=61.15%]
Validation Accuracy: 61.15% | Loss: 2.5284
Epoch 23/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.20it/s, loss=0.793, acc=61.09%]
Validation Accuracy: 61.09% | Loss: 2.5283
Epoch 23/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.16it/s, loss=0.793, acc=61.10%]
Validation Accuracy: 61.10% | Loss: 2.5300
Epoch 24/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.59, acc=60.87%]
Epoch 24/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.58, acc=60.92%]

Epoch 24/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.58, acc=60.87%]
Epoch 24/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.795, acc=61.11%]
Validation Accuracy: 61.11% | Loss: 2.5356
Epoch 24/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.794, acc=61.18%]
Validation Accuracy: 61.18% | Loss: 2.5333
Epoch 24/40 [Val]:  98%|███████████████████████████████████████████████████████████████▋ | 192/196 [00:37<00:00,  5.14it/s, loss=2.54, acc=60.90%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch24.pth
Epoch 24/40 [Val]:  99%|████████████████████████████████████████████████████████████████▎| 194/196 [00:37<00:00,  5.20it/s, loss=2.54, acc=60.99%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch24.pth
Epoch 24/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.796, acc=60.98%]
Validation Accuracy: 60.98% | Loss: 2.5393
Epoch 24/40 [Val]:  99%|████████████████████████████████████████████████████████████████▋| 195/196 [00:38<00:00,  5.17it/s, loss=2.54, acc=60.91%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch24.pth
Epoch 24/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.14it/s, loss=0.797, acc=60.92%]
Validation Accuracy: 60.92% | Loss: 2.5425
Error in multipart upload: An error occurred (MalformedXML) when calling the CompleteMultipartUpload operation: The XML you provided was not well-formed or did not validate against our published schema
Successfully uploaded checkpoint_epoch24.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch24.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 25/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch24.pth
Successfully uploaded checkpoint_epoch24.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch24.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 25/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded checkpoint_epoch24.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch24.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 25/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded checkpoint_epoch24.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch24.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 25/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.56, acc=61.57%]

Epoch 25/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.78it/s, loss=2.56, acc=61.60%]
Epoch 25/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.55, acc=61.67%]
Epoch 25/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.775, acc=62.50%]
Validation Accuracy: 62.50% | Loss: 2.4705
Epoch 25/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.777, acc=62.34%]
Validation Accuracy: 62.34% | Loss: 2.4767
Epoch 25/40 [Val]:  98%|████████████████████████████████████████████████████████████████ | 193/196 [00:37<00:00,  5.17it/s, loss=2.48, acc=62.32%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 25/40 [Val]:  98%|████████████████████████████████████████████████████████████████ | 193/196 [00:37<00:00,  5.15it/s, loss=2.48, acc=62.32%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 25/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.776, acc=62.34%]
Validation Accuracy: 62.34% | Loss: 2.4753
Epoch 25/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.16it/s, loss=0.777, acc=62.36%]
Validation Accuracy: 62.36% | Loss: 2.4766
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 26/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 26/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 26/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 26/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.53, acc=62.26%]
Epoch 26/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.53, acc=62.20%]
Epoch 26/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.53, acc=62.24%]
Epoch 26/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.53, acc=62.15%]
Epoch 26/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.767, acc=63.08%]
Validation Accuracy: 63.08% | Loss: 2.4454
Epoch 26/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.767, acc=63.09%]
Validation Accuracy: 63.09% | Loss: 2.4452
Epoch 26/40 [Val]:  98%|████████████████████████████████████████████████████████████████ | 193/196 [00:37<00:00,  5.20it/s, loss=2.45, acc=63.06%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch26.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch26.pth
Epoch 26/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.767, acc=63.06%]
Validation Accuracy: 63.06% | Loss: 2.4461
Epoch 26/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.767, acc=63.09%]
Validation Accuracy: 63.09% | Loss: 2.4443
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch26.pth
Successfully uploaded checkpoint_epoch26.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch26.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 27/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.5, acc=63.09%]
Epoch 27/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.5, acc=62.88%]
Epoch 27/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.757, acc=64.04%]
Validation Accuracy: 64.04% | Loss: 2.4132
Epoch 27/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.757, acc=63.93%]
Validation Accuracy: 63.93% | Loss: 2.4152
Epoch 27/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.20it/s, loss=0.759, acc=63.84%]
Validation Accuracy: 63.84% | Loss: 2.4190
Epoch 27/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.16it/s, loss=0.757, acc=64.00%]
Validation Accuracy: 64.00% | Loss: 2.4133
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 28/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.47, acc=63.54%]
Epoch 28/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.78it/s, loss=2.47, acc=63.50%]
Epoch 28/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.48, acc=63.48%]
Epoch 28/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.752, acc=64.20%]
Validation Accuracy: 64.20% | Loss: 2.3983
Epoch 28/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.24it/s, loss=0.752, acc=64.22%]
Validation Accuracy: 64.22% | Loss: 2.3971
Epoch 28/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.16it/s, loss=0.752, acc=64.19%]
Validation Accuracy: 64.19% | Loss: 2.3968
Epoch 28/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.14it/s, loss=0.751, acc=64.31%]
Validation Accuracy: 64.31% | Loss: 2.3955
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch28.pth
Successfully uploaded checkpoint_epoch28.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch28.pth
Epoch 29/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 29/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.44, acc=64.32%]
Epoch 29/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.45, acc=64.14%]
Epoch 29/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.736, acc=65.56%]
Validation Accuracy: 65.56% | Loss: 2.3479
Epoch 29/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.736, acc=65.54%]
Validation Accuracy: 65.54% | Loss: 2.3479
Epoch 29/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.737, acc=65.50%]
Validation Accuracy: 65.50% | Loss: 2.3489
Epoch 29/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.15it/s, loss=0.736, acc=65.53%]
Validation Accuracy: 65.53% | Loss: 2.3472
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 30/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.41, acc=65.03%]
Epoch 30/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.41, acc=64.98%]
Epoch 30/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.41, acc=65.15%]
Epoch 30/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.725, acc=66.53%]
Validation Accuracy: 66.53% | Loss: 2.3109
Epoch 30/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.23it/s, loss=0.725, acc=66.52%]
Validation Accuracy: 66.52% | Loss: 2.3113
Epoch 30/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.16it/s, loss=0.724, acc=66.55%]
Validation Accuracy: 66.55% | Loss: 2.3081
Epoch 30/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.13it/s, loss=0.724, acc=66.57%]
Validation Accuracy: 66.57% | Loss: 2.3101
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch30.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch30.pth
Successfully uploaded checkpoint_epoch30.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch30.pth
Successfully uploaded checkpoint_epoch30.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch30.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded checkpoint_epoch30.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch30.pth
Successfully uploaded checkpoint_epoch30.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch30.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 31/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 31/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 31/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 31/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:40<00:00,  1.79it/s, loss=2.38, acc=65.82%]
Epoch 31/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.38, acc=65.93%]
Epoch 31/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:40<00:00,  1.79it/s, loss=2.37, acc=65.89%]

Epoch 31/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.29it/s, loss=0.721, acc=66.60%]
Validation Accuracy: 66.60% | Loss: 2.2982
Epoch 31/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.721, acc=66.58%]
Validation Accuracy: 66.58% | Loss: 2.2989
Epoch 31/40 [Val]:  99%|█████████████████████████████████████████████████████████████████▎| 194/196 [00:37<00:00,  5.23it/s, loss=2.3, acc=66.60%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 31/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.22it/s, loss=0.72, acc=66.60%]
Validation Accuracy: 66.60% | Loss: 2.2949
Epoch 31/40 [Val]: 100%|█████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.20it/s, loss=0.72, acc=66.65%]
Validation Accuracy: 66.65% | Loss: 2.2956
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 32/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 32/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 32/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 32/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.34, acc=66.90%]
Epoch 32/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.34, acc=66.86%]
Epoch 32/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.34, acc=66.87%]

Epoch 32/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.706, acc=67.59%]
Validation Accuracy: 67.59% | Loss: 2.2508
Epoch 32/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.705, acc=67.63%]
Validation Accuracy: 67.63% | Loss: 2.2478
Epoch 32/40 [Val]:  98%|████████████████████████████████████████████████████████████████ | 193/196 [00:37<00:00,  5.17it/s, loss=2.25, acc=67.71%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch32.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch32.pth
Epoch 32/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.16it/s, loss=0.705, acc=67.72%]
Validation Accuracy: 67.72% | Loss: 2.2473
Epoch 32/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.14it/s, loss=0.705, acc=67.72%]
Validation Accuracy: 67.72% | Loss: 2.2466
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch32.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch32.pth
Successfully uploaded checkpoint_epoch32.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch32.pth
Successfully uploaded checkpoint_epoch32.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch32.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded checkpoint_epoch32.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch32.pth
Successfully uploaded checkpoint_epoch32.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch32.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 33/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 33/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 33/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 33/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.3, acc=67.73%]

Epoch 33/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.29, acc=67.87%]

Epoch 33/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.694, acc=68.48%]
Validation Accuracy: 68.48% | Loss: 2.2116
Epoch 33/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.694, acc=68.57%]
Validation Accuracy: 68.57% | Loss: 2.2115
Epoch 33/40 [Val]:  98%|████████████████████████████████████████████████████████████████ | 193/196 [00:37<00:00,  5.18it/s, loss=2.21, acc=68.51%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 33/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.18it/s, loss=0.693, acc=68.57%]
Validation Accuracy: 68.57% | Loss: 2.2107
Epoch 33/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.694, acc=68.54%]
Validation Accuracy: 68.54% | Loss: 2.2123
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 34/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 34/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 34/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 34/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:40<00:00,  1.78it/s, loss=2.26, acc=68.88%]
Epoch 34/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:40<00:00,  1.79it/s, loss=2.26, acc=68.89%]
Epoch 34/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:40<00:00,  1.79it/s, loss=2.26, acc=68.81%]

Epoch 34/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.684, acc=69.56%]
Validation Accuracy: 69.56% | Loss: 2.1822
Epoch 34/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.684, acc=69.53%]
Validation Accuracy: 69.53% | Loss: 2.1821
Epoch 34/40 [Val]:  99%|████████████████████████████████████████████████████████████████▋| 195/196 [00:37<00:00,  5.24it/s, loss=2.18, acc=69.54%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch34.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch34.pth
Epoch 34/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.22it/s, loss=0.684, acc=69.54%]
Validation Accuracy: 69.54% | Loss: 2.1813
Error in multipart upload: An error occurred (MalformedXML) when calling the CompleteMultipartUpload operation: The XML you provided was not well-formed or did not validate against our published schema
Successfully uploaded checkpoint_epoch34.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch34.pth
Epoch 34/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.21it/s, loss=0.684, acc=69.59%]
Validation Accuracy: 69.59% | Loss: 2.1815
Error in multipart upload: An error occurred (EntityTooSmall) when calling the CompleteMultipartUpload operation: Your proposed upload is smaller than the minimum allowed size
Successfully uploaded checkpoint_epoch34.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch34.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch34.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch34.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded checkpoint_epoch34.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch34.pth
Successfully uploaded checkpoint_epoch34.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch34.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 35/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 35/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 35/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 35/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.22, acc=69.77%]
Epoch 35/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:40<00:00,  1.78it/s, loss=2.22, acc=70.04%]
Epoch 35/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.22, acc=69.88%]

Epoch 35/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.676, acc=70.28%]
Validation Accuracy: 70.28% | Loss: 2.1560
Epoch 35/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.676, acc=70.26%]
Validation Accuracy: 70.26% | Loss: 2.1564
Epoch 35/40 [Val]:  99%|████████████████████████████████████████████████████████████████▎| 194/196 [00:37<00:00,  5.21it/s, loss=2.16, acc=70.25%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 35/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.676, acc=70.26%]
Validation Accuracy: 70.26% | Loss: 2.1563
Epoch 35/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.17it/s, loss=0.676, acc=70.33%]
Validation Accuracy: 70.33% | Loss: 2.1550
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 36/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 36/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 36/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 36/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.18, acc=70.88%]
Epoch 36/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.18, acc=70.91%]

Epoch 36/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.18, acc=70.97%]
Epoch 36/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.669, acc=70.80%]
Validation Accuracy: 70.80% | Loss: 2.1338
Epoch 36/40 [Val]:  99%|████████████████████████████████████████████████████████████████▋| 195/196 [00:37<00:00,  5.26it/s, loss=2.13, acc=70.83%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch36.pth
Epoch 36/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.24it/s, loss=0.669, acc=70.83%]
Validation Accuracy: 70.83% | Loss: 2.1318
Epoch 36/40 [Val]:  98%|███████████████████████████████████████████████████████████████▋ | 192/196 [00:37<00:00,  5.17it/s, loss=2.13, acc=70.84%]Error in multipart upload: An error occurred (MalformedXML) when calling the CompleteMultipartUpload operation: The XML you provided was not well-formed or did not validate against our published schema
Successfully uploaded checkpoint_epoch36.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch36.pth
Epoch 36/40 [Val]:  98%|████████████████████████████████████████████████████████████████ | 193/196 [00:37<00:00,  5.17it/s, loss=2.13, acc=70.84%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch36.pth
Epoch 36/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.669, acc=70.87%]
Validation Accuracy: 70.87% | Loss: 2.1323
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 36/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.15it/s, loss=0.669, acc=70.87%]
Validation Accuracy: 70.87% | Loss: 2.1318
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch36.pth
Successfully uploaded checkpoint_epoch36.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch36.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch36.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded checkpoint_epoch36.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch36.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 37/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 37/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded checkpoint_epoch36.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch36.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 37/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 37/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.14, acc=71.78%]
Epoch 37/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:46<00:00,  1.77it/s, loss=2.14, acc=71.82%]
Epoch 37/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:46<00:00,  1.77it/s, loss=2.14, acc=71.81%]

Epoch 37/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.663, acc=71.34%]
Validation Accuracy: 71.34% | Loss: 2.1130
Epoch 37/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.663, acc=71.33%]
Validation Accuracy: 71.33% | Loss: 2.1143
Epoch 37/40 [Val]:  97%|███████████████████████████████████████████████████████████████▎ | 191/196 [00:37<00:00,  5.11it/s, loss=2.12, acc=71.24%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 37/40 [Val]:  99%|████████████████████████████████████████████████████████████████▎| 194/196 [00:37<00:00,  5.22it/s, loss=2.12, acc=71.26%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 37/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.663, acc=71.26%]
Validation Accuracy: 71.26% | Loss: 2.1153
Epoch 37/40 [Val]:  99%|████████████████████████████████████████████████████████████████▎| 194/196 [00:38<00:00,  5.12it/s, loss=2.11, acc=71.29%]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Epoch 37/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.12it/s, loss=0.663, acc=71.28%]
Validation Accuracy: 71.28% | Loss: 2.1145
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 38/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 38/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 38/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 38/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:44<00:00,  1.78it/s, loss=2.11, acc=72.55%]
Epoch 38/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.11, acc=72.59%]

Epoch 38/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:45<00:00,  1.77it/s, loss=2.11, acc=72.55%]
Epoch 38/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.655, acc=71.81%]
Validation Accuracy: 71.81% | Loss: 2.0900
Epoch 38/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.656, acc=71.78%]
Validation Accuracy: 71.78% | Loss: 2.0906
Epoch 38/40 [Val]:  99%|████████████████████████████████████████████████████████████████▎| 194/196 [00:37<00:00,  5.21it/s, loss=2.09, acc=71.79%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch38.pth
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch38.pth
Epoch 38/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.656, acc=71.79%]
Validation Accuracy: 71.79% | Loss: 2.0904
Epoch 38/40 [Val]:  99%|████████████████████████████████████████████████████████████████▎| 194/196 [00:38<00:00,  5.11it/s, loss=2.09, acc=71.79%]Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch38.pth
Epoch 38/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.12it/s, loss=0.656, acc=71.80%]
Validation Accuracy: 71.80% | Loss: 2.0905
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch38.pth
Successfully uploaded checkpoint_epoch38.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch38.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded checkpoint_epoch38.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch38.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded checkpoint_epoch38.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch38.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded checkpoint_epoch38.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch38.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 39/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 39/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 39/40 [Train]:   0%|                                                                                               | 0/1251 [00:00<?, ?it/s]Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062453.log
Successfully uploaded training_20241230_062453.log to s3://imagenet-data-170/training_logs/training_20241230_062453.log
Uploading to S3: s3://imagenet-data-170/training_logs/training_20241230_062157.log
Successfully uploaded training_20241230_062157.log to s3://imagenet-data-170/training_logs/training_20241230_062157.log
Epoch 39/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.09, acc=73.19%]
Epoch 39/40 [Train]: 100%|██████████████████████████████████████████████████████████████| 1251/1251 [11:41<00:00,  1.78it/s, loss=2.1, acc=73.00%]
Epoch 39/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.27it/s, loss=0.653, acc=72.02%]
Validation Accuracy: 72.02% | Loss: 2.0831
Epoch 39/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.25it/s, loss=0.654, acc=72.05%]
Validation Accuracy: 72.05% | Loss: 2.0845
Epoch 39/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.18it/s, loss=0.654, acc=72.01%]
Validation Accuracy: 72.01% | Loss: 2.0843
Epoch 39/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.15it/s, loss=0.654, acc=72.03%]
Validation Accuracy: 72.03% | Loss: 2.0844
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded best_model.pth to s3://imagenet-data-170/models/best_model.pth
Epoch 40/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:42<00:00,  1.78it/s, loss=2.08, acc=73.40%]

Epoch 40/40 [Train]: 100%|█████████████████████████████████████████████████████████████| 1251/1251 [11:43<00:00,  1.78it/s, loss=2.08, acc=73.41%]
Epoch 40/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.28it/s, loss=0.653, acc=72.08%]
Validation Accuracy: 72.08% | Loss: 2.0834
Epoch 40/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.26it/s, loss=0.653, acc=72.08%]
Validation Accuracy: 72.08% | Loss: 2.0829
Epoch 40/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:37<00:00,  5.19it/s, loss=0.653, acc=72.01%]
Validation Accuracy: 72.01% | Loss: 2.0827
Epoch 40/40 [Val]: 100%|████████████████████████████████████████████████████████████████| 196/196 [00:38<00:00,  5.16it/s, loss=0.653, acc=72.06%]
Validation Accuracy: 72.06% | Loss: 2.0824
Uploading to S3: s3://imagenet-data-170/models/checkpoints/checkpoint_epoch40.pth
Successfully uploaded checkpoint_epoch40.pth to s3://imagenet-data-170/models/checkpoints/checkpoint_epoch40.pth
Uploading to S3: s3://imagenet-data-170/models/best_model.pth
Successfully uploaded final_model.pth to s3://imagenet-data-170/models/final_model.pth
