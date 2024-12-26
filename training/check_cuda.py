import os
import sys
import torch
try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    print(f"CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
except ImportError as e:
    print(f"Error importing CuPy: {e}")

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available (PyTorch): {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")

print("\nEnvironment variables:")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}") 