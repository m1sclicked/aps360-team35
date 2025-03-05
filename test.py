import sys
import platform

def check_system_info():
    print("System Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print("\nGPU Diagnostics:")

    try:
        import torch
        print("\nPyTorch Information:")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed or import failed")

    try:
        import tensorflow as tf
        print("\nTensorFlow Information:")
        print(f"TensorFlow Version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU Devices: {gpus}")
    except ImportError:
        print("TensorFlow not installed or import failed")

    try:
        import nvidia_smi
        print("\nNVIDIA SMI Information:")
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        print(nvidia_smi.nvmlDeviceGetName(handle).decode('utf-8'))
    except ImportError:
        print("nvidia-smi not available. Install with 'pip install nvidia-smi'")

if __name__ == "__main__":
    check_system_info()