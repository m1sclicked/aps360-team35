import sys 
import platform 
print('System Diagnostics:') 
print(f'Python: {sys.version}') 
print(f'Platform: {platform.platform()}') 
try: 
    import torch 
    print('\nPyTorch Information:') 
    print(f'Version: {torch.__version__}') 
    print(f'CUDA Available: {torch.cuda.is_available()}') 
    if torch.cuda.is_available(): 
        print(f'CUDA Version: {torch.version.cuda}') 
        print(f'GPU Device: {torch.cuda.get_device_name(0)}') 
except ImportError: 
    print('PyTorch not installed') 
try: 
    import tensorflow as tf 
    print('\nTensorFlow Information:') 
    print(f'Version: {tf.__version__}') 
    try: 
        from tensorflow_directml_plugin import load_op_library 
        load_op_library() 
        print('DirectML Plugin Loaded Successfully') 
    except ImportError as e: 
        print(f'DirectML Plugin Import Error: {e}') 
        print('Detailed DirectML Plugin Installation Check:') 
        import pkg_resources 
        try: 
            print(pkg_resources.get_distribution('tensorflow-directml-plugin')) 
        except pkg_resources.DistributionNotFound: 
            print('tensorflow-directml-plugin not found in installed packages') 
    gpus = tf.config.list_physical_devices('GPU') 
    print(f'GPU Devices: {gpus}') 
except ImportError as e: 
    print(f'TensorFlow Import Error: {e}') 
