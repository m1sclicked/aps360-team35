import sys 
import platform 
print('System Diagnostics:') 
print(f'Python: {sys.version}') 
print(f'Platform: {platform.platform()}') 
try: 
    import numpy as np 
    print(f'\nNumPy Version: {np.__version__}') 
except ImportError as e: 
    print(f'NumPy Import Error: {e}') 
try: 
    import torch 
    print('\nPyTorch Information:') 
    print(f'Version: {torch.__version__}') 
    print(f'CUDA Available: {torch.cuda.is_available()}') 
    if torch.cuda.is_available(): 
        print(f'CUDA Version: {torch.version.cuda}') 
        print(f'GPU Device: {torch.cuda.get_device_name(0)}') 
except ImportError as e: 
    print(f'PyTorch Import Error: {e}') 
try: 
    import tensorflow as tf 
    print('\nTensorFlow Information:') 
    print(f'Version: {tf.__version__}') 
    print('Checking DirectML availability...') 
    try: 
        from tensorflow.python.framework.errors_impl import NotFoundError 
        try: 
            print(f'Available GPU devices: {tf.config.list_physical_devices("GPU")}') 
            print(f'Available DirectML devices: {tf.config.list_physical_devices("DML")}') 
            with tf.device('DML:0'): 
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]]) 
                b = tf.constant([[1.0, 1.0], [1.0, 1.0]]) 
                c = tf.matmul(a, b) 
            print('DirectML test: Successful matrix multiplication using DirectML') 
        except NotFoundError as e: 
            print(f'DirectML device error: {e}') 
        except Exception as e: 
            print(f'DirectML test error: {e}') 
    except ImportError as e: 
        print(f'DirectML plugin import error: {e}') 
except ImportError as e: 
    print(f'TensorFlow Import Error: {e}') 
