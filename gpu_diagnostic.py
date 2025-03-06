import sys 
import platform 
import numpy as np 
print('System Diagnostics:') 
print(f'Python: {sys.version}') 
print(f'Platform: {platform.platform()}') 
print(f'NumPy: {np.__version__}') 
try: 
    import protobuf 
    print(f'Protobuf: {protobuf.__version__}') 
except (ImportError, AttributeError): 
    import google.protobuf 
    print(f'Protobuf: {google.protobuf.__version__}') 
try: 
    import flatbuffers 
    print(f'Flatbuffers: {flatbuffers.__version__ if hasattr(flatbuffers, "__version__") else "Unknown"}') 
except ImportError as e: 
    print(f'Flatbuffers Import Error: {e}') 
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
    gpus = tf.config.list_physical_devices('GPU') 
    print(f'GPU Devices: {gpus}') 
except ImportError as e: 
    print(f'TensorFlow Import Error: {e}') 
except Exception as e: 
    print(f'TensorFlow Error: {e}') 
try: 
    import mediapipe as mp 
    print('\nMediaPipe Information:') 
    print(f'Version: {mp.__version__}') 
except ImportError as e: 
    print(f'MediaPipe Import Error: {e}') 
