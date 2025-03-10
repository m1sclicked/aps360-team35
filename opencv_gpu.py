import cv2
import numpy as np
import time

def print_opencv_info():
    # Check OpenCV version and build info
    print(f"OpenCV Version: {cv2.__version__}")
    
    build_info = cv2.getBuildInformation()
    cuda_info = build_info[build_info.find('CUDA'):build_info.find('CUDA')+500]
    print("\nCUDA Build Information:")
    print(cuda_info[:cuda_info.find('\n\n')])
    
    # Check CUDA devices
    devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"\nCUDA Devices available: {devices}")
    
    if devices > 0:
        for i in range(devices):
            print(f"\nCUDA Device {i} Information:")
            print(cv2.cuda.printCudaDeviceInfo(i))

def test_performance():
    print("\n--- Performance Tests ---")
    
    # Create larger test images of different sizes
    sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
    
    for size in sizes:
        print(f"\nTesting with image size: {size[0]}x{size[1]}")
        img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        
        # Test multiple operations
        
        # 1. Gaussian Blur
        print("\n1. Gaussian Blur Test:")
        # CPU timing
        start_time = time.time()
        cpu_blur = cv2.GaussianBlur(img, (21, 21), 0)
        cpu_time = time.time() - start_time
        print(f"   CPU time: {cpu_time:.4f} seconds")
        
        # GPU timing - including upload/download time
        start_time = time.time()
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (21, 21), 0).apply(gpu_img)
        result_blur = gpu_blur.download()
        gpu_time_total = time.time() - start_time
        print(f"   GPU time (total): {gpu_time_total:.4f} seconds")
        
        # GPU timing - excluding upload/download time
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        start_time = time.time()
        gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (21, 21), 0).apply(gpu_img)
        gpu_time_compute = time.time() - start_time
        result_blur = gpu_blur.download()
        print(f"   GPU time (compute only): {gpu_time_compute:.4f} seconds")
        print(f"   Speedup (compute only): {cpu_time/gpu_time_compute:.2f}x")
        
        # 2. Resize Test
        print("\n2. Resize Test:")
        target_size = (int(size[0]/2), int(size[1]/2))
        
        # CPU timing
        start_time = time.time()
        cpu_resize = cv2.resize(img, target_size)
        cpu_time = time.time() - start_time
        print(f"   CPU time: {cpu_time:.4f} seconds")
        
        # GPU timing - compute only
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        start_time = time.time()
        gpu_resize = cv2.cuda.resize(gpu_img, target_size)
        gpu_time_compute = time.time() - start_time
        result_resize = gpu_resize.download()
        print(f"   GPU time (compute only): {gpu_time_compute:.4f} seconds")
        print(f"   Speedup (compute only): {cpu_time/gpu_time_compute:.2f}x")
        
        # 3. Canny Edge Detection
        print("\n3. Canny Edge Detection:")
        # CPU timing
        start_time = time.time()
        cpu_canny = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
        cpu_time = time.time() - start_time
        print(f"   CPU time: {cpu_time:.4f} seconds")
        
        # GPU timing - compute only
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        start_time = time.time()
        detector = cv2.cuda.createCannyEdgeDetector(100, 200)
        gpu_canny = detector.detect(gpu_img)
        gpu_time_compute = time.time() - start_time
        result_canny = gpu_canny.download()
        print(f"   GPU time (compute only): {gpu_time_compute:.4f} seconds")
        print(f"   Speedup (compute only): {cpu_time/gpu_time_compute:.2f}x")

if __name__ == "__main__":
    print_opencv_info()
    test_performance()