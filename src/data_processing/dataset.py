import os
import json
import numpy as np
import torch
import pickle
import subprocess
import zipfile
from tqdm import tqdm
import cv2 as cv
from torch.utils.data import Dataset
#import mediapipe as mp

class GestureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class GestureSequenceDataset(Dataset):
    """Dataset class that handles sequences of gesture features"""
    def __init__(self, features, labels, seq_length=150, normalize=True):
        """
        Args:
            features: List of variable-length feature sequences, each representing a video
            labels: List of class labels
            seq_length: Maximum sequence length (will pad or truncate)
            normalize: Whether to normalize features
        """
        self.seq_length = seq_length
        self.labels = torch.LongTensor(labels)
        
        # Process sequences to have uniform length through padding/truncation
        self.processed_features = []
        
        for sequence in features:
            processed_seq = self.process_sequence(sequence, self.seq_length, normalize)
            self.processed_features.append(processed_seq)
        
        # Convert to tensor
        self.processed_features = torch.stack(self.processed_features)
        
        # Create attention masks (1 for real data, 0 for padding)
        self.masks = torch.zeros((len(features), seq_length), dtype=torch.bool)
        for i, seq in enumerate(features):
            seq_len = min(len(seq), seq_length)
            self.masks[i, :seq_len] = 1
    
    def process_sequence(self, sequence, max_length, normalize):
    # Convert to tensor if not already
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.FloatTensor(sequence)
        
        # Normalize if requested
        if normalize:
            # Calculate mean and std across sequence, handling zeros
            mask = (sequence != 0).float()
            if mask.sum() > 0:  # Only normalize if there are non-zero values
                seq_mean = (sequence.sum(dim=0) / (mask.sum(dim=0) + 1e-8))
                seq_std = torch.sqrt(((sequence - seq_mean)**2 * mask).sum(dim=0) / 
                                        (mask.sum(dim=0) + 1e-8) + 1e-8)
                sequence = (sequence - seq_mean) / seq_std
                sequence = sequence * mask  # Keep zeros as zeros
        
        # Handle sequence length
        seq_len = sequence.size(0)
        
        if seq_len >= max_length:
            # Truncate if sequence is too long
            return sequence[:max_length]
        else:
            # Pad with zeros if sequence is too short
            padding = torch.zeros((max_length - seq_len, sequence.size(1)), dtype=torch.float32)
            return torch.cat([sequence, padding], dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.processed_features[idx], self.masks[idx], self.labels[idx]

# def initialize_mediapipe():
#     """
#     Initialize MediaPipe hand tracking as an alternative to OpenPose
    
#     Returns:
#         mediapipe_hands: MediaPipe hands solution
#         None, None: Placeholders for compatibility with OpenPose interface
#     """
#     try:
#         import mediapipe as mp
#         print("Initializing MediaPipe hand tracking...")
#         mp_hands = mp.solutions.hands
#         hands = mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         return hands, None, None
#     except ImportError:
#         print("MediaPipe not found. Installing...")
#         subprocess.run(["pip", "install", "mediapipe"], check=True)
#         import mediapipe as mp
#         mp_hands = mp.solutions.hands
#         hands = mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         return hands, None, None

def download_wlasl_data(data_path="data/wlasl_data", download=False):
    """
    Locally load or download WLASL dataset from the updated Kaggle source
    
    Args:
        data_path (str): Path to save the dataset
        download (bool): Whether to attempt downloading the dataset
    """
    import os
    import json
    import subprocess
    import zipfile
    import shutil
    
    os.makedirs(data_path, exist_ok=True)

    json_path = os.path.join(data_path, "WLASL_v0.3.json")

    # Check if dataset already exists
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            wlasl_data = json.load(f)
        return wlasl_data

    # If download is requested
    if download:
        try:
            # Download dataset using Kaggle CLI with the updated source
            download_cmd = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "sttaseen/wlasl2000-resized",
                "-p",
                data_path,
            ]
            print("Downloading WLASL dataset from sttaseen/wlasl2000-resized...")
            subprocess.run(download_cmd, check=True)

            # Unzip the downloaded file to a temporary directory
            zip_path = os.path.join(data_path, "wlasl2000-resized.zip")
            print(f"Extracting dataset from {zip_path}...")
            
            # Create a temporary directory for extraction
            temp_dir = os.path.join(data_path, "temp_extract")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract to temporary directory
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Now find the "wlasl complete" directory in the temp directory
            wlasl_complete_dir = None
            for root, dirs, files in os.walk(temp_dir):
                if os.path.basename(root) == "wlasl complete":
                    wlasl_complete_dir = root
                    break
            
            if wlasl_complete_dir:
                print(f"Found 'wlasl complete' directory at {wlasl_complete_dir}")
                
                # Move all contents directly to data_path - no subdirectories
                for item in os.listdir(wlasl_complete_dir):
                    src = os.path.join(wlasl_complete_dir, item)
                    dst = os.path.join(data_path, item)
                    
                    # If it's a file, simply move it
                    if os.path.isfile(src):
                        shutil.move(src, dst)
                    # If it's a directory, copy all its contents directly to data_path
                    elif os.path.isdir(src):
                        for subitem in os.listdir(src):
                            subsrc = os.path.join(src, subitem)
                            subdst = os.path.join(data_path, subitem)
                            shutil.move(subsrc, subdst)
            else:
                print("Warning: Could not find 'wlasl complete' directory. Attempting direct extraction.")
                # If we didn't find the expected directory structure, try to guess what's important
                # Look for json files and video directories anywhere in the temp_dir
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.json'):
                            src = os.path.join(root, file)
                            dst = os.path.join(data_path, file)
                            shutil.move(src, dst)
                    
                    for dir_name in dirs:
                        if dir_name.lower() in ['videos', 'video']:
                            src_dir = os.path.join(root, dir_name)
                            # Move all contents of the videos directory directly to data_path/videos
                            videos_dir = os.path.join(data_path, 'videos')
                            os.makedirs(videos_dir, exist_ok=True)
                            
                            for video_file in os.listdir(src_dir):
                                src_file = os.path.join(src_dir, video_file)
                                dst_file = os.path.join(videos_dir, video_file)
                                if os.path.isfile(src_file):
                                    shutil.move(src_file, dst_file)
            
            # Clean up
            shutil.rmtree(temp_dir)
            os.remove(zip_path)
            print("Extraction complete, temporary files removed.")
            
            # Check if the WLASL_v0.3.json file exists now
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    wlasl_data = json.load(f)
                print(f"Successfully loaded WLASL data with {len(wlasl_data)} classes.")
                return wlasl_data
            else:
                print(f"Could not find WLASL_v0.3.json after extraction. Dataset structure may have changed.")
                return None

        except subprocess.CalledProcessError:
            print("Failed to download dataset. Check your Kaggle API credentials.")
            print("Make sure you have installed the Kaggle CLI and configured your API key:")
            print("1. Install Kaggle: pip install kaggle")
            print("2. Get your API key from kaggle.com/account")
            print("3. Place the kaggle.json file in ~/.kaggle/")
            return None
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            import traceback
            traceback.print_exc()
            return None

    # If download not requested and dataset not found
    print("Dataset not found. Use download=True to fetch the dataset.")
    return None



def check_cuda_availability():
    """
    Check if CUDA is available for OpenCV DNN
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    # Check if CUDA is available in OpenCV
    if cv.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA is available with {cv.cuda.getCudaEnabledDeviceCount()} device(s)")
        return True
    else:
        print("CUDA is not available in OpenCV")
        return False


# Initialize OpenPose
def initialize_openpose(model_path="models/openpose", use_gpu=True):
    """
    Initialize OpenPose model with GPU support if available
    
    Args:
        model_path (str): Path to OpenPose models
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        net: The OpenPose body model
        hand_net: The OpenPose hand model
        input_width, input_height: Input dimensions for the model
    """
    # Check if model folder exists
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        print(f"Created model directory at {model_path}. Please place OpenPose models here.")
        
    # Paths to the pre-trained OpenPose models
    proto_file = os.path.join(model_path, "pose_deploy.prototxt")
    weights_file = os.path.join(model_path, "pose_iter_584000.caffemodel")
    hand_proto = os.path.join(model_path, "hand/pose_deploy.prototxt")
    hand_weights = os.path.join(model_path, "hand/pose_iter_102000.caffemodel")
    
    # Check if model files exist
    if not all(os.path.exists(f) for f in [proto_file, weights_file, hand_proto, hand_weights]):
        raise FileNotFoundError(
            "OpenPose model files not found. Please download from https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models"
        )
    
    # Load OpenPose models
    print("Loading OpenPose models...")
    
    # Body model
    net = cv.dnn.readNetFromCaffe(proto_file, weights_file)
    
    # Hand model
    hand_net = cv.dnn.readNetFromCaffe(hand_proto, hand_weights)
    
    # Try to use GPU if requested
    if use_gpu and check_cuda_availability():
        print("Using CUDA acceleration for OpenPose")
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        
        hand_net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        hand_net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    else:
        print("Using CPU for OpenPose")
    
    input_width, input_height = 368, 368  # Default OpenPose input size
    
    return net, hand_net, input_width, input_height

def extract_keypoints_openpose(frame, net, hand_net, input_width=368, input_height=368):
    """
    Extract hand keypoints using OpenPose
    
    Args:
        frame: Input video frame
        net: OpenPose body network
        hand_net: OpenPose hand network
        input_width, input_height: Input dimensions for the model
        
    Returns:
        numpy.ndarray: Flattened array of hand keypoints
    """
    frame_height, frame_width = frame.shape[0], frame.shape[1]
    
    # Prepare the input blob for the body model
    input_blob = cv.dnn.blobFromImage(
        frame, 1.0 / 255, (input_width, input_height), (0, 0, 0), swapRB=False, crop=False
    )
    
    # Set the input and run the body network
    net.setInput(input_blob)
    body_output = net.forward()
    
    # Initialize hand keypoints with zeros
    left_hand = np.zeros(21 * 3)  # 21 keypoints, 3 dimensions (x, y, confidence)
    right_hand = np.zeros(21 * 3)
    
    # Process body keypoints to find wrists
    left_wrist = None
    right_wrist = None
    
    # Process heatmap output to find keypoints
    num_keypoints = body_output.shape[1]
    
    for i in range(num_keypoints):
        # Get the heatmap for this keypoint
        heatmap = body_output[0, i, :, :]
        
        # Find the location with maximum confidence
        _, confidence, _, point = cv.minMaxLoc(heatmap)
        
        # Only process if confidence is above threshold
        if confidence > 0.1:
            # Convert point to original image coordinates
            x = int(point[0] * frame_width / heatmap.shape[1])
            y = int(point[1] * frame_height / heatmap.shape[0])
            
            if i == 4:  # Right wrist
                right_wrist = (x, y)
            elif i == 7:  # Left wrist
                left_wrist = (x, y)
    
    # The rest of the function remains the same
    # Process hands if wrists are detected
    if left_wrist is not None:
        # Crop and process left hand
        hand_roi = get_hand_roi(frame, left_wrist)
        if hand_roi is not None:
            left_hand_keypoints = process_hand(hand_roi, hand_net, input_width, input_height)
            if left_hand_keypoints is not None:
                left_hand = left_hand_keypoints.flatten()
    
    if right_wrist is not None:
        # Crop and process right hand
        hand_roi = get_hand_roi(frame, right_wrist)
        if hand_roi is not None:
            right_hand_keypoints = process_hand(hand_roi, hand_net, input_width, input_height)
            if right_hand_keypoints is not None:
                right_hand = right_hand_keypoints.flatten()
    
    # Concatenate left and right hand keypoints
    features = np.concatenate([left_hand, right_hand])
    
    # Check if we actually detected any hands
    non_zero_values = np.count_nonzero(features)
    if non_zero_values > 0:
        return features
    return None

def get_hand_roi(frame, wrist_pos, roi_size=250):
    """
    Extract a region of interest around a hand based on wrist position
    
    Args:
        frame: Input frame
        wrist_pos: (x, y) coordinates of wrist
        roi_size: Size of ROI in pixels
        
    Returns:
        numpy.ndarray: Cropped hand image
    """
    x, y = wrist_pos
    half_size = roi_size // 2
    
    # Calculate ROI boundaries
    left = max(0, x - half_size)
    top = max(0, y - half_size)
    right = min(frame.shape[1], x + half_size)
    bottom = min(frame.shape[0], y + half_size)
    
    # Check if ROI is valid
    if right - left <= 0 or bottom - top <= 0:
        return None
    
    # Extract ROI
    hand_roi = frame[top:bottom, left:right]
    
    return hand_roi

def process_hand(hand_roi, hand_net, input_width, input_height):
    """
    Process a hand region using OpenPose hand network
    
    Args:
        hand_roi: Cropped hand image
        hand_net: OpenPose hand network
        input_width, input_height: Input dimensions for the model
        
    Returns:
        numpy.ndarray: Hand keypoints
    """
    # Prepare input blob for hand network
    hand_blob = cv.dnn.blobFromImage(
        hand_roi, 1.0 / 255, (input_width, input_height), (0, 0, 0), swapRB=False, crop=False
    )
    
    # Set the input and run the hand network
    hand_net.setInput(hand_blob)
    hand_output = hand_net.forward()
    
    # Process hand keypoints
    hand_keypoints = np.zeros((21, 3))
    hand_roi_height, hand_roi_width = hand_roi.shape[0], hand_roi.shape[1]
    
    keypoints_detected = 0
    
    # Check the shape of hand_output to better handle it
    # For heatmap output format: (1, num_keypoints, height, width)
    if len(hand_output.shape) == 4:
        num_keypoints = hand_output.shape[1]
        
        for i in range(min(21, num_keypoints)):  # OpenPose hand model has 21 keypoints
            # Get the heatmap for this keypoint
            heatmap = hand_output[0, i, :, :]
            
            # Find the location with maximum confidence
            _, confidence, _, point = cv.minMaxLoc(heatmap)
            
            if confidence > 0.1:  # confidence threshold
                # Convert point to original hand ROI coordinates
                x = point[0] * hand_roi_width / heatmap.shape[1]
                y = point[1] * hand_roi_height / heatmap.shape[0]
                
                hand_keypoints[i] = [x, y, confidence]
                keypoints_detected += 1
    
    # Only return keypoints if we actually detected something
    if keypoints_detected > 0:
        return hand_keypoints
    return None

def process_video_openpose(video_path, net, hand_net, input_width=368, input_height=368, sample_rate=2):
    """
    Process a single video to extract hand landmarks using OpenPose
    Returns a sequence of features instead of averaging them
    
    Args:
        video_path: Path to the video file
        net: OpenPose body network
        hand_net: OpenPose hand network
        input_width, input_height: Input dimensions for the model
        sample_rate: Process every nth frame for efficiency
        
    Returns:
        List of feature arrays, one per processed frame
    """
    try:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None

        frames_features = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % sample_rate != 0:
                continue

            # Process the frame with OpenPose
            try:
                frame_features = extract_keypoints_openpose(frame, net, hand_net, input_width, input_height)
                
                # Only add frame features if they contain hand detections
                if frame_features is not None:
                    frames_features.append(frame_features)
            except Exception as e:
                import traceback
                print(f"Error processing frame {frame_count}: {str(e)}")
                print(traceback.format_exc())
                continue

        cap.release()

        # If no frames with hand landmarks were processed
        if len(frames_features) == 0:
            return None

        # Return sequence of frame features
        return frames_features

    except Exception as e:
        import traceback
        print(f"Error processing video {video_path}: {str(e)}")
        print(traceback.format_exc())
        return None

def prepare_dataset_openpose(wlasl_data, data_path, num_classes=100):
    """
    Prepare dataset by extracting sequence features from videos using OpenPose
    
    Args:
        wlasl_data: WLASL dataset JSON
        data_path: Path to dataset
        num_classes: Number of classes to process
        
    Returns:
        features, labels: Sequences of extracted features and corresponding labels
    """
    # Initialize OpenPose
    try:
        net, hand_net, input_width, input_height = initialize_openpose(use_gpu=True)
    except FileNotFoundError as e:
        print(f"OpenPose initialization failed: {e}")
        print("Please download OpenPose models and place them in the models/openpose directory.")
        return [], []
    
    features = []  # Now this will be a list of sequences
    labels = []
    processed_classes = []
    
    # Ensure the videos directory exists
    videos_dir = os.path.join(data_path, "videos")
    if not os.path.exists(videos_dir):
        print(f"Videos directory not found: {videos_dir}")
        return features, labels

    # Get list of available video files
    available_videos = {
        f.split(".")[0] for f in os.listdir(videos_dir) if f.endswith(".mp4")
    }
    print(f"Total available videos: {len(available_videos)}")

    # Process classes
    for class_idx, sign_class in enumerate(tqdm(wlasl_data[:num_classes])):
        class_name = sign_class["gloss"]
        instances = sign_class["instances"]
        
        # Filter for actually available videos for this class
        available_class_videos = [
            entry for entry in instances if str(entry["video_id"]) in available_videos
        ]
        
        # Process all available videos for this class
        class_features = []
        
        for entry in available_class_videos:
            video_id = str(entry["video_id"])
            video_path = os.path.join(videos_dir, f"{video_id}.mp4")

            # Extract sequence of features with OpenPose
            video_features = process_video_openpose(video_path, net, hand_net, input_width, input_height)

            # Only add if processing was successful
            if video_features is not None and len(video_features) > 0:
                features.append(video_features)  # Add the whole sequence
                labels.append(class_idx)
                class_features.append(video_features)
        
        # Only add class if it has samples
        if len(class_features) > 0:
            processed_classes.append(class_name)
            print(f"Class {class_name}: {len(class_features)} videos processed")
    
    print(f"Final processed classes: {len(processed_classes)}")
    print(f"Class names: {processed_classes}")
    print(f"Total processed videos: {len(features)}")
    
    return features, labels

def preprocess_wlasl_dataset_openpose(data_path="data/wlasl_data", output_path="data/preprocessed", num_classes=100):
    """
    Preprocess the WLASL dataset using OpenPose by extracting sequence features from all videos
    and saving them to disk for fast loading later.
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    print(f"Created output directory: {output_path}")
    
    # Load WLASL data
    wlasl_data_path = os.path.join(data_path, "WLASL_v0.3.json")
    print(f"Looking for WLASL data at: {wlasl_data_path}")
    
    # Check if JSON file exists
    if not os.path.exists(wlasl_data_path):
        print(f"WLASL JSON file not found at {wlasl_data_path}")
        return
    
    # Load JSON
    print("Loading WLASL JSON data...")
    with open(wlasl_data_path, "r") as f:
        wlasl_data = json.load(f)
    print(f"Loaded {len(wlasl_data)} classes from JSON")
    
    # Initialize OpenPose
    print("Initializing OpenPose...")
    try:
        net, hand_net, input_width, input_height = initialize_openpose(use_gpu=True)
        print("OpenPose initialized successfully")
    except FileNotFoundError as e:
        print(f"OpenPose initialization failed: {e}")
        print("Please download OpenPose models and place them in the models/openpose directory.")
        return
    
    # Dictionary to store sequence features indexed by video_id
    videos_features = {}
    
    # Dictionary to map classes to video_ids
    class_to_videos = {}
    class_indices = {}
    
    # Ensure the videos directory exists
    videos_dir = os.path.join(data_path, "videos")
    if not os.path.exists(videos_dir):
        print(f"Videos directory not found: {videos_dir}")
        return
    
    # Get list of available video files
    available_videos = {
        f.split(".")[0] for f in os.listdir(videos_dir) if f.endswith(".mp4")
    }
    print(f"Found {len(available_videos)} available videos")
    
    # Process videos for specified number of classes
    print("Starting video processing...")
    for class_idx, sign_class in enumerate(tqdm(wlasl_data[:num_classes])):
        class_name = sign_class["gloss"]
        instances = sign_class["instances"]
        
        print(f"\nProcessing class {class_idx + 1}/{num_classes}: {class_name}")
        
        # Initialize class entry
        class_to_videos[class_name] = []
        class_indices[class_name] = class_idx
        
        # Filter for actually available videos for this class
        available_class_videos = [
            entry for entry in instances if str(entry["video_id"]) in available_videos
        ]
        
        print(f"Found {len(available_class_videos)} videos for class {class_name}")
        
        # Process all available videos for this class
        successful_videos = 0
        for entry in available_class_videos:
            video_id = str(entry["video_id"])
            video_path = os.path.join(videos_dir, f"{video_id}.mp4")
            
            print(f"  Processing video {video_id}...", end=" ")
            
            # Extract sequence of features with OpenPose
            video_features = process_video_openpose(video_path, net, hand_net, input_width, input_height)
            
            # Store features if processing was successful
            if video_features is not None and len(video_features) > 0:
                # Store as a list of lists (sequence of features per frame)
                videos_features[video_id] = [frame.tolist() for frame in video_features]
                class_to_videos[class_name].append(video_id)
                successful_videos += 1
                print(f"Success - {len(video_features)} frames processed")
            else:
                print("Failed")
        
        print(f"Successfully processed {successful_videos}/{len(available_class_videos)} videos for class {class_name}")
    
    # Save the preprocessed data
    print("\nSaving preprocessed data...")
    
    # Save features - now this contains sequences
    features_path = os.path.join(output_path, "video_sequence_features.json")
    with open(features_path, "w") as f:
        json.dump(videos_features, f)
    print(f"Saved sequence features to {features_path}")
    
    # Save class mapping
    class_map_path = os.path.join(output_path, "class_to_videos.json")
    with open(class_map_path, "w") as f:
        json.dump(class_to_videos, f)
    print(f"Saved class mapping to {class_map_path}")
    
    # Save class indices
    class_indices_path = os.path.join(output_path, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(class_indices, f)
    print(f"Saved class indices to {class_indices_path}")
    
    # Count and report
    processed_videos = sum(len(videos) for videos in class_to_videos.values())
    print(f"\nPreprocessing complete!")
    print(f"Processed {processed_videos} videos across {len(class_to_videos)} classes.")
    print(f"Data saved to {output_path}")

def integrated_prepare_sequence_dataset(data_path="data/wlasl_data", 
                                       num_classes=100, 
                                       use_preprocessing=True,
                                       min_videos_per_class=1):
    """
    Integrated function to either load preprocessed sequence data or process videos on-the-fly
    
    Args:
        data_path (str): Path to the original dataset
        num_classes (int): Number of classes to include
        use_preprocessing (bool): Whether to use preprocessing
        min_videos_per_class (int): Minimum videos per class
        
    Returns:
        features (list): List of sequence features
        labels (list): Corresponding labels
    """
    preprocessed_path = os.path.join(data_path, "preprocessed")
    
    # First check if preprocessed sequence data exists
    features_path = os.path.join(preprocessed_path, "video_sequence_features.json")
    class_map_path = os.path.join(preprocessed_path, "class_to_videos.json")
    
    if use_preprocessing and os.path.exists(features_path) and os.path.exists(class_map_path):
        print("Loading preprocessed sequence data...")
        return load_preprocessed_sequence_dataset(preprocessed_path, min_videos_per_class)
    
    # If not found, check for old format data
    old_features_path = os.path.join(preprocessed_path, "video_features.json")
    if use_preprocessing and os.path.exists(old_features_path) and os.path.exists(class_map_path):
        print("Found old format preprocessed data (non-sequence).")
        print("Consider regenerating sequence data for better results.")
        return load_preprocessed_sequence_dataset(preprocessed_path, min_videos_per_class)
    
    # If not, ask if user wants to preprocess
    if use_preprocessing:
        print("Preprocessed sequence data not found.")
        choice = input("Would you like to preprocess the dataset now? (y/n): ")
        
        if choice.lower() == 'y':
            print("Starting preprocessing...")
            preprocess_wlasl_dataset_openpose(data_path, preprocessed_path, num_classes)
            return load_preprocessed_sequence_dataset(preprocessed_path, min_videos_per_class)
    
    # If no preprocessing, load raw data and process on-the-fly
    print("Processing videos on-the-fly (this will be slow)...")
    wlasl_data_path = os.path.join(data_path, "WLASL_v0.3.json")
    
    # Check if JSON file exists
    if not os.path.exists(wlasl_data_path):
        print(f"WLASL JSON file not found at {wlasl_data_path}")
        return [], []
        
    with open(wlasl_data_path, "r") as f:
        wlasl_data = json.load(f)
        
    return prepare_dataset_openpose(wlasl_data, data_path, num_classes)

def load_preprocessed_sequence_dataset(preprocessed_path="data/preprocessed", min_videos_per_class=1):
    """
    Load the preprocessed sequence dataset
    
    Args:
        preprocessed_path (str): Path to the preprocessed data
        min_videos_per_class (int): Minimum number of videos required for a class to be included
        
    Returns:
        features (list): List of sequence features (each sequence is a list of frame features)
        labels (list): Corresponding labels
    """
    # Load sequence features
    features_path = os.path.join(preprocessed_path, "video_sequence_features.json")
    if not os.path.exists(features_path):
        print(f"Sequence features file not found at {features_path}")
        # Try the old format
        features_path = os.path.join(preprocessed_path, "video_features.json")
        if not os.path.exists(features_path):
            print(f"No features file found at {preprocessed_path}")
            return [], []
        print("Using old format features file (non-sequence data)")
    
    with open(features_path, "r") as f:
        videos_features = json.load(f)
    
    # Load class mapping
    class_map_path = os.path.join(preprocessed_path, "class_to_videos.json")
    with open(class_map_path, "r") as f:
        class_to_videos = json.load(f)
    
    # Load class indices
    class_indices_path = os.path.join(preprocessed_path, "class_indices.json")
    with open(class_indices_path, "r") as f:
        original_class_indices = json.load(f)
    
    # Filter classes with enough videos
    valid_classes = {}
    for class_name, video_ids in class_to_videos.items():
        if len(video_ids) >= min_videos_per_class:
            valid_classes[class_name] = video_ids
    
    # Create new class indices
    new_class_indices = {}
    for i, class_name in enumerate(valid_classes.keys()):
        new_class_indices[class_name] = i
    
    # Build feature and label arrays
    features = []
    labels = []
    
    for class_name, video_ids in valid_classes.items():
        new_index = new_class_indices[class_name]
        for video_id in video_ids:
            if video_id in videos_features:
                # Check if it's sequence data or old format
                video_data = videos_features[video_id]
                if isinstance(video_data[0], list):
                    # It's already a sequence
                    features.append(video_data)
                else:
                    # It's the old format (single feature vector)
                    # Create a dummy "sequence" with one frame
                    features.append([video_data])
                labels.append(new_index)
    
    print(f"Loaded {len(features)} samples from {len(valid_classes)} classes")
    
    # Count samples per class
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("Samples per class:")
    for idx, count in sorted(class_counts.items()):
        class_name = list(new_class_indices.keys())[list(new_class_indices.values()).index(idx)]
        print(f"  Class {idx} ({class_name}): {count} samples")
    
    # Also print some sequence length stats
    seq_lengths = [len(seq) for seq in features]
    print(f"Sequence length stats: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.1f}")
    
    return features, labels

def prepare_sequential_dataset(features, labels, seq_length=150, batch_size=32, test_size=0.2, 
                            use_augmentation=True, augmentation_params=None, augmentation_factor=3):
    """
    Prepare sequential dataset with train/val/test split and optional augmentation
    
    Args:
        features: List of sequences (each a list of frame features)
        labels: List of class labels
        seq_length: Maximum sequence length
        batch_size: Batch size for data loaders
        test_size: Proportion of data to use for testing
        use_augmentation: Whether to apply data augmentation to training set
        augmentation_params: Dictionary of parameters for the augmenter
        augmentation_factor: Target multiplier for dataset size (including original data)
        
    Returns:
        train_loader, val_loader, test_loader, input_dim
    """
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from src.data_processing.dataset import GestureSequenceDataset
    from src.data_processing.sequence_augmenter import GestureSequenceAugmenter, AugmentedGestureSequenceDataset
    import copy
    import numpy as np
    import torch
    
    # Determine input dimension from first frame of first sequence
    if len(features) > 0 and len(features[0]) > 0:
        input_dim = len(features[0][0])
    else:
        raise ValueError("Empty features list or sequences")
    
    # Print original dataset size
    print(f"Original dataset size: {len(features)} sequences")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=test_size, random_state=42
    )
    
    print(f"Train set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    print(f"Test set: {len(X_test)} sequences")
    
    # Create base datasets for validation and test
    val_dataset = GestureSequenceDataset(X_val, y_val, seq_length=seq_length)
    test_dataset = GestureSequenceDataset(X_test, y_test, seq_length=seq_length)
    
    # For training data, we have options based on augmentation
    if use_augmentation and augmentation_factor > 1:
        # Instead of using the complex augmentation with masks, 
        # let's rely on the AugmentedGestureSequenceDataset which is already set up properly
        print(f"Using on-the-fly augmentation with AugmentedGestureSequenceDataset")
        print(f"Original train set: {len(X_train)} sequences")
        
        # Create the base training dataset
        base_train_dataset = GestureSequenceDataset(X_train, y_train, seq_length=seq_length)
        
        # Create augmenter with custom parameters or defaults
        if augmentation_params is None:
            augmenter = GestureSequenceAugmenter(
                jitter_range=0.02,
                scale_range=(0.8, 1.2),
                rotation_range=(-20, 20),
                translation_range=0.1,
                time_stretch_range=(0.75, 1.25),
                dropout_prob=0.05,
                swap_hands_prob=0.3,
                mirror_prob=0.5
            )
        else:
            augmenter = GestureSequenceAugmenter(**augmentation_params)
        
        # Instead of trying to create explicit augmented samples, 
        # we'll use the on-the-fly approach but increase the dataset size
        train_dataset = []
        
        # Add the original dataset
        train_dataset.append(base_train_dataset)
        
        # Add multiple augmented versions to reach the desired size
        for i in range(augmentation_factor - 1):
            # Each added dataset provides on-the-fly augmentation
            aug_dataset = AugmentedGestureSequenceDataset(
                base_train_dataset,
                augmenter,
                augment_prob=1.0  # Always augment
            )
            train_dataset.append(aug_dataset)
        
        # Combine datasets using ConcatDataset
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_dataset)
        
        print(f"Created combined dataset with {len(train_dataset)} samples")
        print(f"Augmentation factor: {len(train_dataset) / len(X_train):.2f}x")
    
    elif use_augmentation:
        # Standard on-the-fly augmentation without explicit size increase
        # Create the base training dataset
        base_train_dataset = GestureSequenceDataset(X_train, y_train, seq_length=seq_length)
        
        # Create augmenter with custom parameters or defaults
        if augmentation_params is None:
            augmenter = GestureSequenceAugmenter(
                jitter_range=0.02,
                scale_range=(0.8, 1.2),
                rotation_range=(-20, 20),
                translation_range=0.1,
                time_stretch_range=(0.75, 1.25),
                dropout_prob=0.05,
                swap_hands_prob=0.3,
                mirror_prob=0.5
            )
        else:
            augmenter = GestureSequenceAugmenter(**augmentation_params)
        
        # Use on-the-fly augmentation
        train_dataset = AugmentedGestureSequenceDataset(
            base_train_dataset,
            augmenter,
            augment_prob=0.8
        )
        print(f"Using standard on-the-fly augmentation with augment_prob=0.8")
    
    else:
        # No augmentation, just use the original training data
        train_dataset = GestureSequenceDataset(X_train, y_train, seq_length=seq_length)
        print("No augmentation applied")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues during debugging
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, input_dim