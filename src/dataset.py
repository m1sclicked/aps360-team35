import os
import json
import numpy as np
import torch
import pickle
import subprocess
import zipfile
from tqdm import tqdm
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import cv2 as cv
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def download_wlasl_data(data_path="data/wlasl_data", download=False):
    """
    Locally load or download WLASL dataset

    Args:
        data_path (str): Path to save the dataset
        download (bool): Whether to attempt downloading the dataset
    """
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
            # Download dataset using Kaggle CLI
            download_cmd = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "risangbaskoro/wlasl-processed",
                "-p",
                data_path,
            ]
            subprocess.run(download_cmd, check=True)

            # Unzip the downloaded file
            zip_path = os.path.join(data_path, "wlasl-processed.zip")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_path)

            # Remove the zip file
            os.remove(zip_path)

            # Load the JSON
            with open(json_path, "r") as f:
                wlasl_data = json.load(f)

            return wlasl_data

        except subprocess.CalledProcessError:
            print("Failed to download dataset. Check your Kaggle API credentials.")
            return None
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None

    # If download not requested and dataset not found
    print("Dataset not found. Use download=True to fetch the dataset.")
    return None

def extract_keypoints(results):
    """
    Extract hand keypoints from MediaPipe results
    """
    # Initialize empty lists for left and right hand landmarks
    lh = []
    rh = []

    # Iterate through detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # Check if it's the left or right hand
            if hand_handedness.classification[0].label == "Left":
                lh = np.array(
                    [[res.x, res.y, res.z] for res in hand_landmarks.landmark]
                ).flatten()
            elif hand_handedness.classification[0].label == "Right":
                rh = np.array(
                    [[res.x, res.y, res.z] for res in hand_landmarks.landmark]
                ).flatten()

    # Pad with zeros if no landmarks are detected for a hand
    lh = lh if len(lh) > 0 else np.zeros(21 * 3)
    rh = rh if len(rh) > 0 else np.zeros(21 * 3)

    return np.concatenate([lh, rh])


def process_video(video_path, hands):
    """
    Process a single video to extract hand landmarks
    Returns None if processing fails or no hands are detected
    """
    try:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None

        frames_features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process the frame
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                frame_features = extract_keypoints(results)
                frames_features.append(frame_features)

        cap.release()

        # If no frames with hand landmarks were processed
        if len(frames_features) == 0:
            return None

        # Return mean of all frame features
        return np.mean(frames_features, axis=0)

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None


def prepare_dataset(wlasl_data, data_path, num_classes=100):
    """
    Prepare dataset by extracting features from videos
    """
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )

    features = []
    labels = []
    processed_classes = []
    
    # Ensure the videos directory exists
    videos_dir = os.path.join(data_path, "videos")
    if not os.path.exists(videos_dir):
        print(f"Videos directory not found: {videos_dir}")
        return np.array(features), np.array(labels)

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

            # Extract features
            video_features = process_video(video_path, hands)

            # Only add if processing was successful
            if video_features is not None:
                class_features.append(video_features)
        
        # Only add class if it has samples
        if len(class_features) > 0:
            features.extend(class_features)
            labels.extend([class_idx] * len(class_features))
            processed_classes.append(class_name)
            print(f"Class {class_name}: {len(class_features)} videos processed")
    
    print(f"Final processed classes: {len(processed_classes)}")
    print(f"Class names: {processed_classes}")
    print(f"Total processed videos: {len(features)}")
    
    return np.array(features), np.array(labels)


def preprocess_wlasl_dataset(data_path="data/wlasl_data", output_path="data/preprocessed", num_classes=100):
    """
    Preprocess the WLASL dataset by extracting features from all videos once
    and saving them to disk for fast loading later.
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load WLASL data
    with open(os.path.join(data_path, "WLASL_v0.3.json"), "r") as f:
        wlasl_data = json.load(f)
    
    if wlasl_data is None:
        print("Failed to load WLASL data")
        return
    
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )
    
    # Dictionary to store features indexed by video_id
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
    print(f"Total available videos: {len(available_videos)}")
    
    # Process videos for specified number of classes
    print("Processing videos...")
    for class_idx, sign_class in enumerate(tqdm(wlasl_data[:num_classes])):
        class_name = sign_class["gloss"]
        instances = sign_class["instances"]
        
        # Initialize class entry
        class_to_videos[class_name] = []
        class_indices[class_name] = class_idx
        
        # Filter for actually available videos for this class
        available_class_videos = [
            entry for entry in instances if str(entry["video_id"]) in available_videos
        ]
        
        print(f"Processing class {class_name} ({len(available_class_videos)} videos)")
        
        # Process all available videos for this class
        for entry in available_class_videos:
            video_id = str(entry["video_id"])
            video_path = os.path.join(videos_dir, f"{video_id}.mp4")
            
            # Extract features
            video_features = process_video(video_path, hands)
            
            # Store features if processing was successful
            if video_features is not None:
                videos_features[video_id] = video_features.tolist()  # Convert to list for JSON serialization
                class_to_videos[class_name].append(video_id)
    
    # Save the preprocessed data
    print("Saving preprocessed data...")
    
    # Save features
    features_path = os.path.join(output_path, "video_features.json")
    with open(features_path, "w") as f:
        json.dump(videos_features, f)
    
    # Save class mapping
    class_map_path = os.path.join(output_path, "class_to_videos.json")
    with open(class_map_path, "w") as f:
        json.dump(class_to_videos, f)
    
    # Save class indices
    class_indices_path = os.path.join(output_path, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(class_indices, f)
    
    # Count and report
    processed_videos = sum(len(videos) for videos in class_to_videos.values())
    print(f"Preprocessing complete! Processed {processed_videos} videos across {len(class_to_videos)} classes.")
    print(f"Data saved to {output_path}")


def load_preprocessed_dataset(preprocessed_path="data/preprocessed", min_videos_per_class=1):
    """
    Load the preprocessed dataset
    
    Args:
        preprocessed_path (str): Path to the preprocessed data
        min_videos_per_class (int): Minimum number of videos required for a class to be included
        
    Returns:
        features (numpy.ndarray): Extracted features
        labels (numpy.ndarray): Corresponding labels
    """
    # Load features
    features_path = os.path.join(preprocessed_path, "video_features.json")
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
                features.append(videos_features[video_id])
                labels.append(new_index)
    
    # Convert to numpy arrays
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    print(f"Loaded {len(features)} samples from {len(valid_classes)} classes")
    
    # Count samples per class
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("Samples per class:")
    for idx, count in sorted(class_counts.items()):
        class_name = list(new_class_indices.keys())[list(new_class_indices.values()).index(idx)]
        print(f"  Class {idx} ({class_name}): {count} samples")
    
    return features_array, labels_array


def integrated_prepare_dataset(data_path="data/wlasl_data", 
                               num_classes=100, 
                               use_preprocessing=True,
                               min_videos_per_class=1):
    """
    Integrated function to either load preprocessed data or process videos on-the-fly
    
    Args:
        data_path (str): Path to the original dataset
        num_classes (int): Number of classes to include
        use_preprocessing (bool): Whether to use preprocessing
        min_videos_per_class (int): Minimum videos per class
        
    Returns:
        features (numpy.ndarray): Extracted features
        labels (numpy.ndarray): Corresponding labels
    """
    preprocessed_path = os.path.join(data_path, "preprocessed")
    
    # First check if preprocessed data exists
    features_path = os.path.join(preprocessed_path, "video_features.json")
    class_map_path = os.path.join(preprocessed_path, "class_to_videos.json")
    
    if use_preprocessing and os.path.exists(features_path) and os.path.exists(class_map_path):
        print("Loading preprocessed data...")
        return load_preprocessed_dataset(preprocessed_path, min_videos_per_class)
    
    # If not, ask if user wants to preprocess
    if use_preprocessing:
        print("Preprocessed data not found.")
        choice = input("Would you like to preprocess the dataset now? (y/n): ")
        
        if choice.lower() == 'y':
            print("Starting preprocessing...")
            preprocess_wlasl_dataset(data_path, preprocessed_path, num_classes)
            return load_preprocessed_dataset(preprocessed_path, min_videos_per_class)
    
    # If no preprocessing, load raw data and process on-the-fly
    print("Processing videos on-the-fly (this will be slow)...")
    with open(os.path.join(data_path, "WLASL_v0.3.json"), "r") as f:
        wlasl_data = json.load(f)
        
    return prepare_dataset(wlasl_data, data_path, num_classes)