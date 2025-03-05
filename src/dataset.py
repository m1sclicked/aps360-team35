import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
from tqdm import tqdm
import subprocess
import zipfile


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
    """
    try:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return np.zeros(126)

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
            else:
                frames_features.append(np.zeros(126))  # Placeholder for missing hands

        cap.release()

        # If no frames were processed
        if len(frames_features) == 0:
            return np.zeros(126)

        # Return mean of all frame features
        return np.mean(frames_features, axis=0)

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return np.zeros(126)


def load_missing_videos(data_path):
    """
    Load the list of missing video IDs
    """
    missing_path = os.path.join(data_path, "missing.txt")

    if not os.path.exists(missing_path):
        return set()

    with open(missing_path, "r") as f:
        missing_videos = set(line.strip() for line in f)

    return missing_videos


def prepare_dataset(wlasl_data, data_path, num_classes=100, max_samples_per_class=None):
    """
    Prepare dataset by extracting features from videos,
    only processing videos that are actually available
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

    # Iterate through classes
    for class_idx, sign_class in enumerate(tqdm(wlasl_data[:num_classes])):
        class_name = sign_class["gloss"]
        instances = sign_class["instances"]

        # Count how many videos are actually available for this class
        available_class_videos = [
            entry for entry in instances if str(entry["video_id"]) in available_videos
        ]

        # Apply max_samples_per_class if specified
        if max_samples_per_class:
            available_class_videos = available_class_videos[:max_samples_per_class]

        # Process available videos for this class
        class_features = []
        for entry in available_class_videos:
            video_id = str(entry["video_id"])
            video_path = os.path.join(videos_dir, f"{video_id}.mp4")

            # Extract features
            video_features = process_video(video_path, hands)

            class_features.append(video_features)

        # Only add the class if it has features
        if class_features:
            features.extend(class_features)
            labels.extend([class_idx] * len(class_features))
            processed_classes.append(class_name)

    print(f"Processed classes: {processed_classes}")
    print(f"Processed videos: {len(features)}")
    return np.array(features), np.array(labels)
