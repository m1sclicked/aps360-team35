import os
import pandas as pd
import re
import numpy as np
from tqdm import tqdm

def preprocess_asl_citizen_dataset(asl_citizen_path="data/ASL_Citizen", output_path=None, target_glosses=None):
    """
    Preprocess the ASL Citizen dataset by extracting sequence features for specific glosses
    and saving them to disk for fast loading later.
    
    Args:
        asl_citizen_path (str): Path to ASL Citizen dataset
        output_path (str): Path to save preprocessed data (defaults to asl_citizen_path/preprocessed)
        target_glosses (list): List of specific glosses to process (case-insensitive)
                              If None, all glosses will be processed
    
    Returns:
        bool: True if preprocessing was successful, False otherwise
    """
    import os
    import json
    from tqdm import tqdm
    from src.data_processing.dataset_asl import load_asl_citizen_data
    from src.data_processing.dataset import initialize_openpose, process_video_openpose
    
    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.join(asl_citizen_path, "preprocessed")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    print(f"Created output directory: {output_path}")
    
    # Set default target glosses if not provided
    if target_glosses is None:
        # These are the common first 10 words in WLASL as mentioned
        target_glosses = ['book', 'drink', 'computer', 'before', 'chair', 
                          'go', 'clothes', 'who', 'candy', 'cousin']
    
    # Convert target glosses to lowercase for case-insensitive matching
    target_glosses = [gloss.lower() for gloss in target_glosses]
    print(f"Processing the following target glosses: {target_glosses}")
    
    # Load ASL Citizen data from csv files directly to handle column names with spaces
    import pandas as pd
    splits_dir = os.path.join(asl_citizen_path, "splits")
    
    if not os.path.exists(splits_dir):
        print(f"ASL Citizen splits directory not found at {splits_dir}")
        return False
    
    # Load each split CSV directly
    asl_splits = {}
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"{split_name}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded {len(df)} entries from {split_name}.csv")
                asl_splits[split_name] = df
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
        else:
            print(f"Split file not found: {csv_path}")
    
    if not asl_splits:
        print("Failed to load any ASL Citizen data. Exiting.")
        return False
    
    # Combine all splits for processing
    all_videos = pd.concat([df for df in asl_splits.values()])
    print(f"Total ASL Citizen videos: {len(all_videos)}")
    
    # Print columns to debug
    print("CSV columns:", all_videos.columns.tolist())
    
    # Check for video filename column - account for spaces in column names
    video_file_col = None
    possible_cols = ['videofile', 'video', 'filename', 'video_file', 'file', 
                    'Video file', 'Video File', 'Video', 'video file']
    
    for col in possible_cols:
        if col in all_videos.columns:
            video_file_col = col
            print(f"Found video filename column: '{col}'")
            break
    
    if video_file_col is None:
        print("Could not identify the video filename column. Available columns:", all_videos.columns.tolist())
        print("Please check the CSV files and provide the correct column name.")
        
        # Try to make an educated guess based on column content
        for col in all_videos.columns:
            # Sample the first few values to check if they look like filenames
            sample_values = all_videos[col].head(5).astype(str).tolist()
            print(f"Column '{col}' contains values like: {sample_values}")
            
            # Check if any value contains typical video filename patterns
            if any(val.lower().endswith(('.mp4', '.avi', '.mov')) or '.' in val for val in sample_values):
                video_file_col = col
                print(f"Guessing '{col}' as the video filename column based on content")
                break
            
        if video_file_col is None:
            # If we still can't identify, use the second column as a last resort
            # (often in these datasets, first col is ID, second is filename)
            if len(all_videos.columns) >= 2:
                video_file_col = all_videos.columns[1]
                print(f"Using second column '{video_file_col}' as a fallback for video filenames")
            else:
                return False
    
    # Check for Gloss column - account for spaces
    gloss_col = None
    possible_gloss_cols = ['Gloss', 'gloss', 'GLOSS', 'Class', 'class', 'sign', 'Sign']
    
    for col in possible_gloss_cols:
        if col in all_videos.columns:
            gloss_col = col
            print(f"Found gloss column: '{col}'")
            break
    
    if gloss_col is None:
        print("Could not identify the gloss column. Available columns:", all_videos.columns.tolist())
        
        # Try to make an educated guess based on column content
        for col in all_videos.columns:
            if col != video_file_col:  # Skip the video filename column
                # Sample the first few values
                sample_values = all_videos[col].head(5).astype(str).tolist()
                print(f"Column '{col}' contains values like: {sample_values}")
                
                # Check if values look like glosses (short words, no file extensions)
                if all(len(val.split()) <= 2 and '.' not in val for val in sample_values):
                    gloss_col = col
                    print(f"Guessing '{col}' as the gloss column based on content")
                    break
                
        if gloss_col is None:
            # If we still can't identify, use the third column as a last resort
            if len(all_videos.columns) >= 3:
                gloss_col = all_videos.columns[2]
                print(f"Using column '{gloss_col}' as a fallback for glosses")
            else:
                return False
    
    # Clean up glosses by removing trailing numbers
    import re
    all_videos[gloss_col] = all_videos[gloss_col].apply(lambda x: re.sub(r'\d+$', '', str(x)).lower().strip())
    
    # Filter for target glosses
    matching_videos = all_videos[all_videos[gloss_col].str.lower().isin(target_glosses)]
    print(f"Videos with matching target glosses: {len(matching_videos)}")
    
    if len(matching_videos) == 0:
        print("No matching videos found for the target glosses. Exiting.")
        return False
    
    # Initialize OpenPose
    try:
        net, hand_net, input_width, input_height = initialize_openpose(use_gpu=True)
        print("OpenPose initialized successfully")
    except Exception as e:
        print(f"Failed to initialize OpenPose: {e}")
        return False
    
    # Dictionary to store sequence features by gloss and video_id
    videos_features = {}
    # Dictionary to map glosses to video_ids
    gloss_to_videos = {gloss.lower(): [] for gloss in target_glosses}
    
    # Process videos
    videos_dir = os.path.join(asl_citizen_path, "videos")
    if not os.path.exists(videos_dir):
        print(f"ASL Citizen videos directory not found at {videos_dir}")
        return False
    
    # Process matching videos
    total_success = 0
    print(f"Processing {len(matching_videos)} videos...")
    
    for idx, row in tqdm(matching_videos.iterrows(), total=len(matching_videos), desc="Processing ASL Citizen videos"):
        gloss = row[gloss_col].lower()
        video_file = str(row[video_file_col])
        
        # Check if the video file has an extension, if not add .mp4
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_file = f"{video_file}.mp4"
        
        video_path = os.path.join(videos_dir, video_file)
        
        # Skip if video file doesn't exist
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            # Try alternate extensions
            for ext in ['.mp4', '.avi', '.mov']:
                alt_path = os.path.join(videos_dir, f"{str(row[video_file_col])}{ext}")
                if os.path.exists(alt_path):
                    video_path = alt_path
                    video_file = f"{str(row[video_file_col])}{ext}"
                    print(f"Found alternative file: {video_path}")
                    break
            else:
                continue  # Skip if no alternative is found
        
        # Process video with OpenPose
        try:
            print(f"  Processing video for gloss '{gloss}': {video_file} ", end="")
            video_features = process_video_openpose(
                video_path, net, hand_net, 
                input_width, input_height, sample_rate=2
            )
            
            # Only add if processing was successful
            if video_features is not None and len(video_features) > 0:
                video_id = video_file.replace('.mp4', '').replace('.avi', '').replace('.mov', '')
                # Store as a list of lists (sequence of features per frame)
                videos_features[video_id] = [frame.tolist() for frame in video_features]
                gloss_to_videos[gloss].append(video_id)
                total_success += 1
                print(f"✓ Success - {len(video_features)} frames")
            else:
                print("✗ Failed - No hand features detected")
        except Exception as e:
            print(f"✗ Error processing {video_path}: {e}")
    
    # Print statistics
    print(f"Successfully processed {total_success} out of {len(matching_videos)} videos")
    
    for gloss, videos in gloss_to_videos.items():
        if videos:  # Only print if there are videos for this gloss
            print(f"  Gloss '{gloss}': {len(videos)} videos")
    
    # Save the preprocessed data
    if total_success > 0:
        print("\nSaving preprocessed data...")
        
        # Save features
        features_path = os.path.join(output_path, "asl_citizen_features.json")
        with open(features_path, "w") as f:
            json.dump(videos_features, f)
        print(f"Saved sequence features to {features_path}")
        
        # Save gloss mapping
        gloss_map_path = os.path.join(output_path, "asl_citizen_gloss_to_videos.json")
        with open(gloss_map_path, "w") as f:
            json.dump(gloss_to_videos, f)
        print(f"Saved gloss mapping to {gloss_map_path}")
        
        print(f"\nPreprocessing complete! Processed {total_success} videos across {len([g for g, v in gloss_to_videos.items() if v])} glosses.")
        return True
    else:
        print("No videos were successfully processed. Preprocessing failed.")
        return False

def load_asl_citizen_data(data_path="data/ASL_Citizen"):
    """
    Load ASL Citizen dataset from CSV files
    
    Args:
        data_path (str): Path to ASL_Citizen dataset
        
    Returns:
        dict: Dictionary mapping split name to DataFrame
    """
    splits_dir = os.path.join(data_path, "splits")
    
    # Check if the splits directory exists
    if not os.path.exists(splits_dir):
        print(f"ASL Citizen splits directory not found at {splits_dir}")
        return None
    
    # Load each split CSV
    splits = {}
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"{split_name}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Clean up Gloss column by removing trailing numbers
                if "Gloss" in df.columns:
                    df["Gloss"] = df["Gloss"].apply(lambda x: re.sub(r'\d+$', '', x).lower())
                splits[split_name] = df
                print(f"Loaded {len(df)} entries from {split_name}.csv")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
        else:
            print(f"Split file not found: {csv_path}")
    
    return splits

def get_wlasl_class_to_gloss_mapping(wlasl_data):
    """
    Create a mapping from class index to gloss name in WLASL
    
    Args:
        wlasl_data: WLASL dataset JSON
        
    Returns:
        dict: Dictionary mapping class index to gloss name
    """
    class_to_gloss = {}
    for class_idx, sign_class in enumerate(wlasl_data):
        gloss = sign_class["gloss"].lower()
        class_to_gloss[class_idx] = gloss
    
    return class_to_gloss

def get_gloss_to_class_mapping(wlasl_data):
    """
    Create a mapping from gloss name to class index in WLASL
    
    Args:
        wlasl_data: WLASL dataset JSON
        
    Returns:
        dict: Dictionary mapping gloss name to class index
    """
    gloss_to_class = {}
    for class_idx, sign_class in enumerate(wlasl_data):
        gloss = sign_class["gloss"].lower()
        gloss_to_class[gloss] = class_idx
    
    return gloss_to_class

def process_asl_citizen_videos(asl_splits, wlasl_data, data_path, openpose_net, openpose_hand_net, 
                              input_width=368, input_height=368, sample_rate=2):
    """
    Process ASL Citizen videos to extract features for glosses that match the WLASL dataset
    
    Args:
        asl_splits: Dictionary of ASL Citizen data splits
        wlasl_data: WLASL dataset JSON
        data_path: Path to ASL_Citizen dataset
        openpose_net: OpenPose body network
        openpose_hand_net: OpenPose hand network
        input_width, input_height: Input dimensions for the model
        sample_rate: Process every nth frame for efficiency
        
    Returns:
        features: List of sequence features
        labels: Corresponding class labels (matching WLASL classes)
    """
    from src.data_processing.dataset import process_video_openpose
    
    # Create mappings
    gloss_to_class = get_gloss_to_class_mapping(wlasl_data)
    
    # Path to videos directory
    videos_dir = os.path.join(data_path, "videos")
    if not os.path.exists(videos_dir):
        print(f"ASL Citizen videos directory not found at {videos_dir}")
        return [], []
    
    # Combine all splits for processing
    all_videos = pd.concat([df for df in asl_splits.values()])
    print(f"Total ASL Citizen videos to process: {len(all_videos)}")
    
    # Filter for glosses that exist in WLASL
    matching_videos = all_videos[all_videos["Gloss"].str.lower().isin(gloss_to_class.keys())]
    print(f"Videos with matching glosses in WLASL: {len(matching_videos)}")
    
    # Process videos
    features = []
    labels = []
    
    for idx, row in tqdm(matching_videos.iterrows(), total=len(matching_videos), desc="Processing ASL Citizen videos"):
        gloss = row["Gloss"].lower()
        video_file = row["videofile"]
        video_path = os.path.join(videos_dir, video_file)
        
        # Skip if video file doesn't exist
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        
        # Process video with OpenPose
        try:
            video_features = process_video_openpose(
                video_path, openpose_net, openpose_hand_net, 
                input_width, input_height, sample_rate
            )
            
            # Only add if processing was successful
            if video_features is not None and len(video_features) > 0:
                features.append(video_features)
                labels.append(gloss_to_class[gloss])
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    print(f"Successfully processed {len(features)} ASL Citizen videos")
    return features, labels

def combine_datasets(wlasl_features, wlasl_labels, asl_citizen_features, asl_citizen_labels):
    """
    Combine features and labels from both datasets with additional validation
    
    Args:
        wlasl_features: List of WLASL sequence features
        wlasl_labels: List of WLASL labels
        asl_citizen_features: List of ASL Citizen sequence features
        asl_citizen_labels: List of ASL Citizen labels
        
    Returns:
        combined_features: Combined list of sequence features
        combined_labels: Combined list of labels
    """
    import numpy as np
    
    # Validate datasets before combining
    valid_wlasl_indices = []
    for i, seq in enumerate(wlasl_features):
        # Check if sequence is not empty
        if len(seq) > 0:
            # Check if any frame has values
            has_values = False
            for frame in seq:
                if np.any(np.array(frame) != 0):
                    has_values = True
                    break
            
            if has_values:
                valid_wlasl_indices.append(i)
    
    valid_asl_indices = []
    for i, seq in enumerate(asl_citizen_features):
        # Check if sequence is not empty
        if len(seq) > 0:
            # Check if any frame has values
            has_values = False
            for frame in seq:
                if np.any(np.array(frame) != 0):
                    has_values = True
                    break
            
            if has_values:
                valid_asl_indices.append(i)
    
    # Filter datasets
    filtered_wlasl_features = [wlasl_features[i] for i in valid_wlasl_indices]
    filtered_wlasl_labels = [wlasl_labels[i] for i in valid_wlasl_indices]
    
    filtered_asl_features = [asl_citizen_features[i] for i in valid_asl_indices]
    filtered_asl_labels = [asl_citizen_labels[i] for i in valid_asl_indices]
    
    # Get dimensions for validation
    wlasl_dims = set()
    if filtered_wlasl_features and filtered_wlasl_features[0]:
        for seq in filtered_wlasl_features:
            if seq and len(seq) > 0:
                wlasl_dims.add(len(seq[0]))
    
    asl_dims = set()
    if filtered_asl_features and filtered_asl_features[0]:
        for seq in filtered_asl_features:
            if seq and len(seq) > 0:
                asl_dims.add(len(seq[0]))
    
    print(f"WLASL feature dimensions: {wlasl_dims}")
    print(f"ASL Citizen feature dimensions: {asl_dims}")
    
    # If dimensions don't match, we need to adjust
    if wlasl_dims and asl_dims and wlasl_dims != asl_dims:
        # Use the most common dimension in WLASL as target
        target_dim = max(wlasl_dims, key=lambda d: sum(1 for seq in filtered_wlasl_features if seq and len(seq[0]) == d))
        print(f"Adjusting feature dimensions to match: {target_dim}")
        
        # Now we need to adjust ASL Citizen features
        adjusted_asl_features = []
        adjusted_asl_labels = []
        
        for i, seq in enumerate(filtered_asl_features):
            if seq and len(seq) > 0:
                curr_dim = len(seq[0])
                if curr_dim == target_dim:
                    # Dimension already matches
                    adjusted_asl_features.append(seq)
                    adjusted_asl_labels.append(filtered_asl_labels[i])
                elif curr_dim < target_dim:
                    # Pad with zeros
                    adjusted_seq = []
                    for frame in seq:
                        padded_frame = list(frame) + [0.0] * (target_dim - curr_dim)
                        adjusted_seq.append(padded_frame)
                    adjusted_asl_features.append(adjusted_seq)
                    adjusted_asl_labels.append(filtered_asl_labels[i])
                else:
                    # Truncate
                    adjusted_seq = []
                    for frame in seq:
                        adjusted_seq.append(frame[:target_dim])
                    adjusted_asl_features.append(adjusted_seq)
                    adjusted_asl_labels.append(filtered_asl_labels[i])
        
        # Now combine the adjusted features
        combined_features = filtered_wlasl_features + adjusted_asl_features
        combined_labels = filtered_wlasl_labels + adjusted_asl_labels
    else:
        # If dimensions match or one dataset is empty, just combine
        combined_features = filtered_wlasl_features + filtered_asl_features
        combined_labels = filtered_wlasl_labels + filtered_asl_labels
    
    # Print statistics
    wlasl_classes = set(filtered_wlasl_labels)
    asl_citizen_classes = set(filtered_asl_labels)
    combined_classes = set(combined_labels)
    
    print(f"WLASL dataset (filtered): {len(filtered_wlasl_features)} videos, {len(wlasl_classes)} classes")
    print(f"ASL Citizen dataset (filtered): {len(filtered_asl_features)} videos, {len(asl_citizen_classes)} classes")
    print(f"Combined dataset: {len(combined_features)} videos, {len(combined_classes)} classes")
    
    # Check for NaN or inf values in the combined features
    has_nan = False
    for seq_idx, seq in enumerate(combined_features):
        for frame_idx, frame in enumerate(seq):
            if any(np.isnan(v) or np.isinf(v) for v in frame):
                print(f"WARNING: Found NaN or Inf in sequence {seq_idx}, frame {frame_idx}")
                has_nan = True
                # Remove the problematic frames
                combined_features[seq_idx][frame_idx] = [0.0 if np.isnan(v) or np.isinf(v) else v for v in frame]
    
    if has_nan:
        print("Fixed NaN/Inf values by replacing them with zeros")
    
    # Count videos per class in combined dataset
    class_counts = {}
    for label in combined_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("Videos per class in combined dataset:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"  Class {class_idx}: {count} videos")
    
    return combined_features, combined_labels

def preprocess_and_combine_datasets(wlasl_data, config):
    """
    Preprocess both WLASL and ASL Citizen datasets and combine them
    
    Args:
        wlasl_data: WLASL dataset JSON
        config: Configuration dictionary
        
    Returns:
        combined_features: Combined list of sequence features
        combined_labels: Combined list of labels
    """
    from src.data_processing.dataset import (
        initialize_openpose, prepare_dataset_openpose, 
        preprocess_wlasl_dataset_openpose, load_preprocessed_sequence_dataset
    )
    
    # Initialize OpenPose
    try:
        net, hand_net, input_width, input_height = initialize_openpose(use_gpu=True)
    except FileNotFoundError as e:
        print(f"OpenPose initialization failed: {e}")
        print("Please download OpenPose models and place them in the models/openpose directory.")
        return [], []
    
    # Check for preprocessed WLASL data
    wlasl_features = []
    wlasl_labels = []
    preprocessed_path = os.path.join(config["data_path"], "preprocessed")
    
    if os.path.exists(os.path.join(preprocessed_path, "video_sequence_features.json")):
        print("Loading preprocessed WLASL sequence data...")
        wlasl_features, wlasl_labels = load_preprocessed_sequence_dataset(
            preprocessed_path, config.get("min_videos_per_class", 1)
        )
    else:
        print("Processing WLASL videos on-the-fly...")
        wlasl_features, wlasl_labels = prepare_dataset_openpose(
            wlasl_data, config["data_path"], config.get("num_classes", 100)
        )
    
    # Process ASL Citizen dataset
    asl_citizen_path = "data/ASL_Citizen"
    asl_splits = load_asl_citizen_data(asl_citizen_path)
    
    if asl_splits:
        print("Processing ASL Citizen videos...")
        asl_citizen_features, asl_citizen_labels = process_asl_citizen_videos(
            asl_splits, wlasl_data, asl_citizen_path, 
            net, hand_net, input_width, input_height
        )
        
        # Combine datasets
        combined_features, combined_labels = combine_datasets(
            wlasl_features, wlasl_labels, 
            asl_citizen_features, asl_citizen_labels
        )
        
        return combined_features, combined_labels
    else:
        print("ASL Citizen dataset not found or invalid. Using only WLASL dataset.")
        return wlasl_features, wlasl_labels

def load_preprocessed_asl_citizen(asl_citizen_path="data/ASL_Citizen", wlasl_data=None, num_classes=10):
    """
    Load preprocessed ASL Citizen features and map them to WLASL class indices
    
    Args:
        asl_citizen_path (str): Path to ASL Citizen dataset
        wlasl_data (list): WLASL dataset JSON
        num_classes (int): Number of classes to consider from WLASL
        
    Returns:
        tuple: (features, labels) or ([], []) if loading fails
    """
    import os
    import json
    from src.data_processing.dataset_asl import get_gloss_to_class_mapping
    
    # Check if preprocessed data exists
    preprocessed_path = os.path.join(asl_citizen_path, "preprocessed")
    features_path = os.path.join(preprocessed_path, "asl_citizen_features.json")
    gloss_map_path = os.path.join(preprocessed_path, "asl_citizen_gloss_to_videos.json")
    
    if not os.path.exists(features_path) or not os.path.exists(gloss_map_path):
        print("Preprocessed ASL Citizen data not found.")
        return [], []
    
    # Load features and gloss mapping
    try:
        with open(features_path, "r") as f:
            videos_features = json.load(f)
        
        with open(gloss_map_path, "r") as f:
            gloss_to_videos = json.load(f)
        
        print(f"Loaded preprocessed ASL Citizen data with {len(videos_features)} videos")
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return [], []
    
    # If no WLASL data provided, can't map to classes
    if wlasl_data is None:
        print("No WLASL data provided, cannot map to class indices.")
        return [], []
    
    # Create mapping from WLASL glosses to class indices
    gloss_to_class = get_gloss_to_class_mapping(wlasl_data[:num_classes])
    
    # Prepare features and labels
    features = []
    labels = []
    
    # Count matching glosses
    matched_counts = {}
    
    # For each gloss in the ASL Citizen dataset
    for gloss, video_ids in gloss_to_videos.items():
        # Only process if the gloss exists in WLASL dataset
        if gloss in gloss_to_class:
            class_idx = gloss_to_class[gloss]
            matched_counts[gloss] = 0
            
            # For each video of this gloss
            for video_id in video_ids:
                if video_id in videos_features:
                    video_data = videos_features[video_id]
                    features.append(video_data)
                    labels.append(class_idx)
                    matched_counts[gloss] += 1
    
    # Print statistics
    print("ASL Citizen videos matched to WLASL classes:")
    for gloss, count in matched_counts.items():
        if count > 0:
            class_idx = gloss_to_class.get(gloss, "N/A")
            print(f"  Gloss '{gloss}' (Class {class_idx}): {count} videos")
    
    print(f"Total: {len(features)} videos across {len([g for g, c in matched_counts.items() if c > 0])} glosses")
    
    return features, labels