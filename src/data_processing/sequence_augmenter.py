import numpy as np
import torch
import random
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import copy

class GestureSequenceAugmenter:
    """
    Augmentation class specifically designed for GestureSequenceDataset
    Works with batched tensor data and respects attention masks
    """
    def __init__(self, 
                 jitter_range=0.02,
                 scale_range=(0.9, 1.1),
                 rotation_range=(-15, 15),
                 translation_range=0.05,
                 time_stretch_range=(0.8, 1.2),
                 dropout_prob=0.05,
                 swap_hands_prob=0.3,
                 mirror_prob=0.5,
                 random_start_prob=0.3,
                 speed_variation_prob=0.4,
                 random_frame_drop_prob=0.3,
                 gaussian_noise_std=0.02,
                 # New parameters
                 hand_jitter_prob=0.7,
                 adaptive_resampling_prob=0.5,
                 sequential_consistency_prob=0.4):
        """
        Initialize the GestureSequenceAugmenter with augmentation parameters
        
        Args:
            jitter_range (float): Maximum position jitter range as fraction of coordinate range
            scale_range (tuple): Range of random scaling factors (min, max)
            rotation_range (tuple): Range of rotation angles in degrees (min, max)
            translation_range (float): Maximum translation range as fraction of coordinate range
            time_stretch_range (tuple): Range of time stretching factors (min, max)
            dropout_prob (float): Probability of dropping out individual keypoints
            swap_hands_prob (float): Probability of swapping left and right hands
            mirror_prob (float): Probability of mirroring keypoints horizontally
            random_start_prob (float): Probability of randomly cropping the start of a sequence
            speed_variation_prob (float): Probability of varying speed within a sequence
            random_frame_drop_prob (float): Probability of dropping random frames in a sequence
            gaussian_noise_std (float): Standard deviation for Gaussian noise addition
            hand_jitter_prob (float): Probability of applying enhanced hand-focused jitter
            adaptive_resampling_prob (float): Probability of applying adaptive resampling
            sequential_consistency_prob (float): Probability of preserving sequential consistency
        """
        self.jitter_range = jitter_range
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.time_stretch_range = time_stretch_range
        self.dropout_prob = dropout_prob
        self.swap_hands_prob = swap_hands_prob
        self.mirror_prob = mirror_prob
        self.random_start_prob = random_start_prob
        self.speed_variation_prob = speed_variation_prob
        self.random_frame_drop_prob = random_frame_drop_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.hand_jitter_prob = hand_jitter_prob
        self.adaptive_resampling_prob = adaptive_resampling_prob
        self.sequential_consistency_prob = sequential_consistency_prob
    
    def augment_batch(self, features, masks, apply_prob=1.0):
        """
        Apply augmentation to a batch of sequences from GestureSequenceDataset
        
        Args:
            features (torch.Tensor): Batch of features (batch_size, seq_len, feature_dim)
            masks (torch.Tensor): Attention masks (batch_size, seq_len)
            apply_prob (float): Probability of applying augmentation to each sample
            
        Returns:
            torch.Tensor: Augmented features
            torch.Tensor: Updated masks
        """
        batch_size = features.shape[0]
        aug_features = features.clone()
        aug_masks = masks.clone()
        
        # Apply augmentations individually to each sample in batch
        for i in range(batch_size):
            if random.random() < apply_prob:
                aug_features[i], aug_masks[i] = self.augment_sequence(
                    aug_features[i], aug_masks[i]
                )
        
        return aug_features, aug_masks
    
    def augment_sequence(self, sequence, mask):
        """
        Apply augmentation to a single sequence and its mask
        
        Args:
            sequence (torch.Tensor): Feature sequence (seq_len, feature_dim)
            mask (torch.Tensor): Attention mask (seq_len)
            
        Returns:
            torch.Tensor: Augmented sequence
            torch.Tensor: Updated mask
        """
        # Get actual sequence length from mask
        seq_len = mask.sum().int().item()
        feature_dim = sequence.shape[1]
        
        # Skip empty sequences
        if seq_len == 0:
            return sequence, mask
        
        # Get only the valid part of the sequence
        valid_seq = sequence[:seq_len].clone()
        
        # Now apply all augmentations to valid sequence
        valid_seq = self._apply_all_augmentations(valid_seq)

        if torch.isnan(valid_seq).any() or torch.isinf(valid_seq).any():
            # If NaN or Inf values are detected, revert to the original sequence
            print("Warning: NaN/Inf values detected after augmentation. Using original sequence.")
            valid_seq = sequence[:seq_len].clone()
        
        # Handle possible sequence length change
        if valid_seq.shape[0] != seq_len:
            new_seq_len = valid_seq.shape[0]
            new_sequence = torch.zeros_like(sequence)
            new_mask = torch.zeros_like(mask)
            
            # Copy augmented data to new tensors
            copy_len = min(new_seq_len, new_sequence.shape[0])
            new_sequence[:copy_len] = valid_seq[:copy_len]
            new_mask[:copy_len] = 1
            
            return new_sequence, new_mask
        else:
            # No length change, just update valid part
            sequence[:seq_len] = valid_seq
            return sequence, mask
    
    def _apply_all_augmentations(self, sequence):
        """
        Apply all possible augmentations to a sequence
        
        Args:
            sequence (torch.Tensor): Feature sequence to augment
            
        Returns:
            torch.Tensor: Augmented sequence
        """
        device = sequence.device
        
        # Convert to NumPy for easier manipulation
        seq_np = sequence.cpu().numpy()
        
        if len(seq_np) < 3:
            # Skip temporal augmentations for very short sequences
            pass
        else:
            # Apply temporal augmentations with sequential consistency
            seq_np = self._preserve_sequential_consistency(seq_np)
            
            # Apply adaptive resampling if it wasn't applied by sequential_consistency
            if len(seq_np) >= 5:
                seq_np = self._apply_adaptive_resampling(seq_np)
            elif self.time_stretch_range[0] != 1.0 and len(seq_np) >= 3:
                # Fall back to regular time stretch for shorter sequences
                if random.random() < 0.5:
                    seq_np = self._apply_time_stretch(seq_np)
            
            # Apply other temporal augmentations
            if random.random() < self.speed_variation_prob and len(seq_np) >= 5:
                seq_np = self._apply_speed_variation(seq_np)
            
            if random.random() < self.random_start_prob and len(seq_np) >= 3:
                seq_np = self._apply_random_start(seq_np)
                
            if random.random() < self.random_frame_drop_prob and len(seq_np) >= 5:
                seq_np = self._apply_random_frame_drop(seq_np)
        
        # Apply spatial augmentations frame by frame
        # Pre-compute augmentation parameters for consistency
        do_swap_hands = random.random() < self.swap_hands_prob
        do_mirror = random.random() < self.mirror_prob
        scale_factor = random.uniform(*self.scale_range)
        rotation_angle = random.uniform(*self.rotation_range)
        translation = (random.uniform(-self.translation_range, self.translation_range),
                       random.uniform(-self.translation_range, self.translation_range))
        
        # Apply the same transformations to each frame
        for i in range(len(seq_np)):
            frame = seq_np[i]
            
            # Assuming equal split between left and right hand
            half_idx = len(frame) // 2
            left_hand = frame[:half_idx]
            right_hand = frame[half_idx:]
            
            # Apply hand swapping if selected
            if do_swap_hands:
                frame = np.concatenate([right_hand, left_hand])
            
            # Apply mirroring if selected
            if do_mirror:
                frame = self._mirror_keypoints(frame)
            
            # Apply hand-focused jitter (with built-in probability)
            frame = self._apply_hand_focused_jitter(frame)
            
            # If hand-focused jitter wasn't applied, try regular jitter
            if random.random() < 0.3:
                frame = self._apply_jitter(frame)
            
            # Apply other spatial transformations
            frame = self._apply_scaling(frame, scale_factor)
            frame = self._apply_rotation(frame, rotation_angle)
            frame = self._apply_translation(frame, translation)
            
            # Apply random dropout
            frame = self._apply_dropout(frame)
            
            # Apply Gaussian noise
            frame = self._apply_gaussian_noise(frame)
            
            seq_np[i] = frame
        
        # Convert back to torch tensor on the original device
        return torch.tensor(seq_np, device=device, dtype=sequence.dtype)
    
    def _mirror_keypoints(self, keypoints):
        """Mirror keypoints horizontally (flip x-coordinates)"""
        mirrored = keypoints.copy()
        
        # Assuming 3 dimensions per keypoint (x, y, conf)
        for i in range(0, len(keypoints), 3):
            if mirrored[i] != 0:  # Only flip valid keypoints
                mirrored[i] = 1.0 - mirrored[i]
        
        return mirrored
    
    def _apply_jitter(self, keypoints):
        """Add random jitter to keypoint positions"""
        jittered = keypoints.copy()
        
        # Apply jitter only to x,y coordinates (not confidence)
        for i in range(0, len(keypoints), 3):
            if jittered[i] != 0 or jittered[i+1] != 0:
                jittered[i] += random.uniform(-self.jitter_range, self.jitter_range)
                jittered[i+1] += random.uniform(-self.jitter_range, self.jitter_range)
        
        return jittered
    
    def _apply_hand_focused_jitter(self, keypoints, prob=None):
        """
        Apply more jitter to hands than to other keypoints
        
        Args:
            keypoints: Array of keypoint coordinates
            prob: Probability of applying hand-focused jitter (defaults to self.hand_jitter_prob)
            
        Returns:
            Jittered keypoints
        """
        # Use class-level probability if none provided
        if prob is None:
            prob = self.hand_jitter_prob
        
        # Skip based on probability
        if random.random() > prob:
            return keypoints
        
        jittered = keypoints.copy()
        
        # Apply stronger jitter to hand points (2-3x more than regular jitter)
        hand_jitter_range = self.jitter_range * 2.5
        
        # Assuming 3 dimensions per keypoint (x, y, conf)
        for i in range(0, len(keypoints), 3):
            if jittered[i] != 0 or jittered[i+1] != 0:  # Only modify valid keypoints
                # Apply stronger jitter to x,y coordinates
                jittered[i] += random.uniform(-hand_jitter_range, hand_jitter_range)
                jittered[i+1] += random.uniform(-hand_jitter_range, hand_jitter_range)
        
        return jittered
    
    def _apply_scaling(self, keypoints, scale_factor=None):
        """Apply scaling to keypoints"""
        if scale_factor is None:
            scale_factor = random.uniform(*self.scale_range)
        
        scaled = keypoints.copy()
        
        # Find center point of valid keypoints
        valid_x, valid_y = [], []
        for i in range(0, len(keypoints), 3):
            if keypoints[i] != 0 or keypoints[i+1] != 0:
                valid_x.append(keypoints[i])
                valid_y.append(keypoints[i+1])
        
        if not valid_x:  # No valid keypoints
            return scaled
        
        center_x = np.mean(valid_x)
        center_y = np.mean(valid_y)
        
        # Scale around center
        for i in range(0, len(keypoints), 3):
            if scaled[i] != 0 or scaled[i+1] != 0:
                scaled[i] = center_x + (scaled[i] - center_x) * scale_factor
                scaled[i+1] = center_y + (scaled[i+1] - center_y) * scale_factor
        
        return scaled
    
    def _apply_rotation(self, keypoints, angle=None):
        """Apply rotation to keypoints"""
        if angle is None:
            angle = random.uniform(*self.rotation_range)
        
        rotated = keypoints.copy()
        angle_rad = np.radians(angle)
        
        # Find center point of valid keypoints
        valid_x, valid_y = [], []
        for i in range(0, len(keypoints), 3):
            if keypoints[i] != 0 or keypoints[i+1] != 0:
                valid_x.append(keypoints[i])
                valid_y.append(keypoints[i+1])
        
        if not valid_x:  # No valid keypoints
            return rotated
        
        center_x = np.mean(valid_x)
        center_y = np.mean(valid_y)
        
        # Rotation matrix elements
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Apply rotation around center
        for i in range(0, len(keypoints), 3):
            if rotated[i] != 0 or rotated[i+1] != 0:
                x = rotated[i] - center_x
                y = rotated[i+1] - center_y
                
                # Rotate
                new_x = x * cos_angle - y * sin_angle
                new_y = x * sin_angle + y * cos_angle
                
                rotated[i] = new_x + center_x
                rotated[i+1] = new_y + center_y
        
        return rotated
    
    def _apply_translation(self, keypoints, translation=None):
        """Apply translation to keypoints"""
        if translation is None:
            dx = random.uniform(-self.translation_range, self.translation_range)
            dy = random.uniform(-self.translation_range, self.translation_range)
        else:
            dx, dy = translation
        
        translated = keypoints.copy()
        
        # Apply translation to x,y coordinates
        for i in range(0, len(keypoints), 3):
            if translated[i] != 0 or translated[i+1] != 0:
                translated[i] += dx
                translated[i+1] += dy
        
        return translated
    
    def _apply_dropout(self, keypoints):
        """Randomly drop out keypoints by zeroing them"""
        dropout = keypoints.copy()
        
        # Apply dropout to entire keypoints (x,y,conf)
        for i in range(0, len(keypoints), 3):
            if random.random() < self.dropout_prob:
                dropout[i:i+3] = 0
        
        return dropout
    
    def _apply_gaussian_noise(self, keypoints):
        """Add Gaussian noise to keypoint positions"""
        noisy = keypoints.copy()
        
        # Apply noise only to x,y coordinates (not confidence)
        for i in range(0, len(keypoints), 3):
            if noisy[i] != 0 or noisy[i+1] != 0:
                # Add noise to x,y coordinates
                noisy[i] += np.random.normal(0, self.gaussian_noise_std)
                noisy[i+1] += np.random.normal(0, self.gaussian_noise_std)
                
                # Optionally add small noise to confidence as well (less than position noise)
                confidence_noise = np.random.normal(0, self.gaussian_noise_std * 0.25)
                noisy[i+2] = max(0.0, min(1.0, noisy[i+2] + confidence_noise))
        
        return noisy
    
    def _apply_time_stretch(self, sequence):
        """Apply time stretching to a sequence with robust error handling"""
        n_frames, n_features = sequence.shape
        
        # Skip for very short sequences
        if n_frames < 3:
            return sequence
        
        # Apply a random stretch factor
        stretch_factor = random.uniform(*self.time_stretch_range)
        new_n_frames = max(3, int(n_frames * stretch_factor))
        
        # Manual linear interpolation - more robust than scipy.interp1d
        new_sequence = np.zeros((new_n_frames, n_features))
        
        # Create mapping between old and new indices
        orig_indices = np.arange(n_frames)
        new_indices = np.linspace(0, n_frames - 1, new_n_frames)
        
        # For each point in new sequence
        for i in range(new_n_frames):
            idx = new_indices[i]
            
            # Find the two original points to interpolate between
            idx_floor = int(np.floor(idx))
            idx_ceil = min(int(np.ceil(idx)), n_frames - 1)
            
            # If they're the same (exact index), just copy
            if idx_floor == idx_ceil:
                new_sequence[i] = sequence[idx_floor]
                continue
            
            # Calculate the interpolation weight
            weight = idx - idx_floor
            
            # Linear interpolation (safer than scipy's interpolation)
            new_sequence[i] = (1 - weight) * sequence[idx_floor] + weight * sequence[idx_ceil]
        
        return new_sequence
    
    def _apply_adaptive_resampling(self, sequence, prob=None):
        """
        Resample sequence with non-uniform time steps to preserve important motion
        
        Args:
            sequence: Array of shape [n_frames, n_features]
            prob: Probability of applying adaptive resampling
            
        Returns:
            Resampled sequence
        """
        # Use class-level probability if none provided
        if prob is None:
            prob = self.adaptive_resampling_prob
        
        # Skip based on probability
        if random.random() > prob or len(sequence) < 5:
            return sequence
        
        n_frames, n_features = sequence.shape
        
        # Calculate motion energy between frames
        motion_energy = np.zeros(n_frames)
        for i in range(1, n_frames):
            motion_energy[i] = np.sum(np.abs(sequence[i] - sequence[i-1]))
        
        # Add baseline importance to ensure all frames have some probability
        importance = motion_energy + np.mean(motion_energy) * 0.2
        importance = importance / importance.sum()
        
        # Create cumulative distribution for sampling
        cum_importance = np.cumsum(importance)
        
        # Define new sequence length based on time_stretch_range
        stretch_factor = random.uniform(*self.time_stretch_range)
        new_n_frames = max(3, int(n_frames * stretch_factor))
        
        # Initialize new sequence
        new_sequence = np.zeros((new_n_frames, n_features))
        
        # Always include first and last frames
        new_sequence[0] = sequence[0]
        if new_n_frames > 1:
            new_sequence[-1] = sequence[-1]
        
        # Sample intermediate frames based on importance
        if new_n_frames > 2:
            # Create sampling points that preserve high-motion frames
            sample_points = np.linspace(0, cum_importance[-1], new_n_frames)
            
            # Map each point to the corresponding frame
            for i in range(1, new_n_frames-1):
                # Find the frame index this point corresponds to
                idx = np.searchsorted(cum_importance, sample_points[i])
                idx = min(idx, n_frames-1)
                new_sequence[i] = sequence[idx]
        
        return new_sequence
    
    def _apply_random_start(self, sequence):
        """Randomly crop the start of a sequence"""
        seq_len = len(sequence)
        
        # Only apply if sequence is long enough
        if seq_len < 3:
            return sequence
        
        # Choose a random start point in the first third
        max_start = seq_len // 3
        start_idx = random.randint(0, max_start)
        
        return sequence[start_idx:]
    
    def _apply_random_frame_drop(self, sequence):
        """Randomly drop frames from a sequence"""
        seq_len = len(sequence)
        
        # Only apply if sequence is long enough
        if seq_len < 5:
            return sequence
        
        # Determine how many frames to drop (between 5-20% of sequence)
        drop_ratio = random.uniform(0.05, 0.2)
        num_to_drop = max(1, int(seq_len * drop_ratio))
        
        # Ensure we don't drop too many frames
        num_to_drop = min(num_to_drop, seq_len // 3)
        
        # Randomly select frames to drop
        drop_indices = set(random.sample(range(seq_len), num_to_drop))
        
        # Create new sequence without dropped frames
        new_sequence = np.array([frame for i, frame in enumerate(sequence) if i not in drop_indices])
        
        return new_sequence
    
    def _apply_speed_variation(self, sequence):
        """Apply variable speed within a sequence with robust error handling"""
        n_frames, n_features = sequence.shape
        
        # Only apply if sequence is long enough
        if n_frames < 5:
            return sequence
        
        # Create random time mapping
        uniform_time = np.linspace(0, 1, n_frames)
        
        # Create non-uniform time with 2-4 control points
        num_points = random.randint(2, 4)
        control_x = np.linspace(0, 1, num_points)
        control_y = np.array([0] + [random.uniform(0.3, 0.7) 
                                    for _ in range(num_points-2)] + [1])
        
        # Ensure monotonically increasing
        control_y.sort()
        
        # Interpolate to create non-uniform time
        non_uniform_time = np.interp(uniform_time, control_x, control_y)
        
        # Create time points for interpolation
        new_frames = sequence.shape[0]  # Keep same number of frames
        interp_time = non_uniform_time * (n_frames - 1)
        
        # Manual interpolation instead of scipy.interp1d
        new_sequence = np.zeros_like(sequence)
        
        for i in range(new_frames):
            idx = interp_time[i]
            idx_floor = int(np.floor(idx))
            idx_ceil = min(int(np.ceil(idx)), n_frames - 1)
            
            # Handle edge case of exact indices
            if idx_floor == idx_ceil:
                new_sequence[i] = sequence[idx_floor]
                continue
            
            # Calculate interpolation weight
            weight = idx - idx_floor
            
            # Linear interpolation
            new_sequence[i] = (1 - weight) * sequence[idx_floor] + weight * sequence[idx_ceil]
        
        return new_sequence
    
    def _preserve_sequential_consistency(self, sequence, prob=None):
        """
        Ensure temporal transformations preserve the logical flow of signs
        
        Args:
            sequence: Array of shape [n_frames, n_features]
            prob: Probability of applying this transformation
            
        Returns:
            Transformed sequence
        """
        # Use class-level probability if none provided
        if prob is None:
            prob = self.sequential_consistency_prob
        
        # Skip based on probability
        if random.random() > prob or len(sequence) < 5:
            return sequence
            
        n_frames, n_features = sequence.shape
            
        # Calculate motion energy
        motion = np.zeros(n_frames-1)
        for i in range(n_frames-1):
            motion[i] = np.sum(np.abs(sequence[i+1] - sequence[i]))
        
        # Detect key frames at motion peaks
        try:
            peaks, _ = find_peaks(motion, distance=max(2, n_frames//10))
        except ImportError:
            # Fallback if scipy is not available
            peaks = []
            for i in range(1, len(motion)-1):
                if motion[i] > motion[i-1] and motion[i] > motion[i+1]:
                    peaks.append(i)
            peaks = np.array(peaks)
        
        # Add start and end frames if not already included
        if len(peaks) == 0 or 0 not in peaks:
            peaks = np.concatenate([[0], peaks])
        if n_frames-1 not in peaks and n_frames-2 not in peaks:
            peaks = np.concatenate([peaks, [n_frames-1]])
        
        # Sort peaks
        peaks.sort()
        
        # Create new sequence that preserves key frames
        new_sequence = []
        
        # Process each segment between key frames
        for i in range(len(peaks)-1):
            start_idx, end_idx = peaks[i], peaks[i+1]
            segment = sequence[start_idx:end_idx+1]
            
            # Apply mild time stretch to this segment
            stretch_factor = random.uniform(0.9, 1.1)  # More conservative stretch
            new_length = max(2, int(len(segment) * stretch_factor))
            
            # Interpolate frames within segment
            if new_length == len(segment):
                new_segment = segment
            else:
                # Linear interpolation
                idx = np.linspace(0, len(segment)-1, new_length)
                new_segment = np.zeros((new_length, n_features))
                
                for j in range(new_length):
                    if idx[j].is_integer():
                        new_segment[j] = segment[int(idx[j])]
                    else:
                        # Interpolate between frames
                        f1 = int(np.floor(idx[j]))
                        f2 = int(np.ceil(idx[j]))
                        weight = idx[j] - f1
                        new_segment[j] = (1 - weight) * segment[f1] + weight * segment[f2]
            
            # Add to result
            new_sequence.append(new_segment)
        
        # Concatenate all segments
        return np.concatenate(new_sequence)


class AugmentedGestureSequenceDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies augmentation to GestureSequenceDataset
    """
    def __init__(self, base_dataset, augmenter, augment_prob=0.8):
        """
        Initialize with a base dataset and augmenter
        
        Args:
            base_dataset: The original GestureSequenceDataset instance
            augmenter: GestureSequenceAugmenter instance
            augment_prob: Probability of augmenting each sample
        """
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.augment_prob = augment_prob
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        features, mask, label = self.base_dataset[idx]
        
        # Apply augmentation with probability
        if random.random() < self.augment_prob:
            # Add batch dimension for augmenter
            batched_features = features.unsqueeze(0)
            batched_mask = mask.unsqueeze(0)
            
            # Apply augmentation
            aug_features, aug_mask = self.augmenter.augment_batch(
                batched_features, batched_mask
            )
            
            # Remove batch dimension
            features = aug_features.squeeze(0)
            mask = aug_mask.squeeze(0)
        
        return features, mask, label