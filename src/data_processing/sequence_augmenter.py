import numpy as np
import torch
import random
from scipy.interpolate import interp1d
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
                 speed_variation_prob=0.4):
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
            sequence (torch.Tensor): Valid sequence (seq_len, feature_dim)
            
        Returns:
            torch.Tensor: Augmented sequence
        """
        device = sequence.device
        
        # Convert to NumPy for easier manipulation
        seq_np = sequence.cpu().numpy()
        
        # Apply temporal augmentations
        if random.random() < self.speed_variation_prob:
            seq_np = self._apply_speed_variation(seq_np)
        
        if random.random() < 0.5 and self.time_stretch_range[0] != 1.0:
            seq_np = self._apply_time_stretch(seq_np)
        
        if random.random() < self.random_start_prob:
            seq_np = self._apply_random_start(seq_np)
        
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
            
            # Apply spatial transformations
            frame = self._apply_jitter(frame)
            frame = self._apply_scaling(frame, scale_factor)
            frame = self._apply_rotation(frame, rotation_angle)
            frame = self._apply_translation(frame, translation)
            
            # Apply random dropout
            frame = self._apply_dropout(frame)
            
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
    
    def _apply_time_stretch(self, sequence):
        """Apply time stretching to a sequence"""
        n_frames, n_features = sequence.shape
        
        # Apply a random stretch factor
        stretch_factor = random.uniform(*self.time_stretch_range)
        new_n_frames = max(2, int(n_frames * stretch_factor))
        
        # Create time points for interpolation
        orig_time = np.arange(n_frames)
        new_time = np.linspace(0, n_frames - 1, new_n_frames)
        
        # Create interpolation function
        interp_func = interp1d(orig_time, sequence, axis=0, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        
        # Apply interpolation
        return interp_func(new_time)
    
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
    
    def _apply_speed_variation(self, sequence):
        """Apply variable speed within a sequence"""
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
        orig_time = np.arange(n_frames)
        interp_time = non_uniform_time * (n_frames - 1)
        
        # Create interpolation function
        interp_func = interp1d(orig_time, sequence, axis=0, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        
        # Apply interpolation to get variable speed sequence
        return interp_func(interp_time)


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