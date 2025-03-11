import numpy as np
from scipy.interpolate import interp1d
import random
import copy

class KeypointAugmenter:
    """
    Class for augmenting keypoint data for sign language recognition
    Specifically designed for hand keypoint data from OpenPose or MediaPipe
    """
    
    def __init__(self, 
                 jitter_range=0.02,
                 scale_range=(0.9, 1.1),
                 rotation_range=(-15, 15),
                 translation_range=0.05,
                 time_stretch_range=(0.8, 1.2),
                 dropout_prob=0.05,
                 swap_hands_prob=0.3,
                 mirror_prob=0.5):
        """
        Initialize the KeypointAugmenter with augmentation parameters
        
        Args:
            jitter_range (float): Maximum position jitter range as fraction of coordinate range
            scale_range (tuple): Range of random scaling factors (min, max)
            rotation_range (tuple): Range of rotation angles in degrees (min, max)
            translation_range (float): Maximum translation range as fraction of coordinate range
            time_stretch_range (tuple): Range of time stretching factors (min, max)
            dropout_prob (float): Probability of dropping out individual keypoints
            swap_hands_prob (float): Probability of swapping left and right hands
            mirror_prob (float): Probability of mirroring keypoints horizontally
        """
        self.jitter_range = jitter_range
        self.scale_range = scale_range
        self.rotation_range = rotation_range  # in degrees
        self.translation_range = translation_range
        self.time_stretch_range = time_stretch_range
        self.dropout_prob = dropout_prob
        self.swap_hands_prob = swap_hands_prob
        self.mirror_prob = mirror_prob
    
    def augment_sample(self, keypoints, is_sequence=False):
        """
        Apply augmentation to a single sample or sequence of keypoints
        
        Args:
            keypoints (numpy.ndarray): Keypoint data - either a single frame or sequence
                For single frame: shape (n_keypoints * n_dims)
                For sequence: shape (n_frames, n_keypoints * n_dims)
            is_sequence (bool): Whether the input is a sequence of frames
        
        Returns:
            numpy.ndarray: Augmented keypoints with same shape as input
        """
        if is_sequence:
            return self._augment_sequence(keypoints)
        else:
            return self._augment_frame(keypoints)
    
    def _augment_frame(self, keypoints):
        """
        Apply augmentation to a single frame of keypoints
        
        Args:
            keypoints (numpy.ndarray): Single frame keypoint data
                Shape: (n_keypoints * n_dims)
        
        Returns:
            numpy.ndarray: Augmented keypoints
        """
        # Make a copy to avoid modifying the original
        augmented = keypoints.copy()
        
        # Assuming a flattened array with format [left_hand, right_hand]
        # and each hand has 21 keypoints with 3 dimensions (x, y, conf)
        half_idx = len(augmented) // 2
        
        # Extract hands separately (assuming equal split)
        left_hand = augmented[:half_idx]
        right_hand = augmented[half_idx:]
        
        # Apply random swapping of hands
        if random.random() < self.swap_hands_prob:
            augmented = np.concatenate([right_hand, left_hand])
        
        # Apply horizontal mirroring
        if random.random() < self.mirror_prob:
            augmented = self._mirror_keypoints(augmented)
        
        # Apply spatial jittering
        if self.jitter_range > 0:
            augmented = self._apply_jitter(augmented)
        
        # Apply random scaling
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            augmented = self._apply_scaling(augmented)
        
        # Apply random rotation
        if self.rotation_range[0] != 0 or self.rotation_range[1] != 0:
            augmented = self._apply_rotation(augmented)
        
        # Apply random translation
        if self.translation_range > 0:
            augmented = self._apply_translation(augmented)
        
        # Apply keypoint dropout (set to zero)
        if self.dropout_prob > 0:
            augmented = self._apply_dropout(augmented)
        
        return augmented
    
    def _augment_sequence(self, sequence):
        """
        Apply augmentation to a sequence of keypoint frames
        
        Args:
            sequence (numpy.ndarray): Sequence of keypoint frames
                Shape: (n_frames, n_keypoints * n_dims)
        
        Returns:
            numpy.ndarray: Augmented sequence
        """
        # Make a copy to avoid modifying the original
        augmented_seq = sequence.copy()
        
        # Apply time stretching
        if self.time_stretch_range[0] != 1.0 or self.time_stretch_range[1] != 1.0:
            augmented_seq = self._apply_time_stretch(augmented_seq)
        
        # Apply frame-wise augmentations consistently across the sequence
        # Pre-determine augmentation parameters
        do_swap_hands = random.random() < self.swap_hands_prob
        do_mirror = random.random() < self.mirror_prob
        scale_factor = random.uniform(*self.scale_range)
        rotation_angle = random.uniform(*self.rotation_range)
        translation = (random.uniform(-self.translation_range, self.translation_range),
                       random.uniform(-self.translation_range, self.translation_range))
        
        # Apply the same transformations to each frame
        for i in range(len(augmented_seq)):
            frame = augmented_seq[i]
            half_idx = len(frame) // 2
            
            # Extract hands separately
            left_hand = frame[:half_idx]
            right_hand = frame[half_idx:]
            
            # Apply hand swapping
            if do_swap_hands:
                frame = np.concatenate([right_hand, left_hand])
            
            # Apply mirroring
            if do_mirror:
                frame = self._mirror_keypoints(frame)
            
            # Apply consistent spatial transformations
            frame = self._apply_jitter(frame)
            frame = self._apply_scaling(frame, scale_factor)
            frame = self._apply_rotation(frame, rotation_angle)
            frame = self._apply_translation(frame, translation)
            frame = self._apply_dropout(frame)
            
            augmented_seq[i] = frame
        
        return augmented_seq
    
    def _mirror_keypoints(self, keypoints):
        """
        Mirror keypoints horizontally (flip x-coordinates)
        
        Args:
            keypoints (numpy.ndarray): Keypoint data
        
        Returns:
            numpy.ndarray: Mirrored keypoints
        """
        mirrored = keypoints.copy()
        
        # Assuming 3 dimensions per keypoint (x, y, conf)
        # Flip only the x-coordinates (every 3rd starting from 0)
        for i in range(0, len(keypoints), 3):
            # Invert x-coordinate (assuming normalized 0-1 range)
            # If your coordinates are in pixel space, you'll need to modify this
            if mirrored[i] != 0:  # Only flip valid keypoints
                mirrored[i] = 1.0 - mirrored[i]
        
        return mirrored
    
    def _apply_jitter(self, keypoints):
        """
        Add random jitter to keypoint positions
        
        Args:
            keypoints (numpy.ndarray): Keypoint data
        
        Returns:
            numpy.ndarray: Jittered keypoints
        """
        jittered = keypoints.copy()
        
        # Apply jitter only to x,y coordinates (not confidence)
        for i in range(0, len(keypoints), 3):
            # Only jitter if keypoint is valid (non-zero)
            if jittered[i] != 0 or jittered[i+1] != 0:
                # Add jitter to x
                jittered[i] += random.uniform(-self.jitter_range, self.jitter_range)
                # Add jitter to y
                jittered[i+1] += random.uniform(-self.jitter_range, self.jitter_range)
        
        return jittered
    
    def _apply_scaling(self, keypoints, scale_factor=None):
        """
        Apply random scaling to keypoints
        
        Args:
            keypoints (numpy.ndarray): Keypoint data
            scale_factor (float, optional): If provided, use this scale factor
        
        Returns:
            numpy.ndarray: Scaled keypoints
        """
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
                # Scale relative to center
                scaled[i] = center_x + (scaled[i] - center_x) * scale_factor
                scaled[i+1] = center_y + (scaled[i+1] - center_y) * scale_factor
        
        return scaled
    
    def _apply_rotation(self, keypoints, angle=None):
        """
        Apply random rotation to keypoints
        
        Args:
            keypoints (numpy.ndarray): Keypoint data
            angle (float, optional): If provided, use this rotation angle
        
        Returns:
            numpy.ndarray: Rotated keypoints
        """
        if angle is None:
            angle = random.uniform(*self.rotation_range)
        
        rotated = keypoints.copy()
        
        # Convert angle to radians
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
                # Translate to origin
                x = rotated[i] - center_x
                y = rotated[i+1] - center_y
                
                # Rotate
                new_x = x * cos_angle - y * sin_angle
                new_y = x * sin_angle + y * cos_angle
                
                # Translate back
                rotated[i] = new_x + center_x
                rotated[i+1] = new_y + center_y
        
        return rotated
    
    def _apply_translation(self, keypoints, translation=None):
        """
        Apply random translation to keypoints
        
        Args:
            keypoints (numpy.ndarray): Keypoint data
            translation (tuple, optional): If provided, use this translation (dx, dy)
        
        Returns:
            numpy.ndarray: Translated keypoints
        """
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
        """
        Randomly drop out keypoints by zeroing them
        
        Args:
            keypoints (numpy.ndarray): Keypoint data
        
        Returns:
            numpy.ndarray: Keypoints with random dropout
        """
        dropout = keypoints.copy()
        
        # Apply dropout to entire keypoints (x,y,conf)
        for i in range(0, len(keypoints), 3):
            if random.random() < self.dropout_prob:
                dropout[i:i+3] = 0
        
        return dropout
    
    def _apply_time_stretch(self, sequence):
        """
        Apply time stretching to a sequence of keypoints
        
        Args:
            sequence (numpy.ndarray): Sequence of keypoint frames
                Shape: (n_frames, n_keypoints * n_dims)
        
        Returns:
            numpy.ndarray: Time-stretched sequence
        """
        n_frames, n_features = sequence.shape
        
        # Apply a random stretch factor
        stretch_factor = random.uniform(*self.time_stretch_range)
        new_n_frames = max(2, int(n_frames * stretch_factor))
        
        # Create time points for interpolation
        orig_time = np.arange(n_frames)
        new_time = np.linspace(0, n_frames - 1, new_n_frames)
        
        # Create interpolation function
        interp_func = interp1d(orig_time, sequence, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Apply interpolation
        stretched_seq = interp_func(new_time)
        
        return stretched_seq

def augment_dataset(features, labels, augmenter, n_augmentations=2, is_sequence=False):
    """
    Generate augmented versions of the dataset
    
    Args:
        features (numpy.ndarray): Original feature array
        labels (numpy.ndarray): Original labels array 
        augmenter (KeypointAugmenter): Augmentation object
        n_augmentations (int): Number of augmented copies to generate per sample
        is_sequence (bool): Whether the features are sequences
    
    Returns:
        numpy.ndarray: Augmented features
        numpy.ndarray: Corresponding labels
    """
    augmented_features = list(features)
    augmented_labels = list(labels)
    
    # For each original sample, generate n_augmentations
    for i in range(len(features)):
        for _ in range(n_augmentations):
            aug_feature = augmenter.augment_sample(features[i], is_sequence=is_sequence)
            augmented_features.append(aug_feature)
            augmented_labels.append(labels[i])
    
    return np.array(augmented_features), np.array(augmented_labels)

def apply_data_augmentation(features, labels, is_sequence=False):
    """
    Example function showing how to use the augmentation tools
    
    Args:
        features (numpy.ndarray): Original features
        labels (numpy.ndarray): Original labels
        is_sequence (bool): Whether the features are sequences
        
    Returns:
        numpy.ndarray: Augmented features
        numpy.ndarray: Corresponding labels
    """
    print(f"Original dataset: {len(features)} samples")
    
    # Create augmenter with default parameters
    augmenter = KeypointAugmenter(
        jitter_range=0.02,
        scale_range=(0.9, 1.1),
        rotation_range=(-15, 15),
        translation_range=0.05,
        dropout_prob=0.05,
        swap_hands_prob=0.3
    )
    
    # Generate augmented dataset
    aug_features, aug_labels = augment_dataset(
        features, labels, augmenter, n_augmentations=3, is_sequence=is_sequence
    )
    
    print(f"Augmented dataset: {len(aug_features)} samples")
    
    return aug_features, aug_labels