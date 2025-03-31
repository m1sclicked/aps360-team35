config = {
    "num_classes": 10,
    "batch_size": 64,  # Increased from 32 to 64
    "num_epochs": 100,
    "learning_rate": 0.0003,
    "data_path": "data/wlasl_data",
    "use_data_augmentation": True,
    "augmentation_factor": 3,
    
    # Updated augmentation parameters to match the new GestureSequenceAugmenter
    "augmentation_params": {
        "jitter_range": 0.03,
        "scale_range": [0.7, 1.3],
        "rotation_range": [-30, 30],
        "translation_range": 0.15,
        "time_stretch_range": [0.7, 1.3],
        "dropout_prob": 0.1,
        "swap_hands_prob": 0.4,
        "mirror_prob": 0.5,
        "random_start_prob": 0.4,
        "speed_variation_prob": 0.5,
        "random_frame_drop_prob": 0.3,
        "gaussian_noise_std": 0.02,
        # New augmenter parameters
        "hand_jitter_prob": 0.7,
        "adaptive_resampling_prob": 0.5,
        "sequential_consistency_prob": 0.4
    },
    
    "use_asl_citizen": True,
    "asl_citizen_path": "data/ASL_Citizen",
    "asl_citizen_target_glosses": ["book", "drink", "computer", "before", "chair", 
                                   "go", "clothes", "who", "candy", "cousin"],
    
    # Model configuration
    "model_type": "bilstm",
    "use_enhanced_model": True,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 3,
    "dropout": 0.4,  # Reduced from 0.5 to 0.4
    "temporal_dropout_prob": 0.1,
    "seq_length": 150,
    "early_stop_patience": 25,
    
    # Enhanced model parameters
    "multi_resolution": True,
    "use_temporal_conv": True,
    "temporal_conv_kernel_sizes": [3, 5, 7],
    "use_gated_residual": True,
    "use_cross_resolution_attention": True,
    "feature_dropout_prob": 0.2,  # Half of the main dropout (0.4)
    
    # Loss function parameters
    "use_focal_loss": True,  # Enable focal loss
    "focal_gamma": 2.0,      # Focusing parameter for focal loss

    # Temperature scaling parameters
    "use_temperature_scaling": True,
    "initial_temperature": 1.5,   # Higher values = softer predictions
    "final_temperature": 1.0,     # Gradually reduce to standard temperature
    
    # Dynamic confidence penalty parameters
    "use_confidence_penalty": True,
    "init_penalty_weight": 0.05,  # Initial confidence penalty weight
    "final_penalty_weight": 0.2,  # Final confidence penalty weight after training
    
    # Regularization parameters
    "weight_decay": 1e-4,
    "l2_lambda": 0.0001,
    "l2_excluded_layers": ["lstm"],
    
    # Learning rate scheduler parameters
    "scheduler_pct_start": 0.3,    # Warm-up for 30% of training
    "scheduler_div_factor": 25.0,  # Start with lr/25
    "scheduler_final_div_factor": 1000.0  # End with lr/1000
}