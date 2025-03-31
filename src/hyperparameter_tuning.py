import os
import sys
import argparse
import json
import torch
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
from datetime import datetime
import random

# Add the current directory to the path to make imports work
sys.path.insert(0, os.getcwd())

# Import your existing modules - fixed imports
from src.config import config
from src.losses import get_loss_function, DynamicFocalConfidenceLoss
from src.data_processing.dataset import prepare_stratified_sequential_dataset
from src.train_save.train import train_sequence_model
from src.train_save.save import save_results

# Import model classes
from src.sequence_models.transformer import ASLTransformerModel
from src.sequence_models.bi_lstm_model import MultiResolutionBiLSTMAttentionModel
from src.sequence_models.enhanced_bi_lstm_model import MultiResolutionBiLSTMAttentionModelEnhanced
from src.sequence_models.multi_temporal_cnn import MultiStageTemporalCNNModel

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def objective(trial, features, labels, base_config=None):
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        features: Feature data
        labels: Label data
        base_config: Base configuration dictionary
        
    Returns:
        Best validation accuracy achieved
    """
    # Create a trial-specific config by updating the base config
    if base_config is None:
        trial_config = config.copy()
    else:
        trial_config = base_config.copy()
    
    # Create a unique trial ID
    trial_id = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ====== COMMON HYPERPARAMETERS ======
    # Learning rate - log uniform scale
    trial_config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    
    # Training parameters
    trial_config["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 96, 128])
    trial_config["dropout"] = trial.suggest_float("dropout", 0.2, 0.6)
    trial_config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Architecture parameters
    trial_config["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 192, 256, 320, 384])
    trial_config["num_layers"] = trial.suggest_int("num_layers", 1, 4)
    
    # Data augmentation parameters
    use_augmentation = trial.suggest_categorical("use_data_augmentation", [True, False])
    trial_config["use_data_augmentation"] = use_augmentation
    
    if use_augmentation:
        augmentation_factor = trial.suggest_int("augmentation_factor", 1, 5)
        trial_config["augmentation_factor"] = augmentation_factor
        
        # Augmentation strength parameters
        aug_params = trial_config.get("augmentation_params", {})
        aug_params["jitter_range"] = trial.suggest_float("jitter_range", 0.01, 0.05)
        aug_params["scale_range"] = [
            trial.suggest_float("scale_min", 0.6, 0.9),
            trial.suggest_float("scale_max", 1.1, 1.4)
        ]
        aug_params["rotation_range"] = [
            -trial.suggest_int("rotation_angle", 10, 45),
            trial.suggest_int("rotation_angle", 10, 45)
        ]
        aug_params["translation_range"] = trial.suggest_float("translation_range", 0.05, 0.25)
        aug_params["dropout_prob"] = trial.suggest_float("aug_dropout_prob", 0.05, 0.2)
        aug_params["gaussian_noise_std"] = trial.suggest_float("gaussian_noise_std", 0.01, 0.05)
        
        trial_config["augmentation_params"] = aug_params
    
    # ====== MODEL SPECIFIC HYPERPARAMETERS ======
    if trial_config["model_type"] == "bilstm":
        # BiLSTM specific parameters
        trial_config["temporal_dropout_prob"] = trial.suggest_float("temporal_dropout_prob", 0.05, 0.2)
        
        # Enhanced model parameters
        if trial_config.get("use_enhanced_model", False):
            # For enhanced model, ensure num_heads is compatible with hidden_dim
            # BiLSTM output has 2x hidden_dim due to bidirectional
            bidirectional_dim = trial_config["hidden_dim"] * 2
            # Find valid number of heads that divide bidirectional_dim evenly
            valid_heads = [h for h in [2, 4, 8] if bidirectional_dim % h == 0]
            if not valid_heads:  # If no valid heads, use 2 and adjust hidden_dim
                print(f"Warning: No valid number of heads for hidden_dim {trial_config['hidden_dim']}. Adjusting hidden_dim.")
                # Find the nearest hidden_dim that works with 2 heads
                nearest_valid_hidden = (bidirectional_dim // 2) * 2 // 2
                trial_config["hidden_dim"] = nearest_valid_hidden
                valid_heads = [2, 4] if nearest_valid_hidden % 2 == 0 else [2]
            
            trial_config["num_heads"] = trial.suggest_categorical("num_heads", valid_heads)
            trial_config["feature_dropout_prob"] = trial.suggest_float("feature_dropout_prob", 0.1, 0.3)
            
            # Multi-resolution parameters
            trial_config["multi_resolution"] = True
            trial_config["use_temporal_conv"] = trial.suggest_categorical("use_temporal_conv", [True, False])
            
            if trial_config["use_temporal_conv"]:
                kernel_sizes = []
                if trial.suggest_categorical("use_small_kernel", [True, False]):
                    kernel_sizes.append(3)
                if trial.suggest_categorical("use_medium_kernel", [True, False]):
                    kernel_sizes.append(5)
                if trial.suggest_categorical("use_large_kernel", [True, False]):
                    kernel_sizes.append(7)
                
                # Ensure at least one kernel size is selected
                if len(kernel_sizes) == 0:
                    kernel_sizes = [3]
                
                trial_config["temporal_conv_kernel_sizes"] = kernel_sizes
            
            trial_config["use_gated_residual"] = trial.suggest_categorical("use_gated_residual", [True, False])
            trial_config["use_cross_resolution_attention"] = trial.suggest_categorical("use_cross_resolution_attention", [True, False])
            
    elif trial_config["model_type"] == "transformer":
        # Transformer specific parameters
        # Ensure num_heads divides hidden_dim
        valid_transformer_heads = [h for h in [2, 4, 8] if trial_config["hidden_dim"] % h == 0]
        if not valid_transformer_heads:
            # Adjust to nearest valid hidden_dim
            nearest_valid = (trial_config["hidden_dim"] // 8) * 8
            if nearest_valid == 0:
                nearest_valid = 8
            trial_config["hidden_dim"] = nearest_valid
            valid_transformer_heads = [2, 4, 8] if nearest_valid % 8 == 0 else ([2, 4] if nearest_valid % 4 == 0 else [2])
        
        trial_config["num_heads"] = trial.suggest_categorical("num_heads", valid_transformer_heads)
        trial_config["dim_feedforward"] = trial.suggest_int("dim_feedforward", 512, 2048, step=256)
        trial_config["positional_encoding"] = trial.suggest_categorical("positional_encoding", ["fixed", "learned"])
        
    elif trial_config["model_type"] == "temporalcnn":
        # TemporalCNN specific parameters
        trial_config["num_blocks"] = trial.suggest_int("num_blocks", 2, 5)
        trial_config["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
        trial_config["use_residual"] = trial.suggest_categorical("use_residual", [True, False])
    
    # ====== LOSS FUNCTION AND REGULARIZATION ======
    # Loss function
    trial_config["use_focal_loss"] = trial.suggest_categorical("use_focal_loss", [True, False])
    
    if trial_config["use_focal_loss"]:
        trial_config["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)
    
    trial_config["use_confidence_penalty"] = trial.suggest_categorical("use_confidence_penalty", [True, False])
    
    if trial_config["use_confidence_penalty"]:
        trial_config["init_penalty_weight"] = trial.suggest_float("init_penalty_weight", 0.01, 0.1)
        trial_config["final_penalty_weight"] = trial.suggest_float("final_penalty_weight", 0.1, 0.3)
    
    # Temperature scaling
    trial_config["use_temperature_scaling"] = trial.suggest_categorical("use_temperature_scaling", [True, False])
    
    if trial_config["use_temperature_scaling"]:
        trial_config["initial_temperature"] = trial.suggest_float("initial_temperature", 1.0, 2.0)
        trial_config["final_temperature"] = trial.suggest_float("final_temperature", 0.5, 1.0)
    
    # Regularization
    trial_config["l2_lambda"] = trial.suggest_float("l2_lambda", 1e-6, 1e-3, log=True)
    
    # Early stopping
    trial_config["early_stop_patience"] = trial.suggest_int("early_stop_patience", 10, 30)
    
    # ====== DATA LOADERS ======
    print(f"Trial {trial.number}: Creating DataLoaders...")
    train_loader, val_loader, test_loader, input_dim = prepare_stratified_sequential_dataset(
        features, 
        labels, 
        seq_length=trial_config["seq_length"], 
        batch_size=trial_config["batch_size"],
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        use_augmentation=trial_config["use_data_augmentation"],
        augmentation_params=trial_config.get("augmentation_params", None),
        augmentation_factor=trial_config.get("augmentation_factor", 3),
    )
    
    # Update input dimension
    trial_config["input_dim"] = input_dim
    
    # ====== MODEL INITIALIZATION ======
    print(f"Trial {trial.number}: Initializing {trial_config['model_type']} model...")
    
    if trial_config["model_type"] == "bilstm":
        if trial_config.get("use_enhanced_model", False):
            model = MultiResolutionBiLSTMAttentionModelEnhanced(
                input_dim=input_dim,
                num_classes=trial_config["num_classes"],
                hidden_dim=trial_config["hidden_dim"],
                num_layers=trial_config["num_layers"],
                dropout=trial_config["dropout"],
                temporal_dropout_prob=trial_config["temporal_dropout_prob"],
                num_heads=trial_config.get("num_heads", 4)
            ).to(device)
        else:
            model = MultiResolutionBiLSTMAttentionModel(
                input_dim=input_dim,
                num_classes=trial_config["num_classes"],
                hidden_dim=trial_config["hidden_dim"],
                num_layers=trial_config["num_layers"],
                dropout=trial_config["dropout"],
                temporal_dropout_prob=trial_config.get("temporal_dropout_prob", 0.1)
            ).to(device)
    elif trial_config["model_type"] == "transformer":
        model = ASLTransformerModel(
            input_dim=input_dim,
            num_classes=trial_config["num_classes"],
            hidden_dim=trial_config["hidden_dim"],
            num_heads=trial_config["num_heads"],
            num_layers=trial_config["num_layers"],
            dropout=trial_config["dropout"]
        ).to(device)
    elif trial_config["model_type"] == "temporalcnn":
        model = MultiStageTemporalCNNModel(
            input_dim=input_dim,
            num_classes=trial_config["num_classes"],
            hidden_dim=trial_config["hidden_dim"],
            dropout=trial_config["dropout"]
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {trial_config['model_type']}")
    
    # ====== LOSS AND OPTIMIZER ======
    # Calculate class weights for balanced loss
    class_counts = np.bincount(labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)
    
    # Create loss function using our factory function
    criterion = get_loss_function(trial_config, class_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=trial_config["learning_rate"], 
        weight_decay=trial_config["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=trial_config["learning_rate"],
        epochs=trial_config["num_epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=trial_config.get("scheduler_pct_start", 0.3),
        div_factor=trial_config.get("scheduler_div_factor", 25.0),
        final_div_factor=trial_config.get("scheduler_final_div_factor", 1000.0)
    )
    
    # ====== TRAINING ======
    print(f"Trial {trial.number}: Starting training...")
    try:
        model, train_losses, val_losses, val_accuracies, test_accuracies, val_precisions, \
        val_recalls, val_f1s, test_precisions, test_recalls, test_f1s = train_sequence_model(
            model, train_loader, val_loader, test_loader,
            criterion, optimizer, scheduler, 
            num_epochs=trial_config["num_epochs"], 
            device=device,
            early_stop_patience=trial_config['early_stop_patience'],
            improvement_threshold=0.001,
            monitor_metric='accuracy',
            l2_lambda=trial_config.get("l2_lambda", 0.0001),
            l2_excluded_layers=trial_config.get("l2_excluded_layers", ('lstm',))
        )
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return a very low score to indicate failure
        return float('-inf')
    
    # Report intermediate values for pruning
    for epoch, val_acc in enumerate(val_accuracies):
        trial.report(val_acc, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Get best validation accuracy
    best_epoch = np.argmax(val_accuracies)
    best_val_accuracy = val_accuracies[best_epoch]
    
    # Get corresponding test accuracy
    if best_epoch < len(test_accuracies):
        test_accuracy = test_accuracies[best_epoch]
    else:
        test_accuracy = test_accuracies[-1]
    
    # ====== SAVE RESULTS ======
    # Create directories for this trial
    trial_dir = os.path.join("results", trial_config["model_type"], "optuna_trials", trial_id)
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save the trial config
    with open(os.path.join(trial_dir, "config.json"), "w") as f:
        json.dump(trial_config, f, indent=4)
    
    # Run final evaluation to get predictions for confusion matrix
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for features, masks, labels in test_loader:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            outputs = model(features, masks)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Save trial results using the existing save_results function
    save_results(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        test_accuracies=test_accuracies,
        val_precisions=val_precisions,
        val_recalls=val_recalls,
        val_f1s=val_f1s,
        test_precisions=test_precisions,
        test_recalls=test_recalls,
        test_f1s=test_f1s,
        scaler=None,
        config=trial_config,
        y_true=y_true,
        y_pred=y_pred,
        base_dir=trial_dir
    )
    
    print(f"Trial {trial.number} completed with best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Corresponding test accuracy: {test_accuracy:.4f}")
    
    return best_val_accuracy


def optimize_hyperparameters(features, labels, base_config=None, num_trials=30, 
                             study_name=None, storage_path=None):
    """
    Main entry point for hyperparameter optimization
    
    Args:
        features: Feature data
        labels: Label data
        base_config: Optional base configuration
        num_trials: Number of trials to run
        study_name: Optional name for resumable study
        storage_path: Optional storage path for resumable study
    
    Returns:
        Best parameters and study object
    """
    # Create results directory
    os.makedirs("results/optuna_study", exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Use more sophisticated pruner and sampler
    sampler = TPESampler(
        seed=42,
        n_startup_trials=5,  # Number of random trials before using TPE
        multivariate=True    # Consider parameter correlations
    )
    
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=15,
        interval_steps=1,
        n_min_trials=2
    )
    
    if study_name and storage_path:
        # Create or load resumable study
        print(f"Creating or loading study '{study_name}' from {storage_path}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
    else:
        # Create new study
        print("Creating new study")
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name="asl_recognition_optimization"
        )
    
    # Run the optimization
    study.optimize(
        lambda trial: objective(trial, features, labels, base_config),
        n_trials=num_trials,
        timeout=None,  # No timeout
        n_jobs=1,      # Run sequentially
        gc_after_trial=True  # Clean up memory after each trial
    )
    
    # Get the best trial
    best_trial = study.best_trial
    
    # Print best parameters
    print("\n" + "="*50)
    print("Best trial:")
    print(f"  Value: {best_trial.value:.4f} (validation accuracy)")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the study results
    study_results_dir = "results/optuna_study"
    study_model_dir = os.path.join(study_results_dir, base_config["model_type"] if base_config else "model")
    os.makedirs(study_model_dir, exist_ok=True)
    
    # Save best parameters
    with open(os.path.join(study_model_dir, "best_params.json"), "w") as f:
        json.dump(best_trial.params, f, indent=4)
    
    # Save all trial information
    try:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(study_model_dir, "trials.csv"), index=False)
        
        # Generate and save plots
        try:
            # Optimization history
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(os.path.join(study_model_dir, "optimization_history.png"))
            plt.close()
            
            # Parameter importances
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(os.path.join(study_model_dir, "param_importances.png"))
            plt.close()
            
            # Parallel coordinate plot for top 10 trials
            plt.figure(figsize=(15, 10))
            optuna.visualization.matplotlib.plot_parallel_coordinate(
                study, 
                params=list(best_trial.params.keys())[:10]  # Use top 10 parameters
            )
            plt.tight_layout()
            plt.savefig(os.path.join(study_model_dir, "parallel_coordinate.png"))
            plt.close()
            
            # Slice plot for important parameters
            param_importances = optuna.importance.get_param_importances(study)
            top_params = list(param_importances.keys())[:min(5, len(param_importances))]
            
            if len(top_params) >= 2:
                plt.figure(figsize=(12, 10))
                optuna.visualization.matplotlib.plot_contour(study, params=top_params[:2])
                plt.tight_layout()
                plt.savefig(os.path.join(study_model_dir, "contour_plot.png"))
                plt.close()
            
        except Exception as e:
            print(f"Error generating plots: {e}")
    except Exception as e:
        print(f"Error saving trial data: {e}")
    
    return best_trial.params, study


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ASL Recognition Hyperparameter Tuning')
    
    # Model selection
    parser.add_argument('--model-type', type=str, 
                        choices=['transformer', 'bilstm', 'temporalcnn'],
                        default='bilstm', help='Model type to use')
    parser.add_argument('--enhanced', action='store_true', 
                        help='Use enhanced BiLSTM model (only for bilstm type)')
    
    # Hyperparameter tuning options
    parser.add_argument('--trials', type=int, default=30, 
                        help='Number of trials for hyperparameter tuning')
    parser.add_argument('--resume-study', action='store_true',
                        help='Resume previous hyperparameter tuning study')
    parser.add_argument('--study-name', type=str, default='asl_recognition_study',
                        help='Name for the Optuna study')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db',
                        help='Storage path for Optuna database')
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Path to custom configuration JSON file')
    parser.add_argument('--save-dir', type=str, default='results', 
                        help='Directory to save results')
    
    # Add data loading options
    parser.add_argument('--data-path', type=str, default='data/wlasl_data',
                        help='Path to WLASL dataset')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes to use')
    parser.add_argument('--use-asl-citizen', action='store_true',
                        help='Use ASL Citizen dataset')
    
    args = parser.parse_args()
    
    # Load custom config or use default
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        print(f"Loaded custom configuration from {args.config}")
    else:
        # Import the default config
        from src.config import config
        custom_config = config.copy()
    
    # Update config with command line arguments
    custom_config["model_type"] = args.model_type
    custom_config["data_path"] = args.data_path
    custom_config["num_classes"] = args.num_classes
    custom_config["use_asl_citizen"] = args.use_asl_citizen
    
    # Set enhanced model flag if specified
    if args.enhanced and args.model_type == 'bilstm':
        custom_config["use_enhanced_model"] = True
        print("Using enhanced BiLSTM model")
    
    # Import necessary modules for data loading
    from main import load_and_prepare_data
    
    # Load and prepare data
    print("Loading and preparing data...")
    data, features, labels = load_and_prepare_data(custom_config)
    
    if features is None or labels is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    print(f"Loaded data with {len(features)} samples and {custom_config['num_classes']} classes")
    
    # Run hyperparameter optimization
    print(f"Starting hyperparameter optimization with {args.trials} trials")
    
    if args.resume_study:
        print(f"Resuming study '{args.study_name}' from {args.storage}")
        best_params, study = optimize_hyperparameters(
            features, labels, 
            base_config=custom_config,
            num_trials=args.trials,
            study_name=args.study_name,
            storage_path=args.storage
        )
    else:
        best_params, study = optimize_hyperparameters(
            features, labels, 
            base_config=custom_config,
            num_trials=args.trials
        )
    
    # Update config with best parameters
    final_config = custom_config.copy()
    final_config.update(best_params)
    
    # Save final config
    with open(os.path.join(args.save_dir, "best_config.json"), "w") as f:
        json.dump(final_config, f, indent=4)
    
    print(f"Hyperparameter optimization completed. Best configuration saved to {args.save_dir}/best_config.json")
    
    # Train final model with best parameters (optional)
    train_final = input("Do you want to train a final model with the best parameters? (y/n): ")
    
    if train_final.lower() == 'y':
        print("Training final model with best parameters...")
        from main import train_model_with_config
        
        # Train the final model
        model, _ = train_model_with_config(features, labels, final_config)
        print("Final model training complete.")
    
    print("Hyperparameter tuning completed successfully!")