import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import data processing utilities
from src.config import config
from src.data_processing.dataset import *
from src.data_processing.dataset_asl import *
from src.data_processing.augmenter import apply_data_augmentation
from src.data_processing.sequence_augmenter import *

# Import models
from src.sequence_models.transformer import ASLTransformerModel
from src.sequence_models.bi_lstm_model import (
    MultiResolutionBiLSTMAttentionModel
)
from src.sequence_models.enhanced_bi_lstm_model import MultiResolutionBiLSTMAttentionModelEnhanced
from src.sequence_models.multi_temporal_cnn import MultiStageTemporalCNNModel

# Import training and losses
from src.losses import get_loss_function
from src.train_save.train import train_sequence_model
from src.train_save.save import save_results

# Import optimizer for hyperparameter tuning
from src.hyperparameter_tuning import optimize_hyperparameters, set_seed

def load_and_prepare_data(config):
    """Load and prepare data according to config"""
    # Set random seeds for reproducibility
    set_seed(42)

    # Step 1: Load the WLASL dataset
    print("Step 1: Loading WLASL dataset...")
    wlasl_data = download_wlasl_data(config["data_path"], download=True)
    if wlasl_data is None:
        print("Failed to load dataset. Exiting.")
        return None, None, None

    # Step 2a: Process the WLASL dataset
    print(f"Step 2a: Processing WLASL dataset with {config['num_classes']} classes...")
    wlasl_features, wlasl_labels = integrated_prepare_sequence_dataset(
        data_path=config["data_path"],
        num_classes=config["num_classes"],
        min_videos_per_class=1,
    )
    features = wlasl_features
    labels = wlasl_labels

    asl_citizen_features = []
    asl_citizen_labels = []
    
    # Step 2b: Process or load ASL Citizen dataset if enabled
    if config["use_asl_citizen"]:
        print("Step 2b: Checking for ASL Citizen dataset...")
        asl_citizen_path = config["asl_citizen_path"]
        preprocessed_path = os.path.join(asl_citizen_path, "preprocessed")
        
        # Check if preprocessed data already exists
        features_path = os.path.join(preprocessed_path, "asl_citizen_features.json")
        gloss_map_path = os.path.join(preprocessed_path, "asl_citizen_gloss_to_videos.json")
        
        if os.path.exists(features_path) and os.path.exists(gloss_map_path):
            print("Found preprocessed ASL Citizen data, loading it...")
            asl_citizen_features, asl_citizen_labels = load_preprocessed_asl_citizen(
                asl_citizen_path=asl_citizen_path,
                wlasl_data=wlasl_data,
                num_classes=config["num_classes"]
            )
            
            if len(asl_citizen_features) > 0:
                print(f"Loaded {len(asl_citizen_features)} preprocessed ASL Citizen sequences")
            else:
                print("No valid ASL Citizen data could be loaded. Using only WLASL dataset.")
        else:
            print("Preprocessed ASL Citizen data not found. Using only WLASL dataset.")
    
    # Combine datasets if ASL Citizen features were loaded
    if len(asl_citizen_features) > 0:
        features, labels = combine_datasets(
            wlasl_features, wlasl_labels,
            asl_citizen_features, asl_citizen_labels
        )
        print(f"Combined dataset has {len(features)} videos")
    else:
        print("No ASL Citizen features were loaded. Using only WLASL dataset.")

    # Get updated number of classes from actual labels
    unique_labels = set(labels)
    config["num_classes"] = len(unique_labels)
    print(f"Final dataset has {config['num_classes']} classes with {len(features)} total videos")

    # Get class names if available
    try:
        import json
        with open(os.path.join(config["data_path"], "WLASL_v0.3.json"), 'r') as f:
            wlasl_json = json.load(f)
        
        # Extract class names based on the unique labels
        unique_labels = sorted(list(unique_labels))
        class_names = []
        
        # First, create a mapping of indices to glosses
        gloss_mapping = {}
        for i, entry in enumerate(wlasl_json):
            gloss_mapping[i] = entry['gloss']
        
        # Now map the class labels to glosses
        for label in unique_labels:
            if label < len(gloss_mapping):
                class_names.append(gloss_mapping[label])
            else:
                # If we couldn't find a name, use a placeholder
                class_names.append(f"Class_{label}")
        
        config["class_names"] = class_names
        print(f"Found class names: {class_names}")
    except Exception as e:
        print(f"Could not extract class names: {e}")
        # Create generic class names
        config["class_names"] = [f"Class_{i}" for i in range(config["num_classes"])]
    
    return wlasl_data, features, labels


def train_model_with_config(features, labels, config):
    """Train model with given configuration"""
    # Create results directory
    os.makedirs(f"results/{config['model_type']}", exist_ok=True)
    
    # Step 3: Prepare DataLoaders for sequence data with augmentation
    print("Step 3: Creating DataLoaders with stratified split and augmentation...")
    train_loader, val_loader, test_loader, input_dim = prepare_stratified_sequential_dataset(
        features, 
        labels, 
        seq_length=config["seq_length"], 
        batch_size=config["batch_size"],
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        use_augmentation=config["use_data_augmentation"],
        augmentation_params=config.get("augmentation_params", None),
        augmentation_factor=config.get("augmentation_factor", 3),
    )
    
    # Update config with input dimension
    config["input_dim"] = input_dim
    
    # Step 4: Initialize model based on configuration
    print(f"Step 4: Initializing {config['model_type']} model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if config["model_type"] == "transformer":
        model = ASLTransformerModel(
            input_dim=input_dim,
            num_classes=config["num_classes"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        ).to(device)
    elif config["model_type"] == "bilstm":
        # Check if we should use the enhanced model
        if config.get("use_enhanced_model", False):
            print("Using Enhanced MultiResolutionBiLSTM model...")
            model = MultiResolutionBiLSTMAttentionModelEnhanced(
                input_dim=input_dim,
                num_classes=config["num_classes"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                temporal_dropout_prob=config.get("temporal_dropout_prob", 0.1),
                num_heads=config.get("num_heads", 4)
            ).to(device)
        else:
            print("Using standard MultiResolutionBiLSTM model...")
            model = MultiResolutionBiLSTMAttentionModel(
                input_dim=input_dim,
                num_classes=config["num_classes"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                temporal_dropout_prob=config.get("temporal_dropout_prob", 0.1)
            ).to(device)
    elif config["model_type"] == "temporalcnn":
        model = MultiStageTemporalCNNModel(
            input_dim=input_dim,
            num_classes=config["num_classes"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"]
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Step 5: Setup training
    print("Step 5: Setting up training...")
    
    # Calculate class weights for balanced loss
    class_counts = np.bincount(labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)

    # Create loss function using our factory function
    criterion = get_loss_function(config, class_weights)

    # Get weight decay from config or use default
    weight_decay = config.get("weight_decay", 1e-5)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        epochs=config["num_epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Warm-up for 30% of training
        div_factor=25.0,  # Start with lr/25
        final_div_factor=1000.0  # End with lr/1000
    )
    
    # Step 6: Train model
    print("Step 6: Training model...")
    model, train_losses, val_losses, val_accuracies, test_accuracies, val_precisions, \
    val_recalls, val_f1s, test_precisions, test_recalls, test_f1s = train_sequence_model(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler, config["num_epochs"], device,
        early_stop_patience=config['early_stop_patience'],
        improvement_threshold=0.001,
        monitor_metric='accuracy',
        l2_lambda=config.get("l2_lambda", 0.0001),
        l2_excluded_layers=config.get("l2_excluded_layers", ('lstm',))
    )
    
    # Step 7: Save results
    print("Step 7: Saving results...")
    result_dir = os.path.join("results", config["model_type"])
    
    # Run one final evaluation to get predictions for confusion matrix
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
    
    # Save all results using the existing save_results function
    saved_paths = save_results(
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
        scaler=None,  # No scaler for sequence models as we normalize per sequence
        config=config,
        y_true=y_true,
        y_pred=y_pred,
        base_dir=result_dir
    )
    
    print(f"Training complete. Results saved to {result_dir}")
    return model, config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ASL Recognition Training & Tuning')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'tune', 'eval'], 
                        default='train', help='Mode: train, tune or evaluate')
    
    # Model configuration
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
    
    # Training and evaluation options
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--config', type=str, help='Path to custom configuration JSON file')
    parser.add_argument('--save-dir', type=str, default='results', 
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load custom config or use default
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        print(f"Loaded custom configuration from {args.config}")
    else:
        custom_config = config.copy()
    
    # Update config with command line arguments
    custom_config["model_type"] = args.model_type
    custom_config["batch_size"] = args.batch_size
    custom_config["num_epochs"] = args.epochs
    custom_config["learning_rate"] = args.learning_rate
    
    # Set enhanced model flag if specified
    if args.enhanced and args.model_type == 'bilstm':
        custom_config["use_enhanced_model"] = True
        print("Using enhanced BiLSTM model")
    
    # Load and prepare data
    wlasl_data, features, labels = load_and_prepare_data(custom_config)
    if features is None or labels is None:
        print("Failed to load data. Exiting.")
        return 1
    
    # Execute according to mode
    if args.mode == 'train':
        print(f"Training {args.model_type} model with batch size {args.batch_size} for {args.epochs} epochs")
        model, final_config = train_model_with_config(features, labels, custom_config)
        print("Training completed successfully")
        
    elif args.mode == 'tune':
        print(f"Tuning hyperparameters for {args.model_type} model with {args.trials} trials")
        
        # Important: Make sure the enhanced flag is set in the config before passing to optimizer
        if args.enhanced and args.model_type == 'bilstm':
            custom_config["use_enhanced_model"] = True
            print("Using enhanced BiLSTM model for hyperparameter tuning")
        
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
        
        print("Hyperparameter tuning completed")
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Update config with best parameters
        custom_config.update(best_params)
        
        # Train final model with best parameters
        print("Training final model with best parameters...")
        model, final_config = train_model_with_config(features, labels, custom_config)
        
        # Save final config
        with open(os.path.join(args.save_dir, "best_config.json"), "w") as f:
            json.dump(final_config, f, indent=4)
        
        print(f"Final model training complete. Best configuration saved to {args.save_dir}/best_config.json")
    
    elif args.mode == 'eval':
        # TODO: Implement evaluation mode for pre-trained models
        print("Evaluation mode not yet implemented")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())