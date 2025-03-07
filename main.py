import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.cnn_model import GestureCNN
from src.svm_model import GestureSVM
from src.dataset import integrated_prepare_dataset, GestureDataset, download_wlasl_data
from src.train import train_model, train_svm_model

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def save_results(model, train_losses, val_losses, val_accuracies, test_accuracies, scaler, config, base_dir="results", grid_search_results=None, y_true=None, y_pred=None):
    """
    Save all results to appropriate directories based on model type.
    
    Args:
        model: The trained model (PyTorch model or SVM wrapper)
        train_losses: List of training losses
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies 
        test_accuracies: List of test accuracies
        scaler: The feature scaler used for preprocessing
        config: Configuration dictionary
        base_dir: Base directory for results
        grid_search_results: Optional GridSearchCV results for SVM plotting
        y_true: Ground truth labels for confusion matrix
        y_pred: Predicted labels for confusion matrix
    
    Returns:
        dict: Paths to saved files
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import json
    from datetime import datetime
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    # Identify model type
    is_torch_model = hasattr(model, 'state_dict') and callable(getattr(model, 'state_dict'))
    is_svm_model = hasattr(model, 'pipeline') and hasattr(model, 'is_trained')
    
    # Get timestamp for unique folder naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine model subfolder name
    if is_svm_model:
        model_type = "svm"
    elif is_torch_model:
        model_type = "cnn"
    else:
        model_type = "unknown"
    
    # Create unique result directory with timestamp
    result_dir = os.path.join(base_dir, model_type, timestamp)
    models_dir = os.path.join(result_dir, "models")
    plots_dir = os.path.join(result_dir, "plots")
    logs_dir = os.path.join(result_dir, "logs")
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    saved_paths = {}
    
    # 1. Save training progress plots
    if is_svm_model and grid_search_results is not None:
        # For SVM with GridSearchCV results, create different plots
        plt.figure(figsize=(15, 10))
        
        # 1.1 Create hyperparameter heatmap for SVM if we have C and gamma parameters
        if 'param_svm__C' in grid_search_results.cv_results_ and 'param_svm__gamma' in grid_search_results.cv_results_:
            plt.subplot(2, 2, 1)
            
            # Extract unique parameter values
            c_values = sorted(list(set(grid_search_results.cv_results_['param_svm__C'])))
            
            # Handle mixed type gamma values by keeping them as strings for sorting
            gamma_values = list(set(grid_search_results.cv_results_['param_svm__gamma']))
            # Separate string and float values
            str_gammas = [g for g in gamma_values if isinstance(g, str)]
            float_gammas = [g for g in gamma_values if isinstance(g, (int, float))]
            # Sort each group separately
            str_gammas.sort()
            float_gammas.sort()
            # Combine them with strings first, then floats
            gamma_values = str_gammas + float_gammas
            
            # Create results matrix
            results_matrix = np.zeros((len(gamma_values), len(c_values)))
            
            # Fill the matrix
            # FIX: Iterate over indices and access by index instead of using strings as indices
            for i in range(len(grid_search_results.cv_results_['param_svm__C'])):
                c_idx = c_values.index(grid_search_results.cv_results_['param_svm__C'][i])
                gamma_idx = gamma_values.index(grid_search_results.cv_results_['param_svm__gamma'][i])
                results_matrix[gamma_idx, c_idx] = grid_search_results.cv_results_['mean_test_score'][i] * 100
            
            # Create heatmap
            im = plt.imshow(results_matrix, interpolation='nearest', cmap='viridis')
            plt.colorbar(im, label='Validation Accuracy (%)')
            
            # Set labels
            plt.xticks(range(len(c_values)), [str(c) for c in c_values])
            plt.yticks(range(len(gamma_values)), [str(g) for g in gamma_values])
            plt.xlabel('C parameter')
            plt.ylabel('gamma parameter')
            plt.title('SVM Parameter Search Results')
            
        # 1.2 Plot C parameter validation curve if available
        if 'param_svm__C' in grid_search_results.cv_results_:
            plt.subplot(2, 2, 2)
            
            # Get unique C values and corresponding mean scores
            c_values = sorted(list(set(grid_search_results.cv_results_['param_svm__C'])))
            mean_scores = []
            std_scores = []
            
            # Get mean score for each C value
            for c in c_values:
                # FIX: Create masks for filtering arrays instead of using dictionary access
                c_mask = grid_search_results.cv_results_['param_svm__C'] == c
                c_scores = grid_search_results.cv_results_['mean_test_score'][c_mask] * 100
                c_std = grid_search_results.cv_results_['std_test_score'][c_mask] * 100
                
                mean_scores.append(np.mean(c_scores))
                std_scores.append(np.mean(c_std))
            
            # Plot validation curve
            plt.semilogx(c_values, mean_scores, 'o-', label='Validation Accuracy')
            plt.fill_between(c_values, 
                             [m - s for m, s in zip(mean_scores, std_scores)],
                             [m + s for m, s in zip(mean_scores, std_scores)],
                             alpha=0.3)
            plt.xlabel('C Parameter (log scale)')
            plt.ylabel('Accuracy (%)')
            plt.title('SVM Validation Curve for C Parameter')
            plt.grid(True)
            plt.legend()
            
        # 1.3 Show best parameters and test accuracy
        plt.subplot(2, 2, 3)
        plt.axis('off')
        best_text = (
            f"Best Parameters:\n"
            f"C: {grid_search_results.best_params_.get('svm__C', 'N/A')}\n"
            f"gamma: {grid_search_results.best_params_.get('svm__gamma', 'N/A')}\n"
            f"kernel: {grid_search_results.best_params_.get('svm__kernel', 'rbf')}\n\n"
            f"Best CV Score: {grid_search_results.best_score_ * 100:.2f}%\n"
            f"Test Accuracy: {test_accuracies[-1] if test_accuracies else 'N/A':.2f}%"
        )
        plt.text(0.5, 0.5, best_text, horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title("SVM Best Results")
        
        # 1.4 Create bar chart of different metrics
        plt.subplot(2, 2, 4)
        x = ['Train', 'Validation', 'Test']
        y = [
            100.0,  # SVM train accuracy is usually 100%
            grid_search_results.best_score_ * 100,
            test_accuracies[-1] if test_accuracies else 0
        ]
        plt.bar(x, y)
        plt.ylim([0, 105])
        plt.title('Model Performance')
        plt.ylabel('Accuracy (%)')
        
        for i, v in enumerate(y):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plots_path = os.path.join(plots_dir, f"{model_type}_parameter_search.png")
        plt.savefig(plots_path)
        plt.close()
        saved_paths['plots'] = plots_path
        
        # Also create the learning curves for SVM if available
        if hasattr(grid_search_results, 'learning_curve_results'):
            plt.figure(figsize=(10, 6))
            train_sizes = grid_search_results.learning_curve_results['train_sizes']
            train_scores = np.mean(grid_search_results.learning_curve_results['train_scores'], axis=1) * 100
            valid_scores = np.mean(grid_search_results.learning_curve_results['test_scores'], axis=1) * 100
            
            plt.plot(train_sizes, train_scores, 'o-', label='Training Accuracy')
            plt.plot(train_sizes, valid_scores, 'o-', label='Validation Accuracy')
            plt.xlabel('Training Examples')
            plt.ylabel('Accuracy (%)')
            plt.title('SVM Learning Curves')
            plt.grid(True)
            plt.legend()
            
            curves_path = os.path.join(plots_dir, f"{model_type}_learning_curves.png")
            plt.savefig(curves_path)
            plt.close()
            saved_paths['learning_curves'] = curves_path
    else:
        # Default plotting for CNN or SVM without grid search
        plt.figure(figsize=(15, 5))
        
        # Losses
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{model_type.upper()} Losses")
        plt.legend()
        
        # Validation Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(val_accuracies)
        plt.title(f"{model_type.upper()} Validation Accuracy")
        
        # Test Accuracy
        plt.subplot(1, 3, 3)
        plt.plot(test_accuracies)
        plt.title(f"{model_type.upper()} Test Accuracy")
        
        plt.tight_layout()
        plots_path = os.path.join(plots_dir, f"{model_type}_training_progress.png")
        plt.savefig(plots_path)
        plt.close()
        saved_paths['plots'] = plots_path
    
    # 2. Generate and save confusion matrix if provided with true and predicted labels
    if y_true is not None and y_pred is not None:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class names if available in config
        class_names = config.get('class_names', [f'Class {i}' for i in range(config['num_classes'])])
        
        # Create confusion matrix display
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names if len(class_names) == config['num_classes'] else None
        )
        
        # Plot with colorbar and labels
        disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
        
        # Adjust layout
        plt.title(f'{model_type.upper()} Confusion Matrix')
        plt.tight_layout()
        
        # Save the figure
        cm_path = os.path.join(plots_dir, f"{model_type}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        saved_paths['confusion_matrix'] = cm_path
        
        # Also save confusion matrix data as CSV
        cm_csv_path = os.path.join(logs_dir, f"{model_type}_confusion_matrix.csv")
        with open(cm_csv_path, 'w') as f:
            # Write header with class names
            if len(class_names) == config['num_classes']:
                f.write("," + ",".join(class_names) + "\n")
            else:
                f.write("," + ",".join([f'Class {i}' for i in range(config['num_classes'])]) + "\n")
            
            # Write confusion matrix rows
            for i, row in enumerate(cm):
                if i < len(class_names):
                    f.write(f"{class_names[i]}," + ",".join(map(str, row)) + "\n")
                else:
                    f.write(f"Class {i}," + ",".join(map(str, row)) + "\n")
                    
        saved_paths['confusion_matrix_csv'] = cm_csv_path
        
        # Calculate and save classification metrics per class
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate precision, recall, and F1-score for each class
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
        
        # Save metrics to CSV
        metrics_per_class_path = os.path.join(logs_dir, f"{model_type}_class_metrics.csv")
        with open(metrics_per_class_path, 'w') as f:
            f.write("class,precision,recall,f1_score,support\n")
            
            for i in range(len(precision)):
                class_name = class_names[i] if i < len(class_names) else f"Class {i}"
                f.write(f"{class_name},{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f},{support[i]}\n")
                
        saved_paths['class_metrics'] = metrics_per_class_path
    
    # 3. Save metrics as CSV
    metrics_df_path = os.path.join(logs_dir, f"{model_type}_metrics.csv")
    with open(metrics_df_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_accuracy,test_accuracy\n")
        
        # Make sure all lists have the same length
        max_epochs = max(len(train_losses), len(val_losses), len(val_accuracies), len(test_accuracies))
        
        for i in range(max_epochs):
            train_loss = train_losses[i] if i < len(train_losses) else ""
            val_loss = val_losses[i] if i < len(val_losses) else ""
            val_acc = val_accuracies[i] if i < len(val_accuracies) else ""
            test_acc = test_accuracies[i] if i < len(test_accuracies) else ""
            
            f.write(f"{i},{train_loss},{val_loss},{val_acc},{test_acc}\n")
    
    saved_paths['metrics_csv'] = metrics_df_path
    
    # 4. Save model with metrics
    metrics = {
        'final_test_accuracy': test_accuracies[-1] if test_accuracies else None,
        'final_val_accuracy': val_accuracies[-1] if val_accuracies else None,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'model_type': model_type
    }
    
    # If we have grid search results, add them to metrics
    if grid_search_results is not None:
        metrics['best_params'] = grid_search_results.best_params_
        metrics['best_cv_score'] = grid_search_results.best_score_ * 100
    
    # Handle different model types
    if is_svm_model:
        # Save SVM model
        model_path = os.path.join(models_dir, f"{model_type}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Also save configuration and metrics separately
        config_path = os.path.join(models_dir, f"{model_type}_config.json")
        with open(config_path, 'w') as f:
            # Convert config to JSON-serializable format if needed
            json_config = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                          for k, v in config.items()}
            json_config.update({'metrics': metrics})
            json.dump(json_config, f, indent=2)
            
    elif is_torch_model:
        # Save PyTorch model
        model_path = os.path.join(models_dir, f"{model_type}_model.pth")
        
        save_data = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics
        }
        
        torch.save(save_data, model_path)
    
    else:
        # Unknown model type, try generic pickling
        model_path = os.path.join(models_dir, f"{model_type}_unknown_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    saved_paths['model'] = model_path
    
    # 5. Save scaler
    scaler_path = os.path.join(models_dir, f"{model_type}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    saved_paths['scaler'] = scaler_path
    
    # 6. Log results
    log_path = os.path.join(logs_dir, f"{model_type}_training_results.txt")
    with open(log_path, 'w') as f:
        f.write(
            f"""
        {model_type.upper()} Model Training completed!
        Final Results:
        - Test Accuracy: {test_accuracies[-1] if test_accuracies else 'N/A':.2f}%
        - Validation Accuracy: {val_accuracies[-1] if val_accuracies else 'N/A':.2f}%
        - Training Loss: {train_losses[-1] if train_losses else 'N/A':.4f}
        
        Configuration:
        {json.dumps(config, indent=2)}
        
        Results saved in: {result_dir}
        """
        )
    
    saved_paths['log'] = log_path
    
    # 7. Create a summary file for easy reference
    summary_path = os.path.join(result_dir, f"{model_type}_summary.json")
    with open(summary_path, 'w') as f:
        summary = {
            'model_type': model_type,
            'timestamp': timestamp,
            'metrics': metrics,
            'config': {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                      for k, v in config.items()},
            'saved_paths': saved_paths
        }
        json.dump(summary, f, indent=2)
    
    saved_paths['summary'] = summary_path
    
    print(f"{model_type.upper()} model results saved to {result_dir}")
    return saved_paths


def main():
    
    # Also set up model-specific directories
    os.makedirs("results/cnn", exist_ok=True)
    os.makedirs("results/svm", exist_ok=True)
    os.makedirs("results/unknown", exist_ok=True)

    # Configuration
    config = {
        "num_classes": 10,
        "min_samples_per_class": 1,
        "max_samples_per_class": 50,
        "batch_size": 16,
        "num_epochs": 200,
        "learning_rate": 0.001,
        "data_path": "data/wlasl_data",
    }

    # Set random seeds for reproducibility
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Download/load dataset
    print("Step 1: Loading WLASL dataset...")
    wlasl_data = download_wlasl_data(config["data_path"], download=True)

    if wlasl_data is None:
        print("Failed to load or download dataset. Exiting.")
        return

    # Prepare dataset
    print(f"Step 2: Preparing dataset with {config['num_classes']} classes...")
    features, labels = integrated_prepare_dataset(
        data_path=config["data_path"],
        num_classes=config["num_classes"],
        min_videos_per_class=config["min_samples_per_class"],
    )

    # Get class names if available
    try:
        import json
        with open(os.path.join(config["data_path"], "WLASL_v0.3.json"), 'r') as f:
            wlasl_json = json.load(f)
        
        # Extract class names based on the selected classes
        unique_labels = sorted(list(set(labels)))
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

    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create datasets and dataloaders
    train_dataset = GestureDataset(X_train_scaled, y_train)
    val_dataset = GestureDataset(X_val_scaled, y_val)
    test_dataset = GestureDataset(X_test_scaled, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Choose model type (SVM or CNN)
    use_svm = False  # Set to False to use CNN instead
    
    if use_svm:
        print("Using SVM model...")
        model = GestureSVM(num_classes=config["num_classes"]).to(device)
        # For SVM, we don't need an optimizer or criterion
        optimizer = None
        criterion = None
        # Perform grid search
        model.grid_search(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            param_grid={
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 0.001, 0.01, 0.1],
                'svm__kernel': ['rbf']
            },
            cv=5
        )

        # Get predictions for confusion matrix
        y_true = y_test
        y_pred = model.predict(X_test)

        # Save results with the grid search visualizations
        config = {
            'model_type': 'svm',
            'num_classes': 10,
            'input_dim': X_train.shape[1],
            'best_params': model.grid_search_results.best_params_,
            'class_names': config.get("class_names", [f"Class_{i}" for i in range(config["num_classes"])])
        }

        train_losses = model.train_losses
        val_losses = model.val_losses
        val_accuracies = model.val_accuracies
        test_accuracies = model.test_accuracies
        
        # Save results with the grid search visualizations and confusion matrix
        saved_paths = save_results(
            model=model,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            test_accuracies=test_accuracies,
            scaler=scaler,
            config=config,
            grid_search_results=model.grid_search_results,
            y_true=y_true,
            y_pred=y_pred
        )
    else:
        print("Using CNN model...")
        model = GestureCNN(num_classes=config["num_classes"]).to(device)
        
        # For CNN, we need optimizer and loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Train CNN model
        train_losses, val_losses, val_accuracies, test_accuracies = train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            config["num_epochs"],
            device,
        )
        
        # Get predictions for confusion matrix
        y_true = []
        y_pred = []
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        # Save all results using our new function
        saved_paths = save_results(
            model=model,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            test_accuracies=test_accuracies,
            scaler=scaler,
            config=config,
            y_true=y_true,
            y_pred=y_pred
        )

    print("Training complete. Check results folder for outputs.")
    return model, scaler, config


if __name__ == "__main__":
    model, scaler, config = main()
