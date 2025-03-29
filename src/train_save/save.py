def save_results(model, train_losses, val_losses, val_accuracies, test_accuracies, 
               scaler, config, base_dir="results", grid_search_results=None, y_true=None, y_pred=None,
               val_precisions=None, val_recalls=None, val_f1s=None, 
               test_precisions=None, test_recalls=None, test_f1s=None):
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
        val_precisions: List of validation precision scores
        val_recalls: List of validation recall scores
        val_f1s: List of validation F1 scores
        test_precisions: List of test precision scores
        test_recalls: List of test recall scores
        test_f1s: List of test F1 scores
    
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
    
    # Determine model subfolder name from config if available, otherwise use default detection
    if 'model_type' in config:
        model_type = config['model_type']
    elif is_svm_model:
        model_type = "svm"
    elif is_torch_model:
        model_type = "cnn"
    else:
        model_type = "unknown"
    
    # Create unique result directory with timestamp
    # FIX: Use base_dir directly instead of joining with model_type again (model_type is already part of base_dir)
    result_dir = os.path.join(base_dir, timestamp)
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
        # Default plotting for CNN or other models without grid search
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{model_type.upper()} Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_plot_path = os.path.join(plots_dir, f"{model_type}_loss_curves.png")
        plt.savefig(loss_plot_path)
        plt.close()
        saved_paths['loss_plot'] = loss_plot_path
        
        # Plot accuracy curves
        plt.figure(figsize=(10, 6))
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.plot(test_accuracies, label="Test Accuracy")
        plt.title(f"{model_type.upper()} Validation and Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        acc_plot_path = os.path.join(plots_dir, f"{model_type}_accuracy_curves.png")
        plt.savefig(acc_plot_path)
        plt.close()
        saved_paths['accuracy_plot'] = acc_plot_path
        
        # Plot precision, recall, F1 if available
        if all(x is not None for x in [val_precisions, val_recalls, val_f1s, 
                                      test_precisions, test_recalls, test_f1s]):
            # Validation metrics
            plt.figure(figsize=(10, 6))
            plt.plot(val_precisions, label="Precision")
            plt.plot(val_recalls, label="Recall")
            plt.plot(val_f1s, label="F1 Score")
            plt.title(f"{model_type.upper()} Validation Metrics")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            val_metrics_path = os.path.join(plots_dir, f"{model_type}_validation_metrics.png")
            plt.savefig(val_metrics_path)
            plt.close()
            saved_paths['validation_metrics_plot'] = val_metrics_path
            
            # Test metrics
            plt.figure(figsize=(10, 6))
            plt.plot(test_precisions, label="Precision")
            plt.plot(test_recalls, label="Recall")
            plt.plot(test_f1s, label="F1 Score")
            plt.title(f"{model_type.upper()} Test Metrics")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            test_metrics_path = os.path.join(plots_dir, f"{model_type}_test_metrics.png")
            plt.savefig(test_metrics_path)
            plt.close()
            saved_paths['test_metrics_plot'] = test_metrics_path
            
            # Final metrics comparison
            plt.figure(figsize=(12, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            val_values = [val_accuracies[-1], val_precisions[-1], val_recalls[-1], val_f1s[-1]]
            test_values = [test_accuracies[-1], test_precisions[-1], test_recalls[-1], test_f1s[-1]]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, val_values, width, label='Validation')
            plt.bar(x + width/2, test_values, width, label='Test')
            
            plt.xticks(x, metrics)
            plt.ylabel('Score')
            plt.title(f'{model_type.upper()} Final Metrics Comparison')
            plt.legend()
            plt.grid(True, axis='y')
            plt.tight_layout()
            
            metrics_comparison_path = os.path.join(plots_dir, f"{model_type}_metrics_comparison.png")
            plt.savefig(metrics_comparison_path)
            plt.close()
            saved_paths['metrics_comparison'] = metrics_comparison_path
    
    # 2. Generate and save confusion matrix if provided with true and predicted labels
    if y_true is not None and y_pred is not None:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class names if available in config
        class_names = config.get('class_names', [f'Class {i}' for i in range(config['num_classes'])])
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))

        num_actual_classes = len(unique_classes)
        display_labels = class_names[:num_actual_classes] if len(class_names) >= num_actual_classes else None
        
        # Create confusion matrix display
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels 
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
        # Create header based on available metrics
        header = ["epoch", "train_loss", "val_loss", "val_accuracy", "test_accuracy"]
        if val_precisions is not None:
            header.extend(["val_precision", "val_recall", "val_f1"])
        if test_precisions is not None:
            header.extend(["test_precision", "test_recall", "test_f1"])
        
        f.write(",".join(header) + "\n")
        
        # Make sure all lists have the same length
        max_epochs = max(
            len(train_losses), len(val_losses), len(val_accuracies), len(test_accuracies),
            len(val_precisions or []), len(val_recalls or []), len(val_f1s or []),
            len(test_precisions or []), len(test_recalls or []), len(test_f1s or [])
        )
        
        for i in range(max_epochs):
            row = [str(i)]
            # Basic metrics
            row.append(str(train_losses[i]) if i < len(train_losses) else "")
            row.append(str(val_losses[i]) if i < len(val_losses) else "")
            row.append(str(val_accuracies[i]) if i < len(val_accuracies) else "")
            row.append(str(test_accuracies[i]) if i < len(test_accuracies) else "")
            
            # Additional validation metrics
            if val_precisions is not None:
                row.append(str(val_precisions[i]) if i < len(val_precisions) else "")
                row.append(str(val_recalls[i]) if i < len(val_recalls) else "")
                row.append(str(val_f1s[i]) if i < len(val_f1s) else "")
                
            # Additional test metrics
            if test_precisions is not None:
                row.append(str(test_precisions[i]) if i < len(test_precisions) else "")
                row.append(str(test_recalls[i]) if i < len(test_recalls) else "")
                row.append(str(test_f1s[i]) if i < len(test_f1s) else "")
            
            f.write(",".join(row) + "\n")
    
    saved_paths['metrics_csv'] = metrics_df_path
    
    # 4. Save model with metrics
    metrics = {
        'final_test_accuracy': test_accuracies[-1] if test_accuracies else None,
        'final_val_accuracy': val_accuracies[-1] if val_accuracies else None,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'model_type': model_type
    }
    
    # Add precision, recall, and F1 metrics if available
    if val_precisions is not None:
        metrics.update({
            'final_val_precision': val_precisions[-1],
            'final_val_recall': val_recalls[-1],
            'final_val_f1': val_f1s[-1],
        })
    
    if test_precisions is not None:
        metrics.update({
            'final_test_precision': test_precisions[-1],
            'final_test_recall': test_recalls[-1],
            'final_test_f1': test_f1s[-1],
        })
    
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
    
    # 5. Save scaler if provided
    if scaler is not None:
        scaler_path = os.path.join(models_dir, f"{model_type}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        saved_paths['scaler'] = scaler_path
    
    # 6. Log results
    log_path = os.path.join(logs_dir, f"{model_type}_training_results.txt")
    with open(log_path, 'w') as f:
        log_text = f"""
{model_type.upper()} Model Training completed!
Final Results:
- Test Accuracy: {test_accuracies[-1] if test_accuracies else 'N/A':.2f}%
- Validation Accuracy: {val_accuracies[-1] if val_accuracies else 'N/A':.2f}%
- Training Loss: {train_losses[-1] if train_losses else 'N/A':.4f}
"""
        
        # Add precision, recall, and F1 metrics if available
        if val_precisions is not None:
            log_text += f"""
Validation Metrics:
- Precision: {val_precisions[-1]:.4f}
- Recall: {val_recalls[-1]:.4f}
- F1 Score: {val_f1s[-1]:.4f}
"""
        
        if test_precisions is not None:
            log_text += f"""
Test Metrics:
- Precision: {test_precisions[-1]:.4f}
- Recall: {test_recalls[-1]:.4f}
- F1 Score: {test_f1s[-1]:.4f}
"""
        
        log_text += f"""
Configuration:
{json.dumps(config, indent=2)}

Results saved in: {result_dir}
"""
        f.write(log_text)
    
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