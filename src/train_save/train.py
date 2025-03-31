import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
):
    """
    Train the model with logging of losses and metrics (Accuracy, Precision, Recall, F1).
    """
    train_losses = []
    val_losses = []
    val_accuracies = []
    test_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        total_train_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- EVAL ---
        model.eval()
        total_val_loss = 0
        val_preds, val_labels = [], []
        test_preds, test_labels = [], []

        with torch.no_grad():
            # --- VALIDATION ---
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

            # --- TEST ---
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(batch_labels.cpu().numpy())

        # --- METRICS CALCULATION ---
        avg_val_loss = total_val_loss / len(val_loader)

        val_accuracy = accuracy_score(val_labels, val_preds) * 100
        val_precision = precision_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)

        test_accuracy = accuracy_score(test_labels, test_preds) * 100
        test_precision = precision_score(test_labels, test_preds, average="weighted", zero_division=0)
        test_recall = recall_score(test_labels, test_preds, average="weighted", zero_division=0)
        test_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)

        # --- LOGGING ---
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.2f}% | Precision: {val_precision:.2f} | Recall: {val_recall:.2f} | F1: {val_f1:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}% | Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1: {test_f1:.2f}")
        print("-" * 50)

        # --- SCHEDULER STEP ---
        if scheduler:
            scheduler.step()

        # --- SAVE METRICS ---
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)

    return (
        train_losses,
        val_losses,
        val_accuracies,
        test_accuracies,
        val_precisions,
        val_recalls,
        val_f1s,
        test_precisions,
        test_recalls,
        test_f1s,
    )


def train_svm_model(model, train_loader, val_loader, test_loader, criterion=None, optimizer=None, num_epochs=None, device=None):
    """
    Train a scikit-learn SVM model using data from PyTorch DataLoaders.
    Maintains compatible return structure with CNN training.
    """
    print("Extracting data from dataloaders...")

    # --- Extract training data ---
    X_train, y_train = [], []
    for features, labels in train_loader:
        X_train.append(features.numpy())
        y_train.append(labels.numpy())
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    # --- Train SVM ---
    model.fit(X_train, y_train)

    # --- Extract val data ---
    X_val, y_val = [], []
    for features, labels in val_loader:
        X_val.append(features.numpy())
        y_val.append(labels.numpy())
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    val_preds = model.predict(X_val)

    # --- Extract test data ---
    X_test, y_test = [], []
    for features, labels in test_loader:
        X_test.append(features.numpy())
        y_test.append(labels.numpy())
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    test_preds = model.predict(X_test)

    # --- Compute metrics ---
    val_accuracy = accuracy_score(y_val, val_preds) * 100
    val_precision = precision_score(y_val, val_preds, average="weighted", zero_division=0)
    val_recall = recall_score(y_val, val_preds, average="weighted", zero_division=0)
    val_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)

    test_accuracy = accuracy_score(y_test, test_preds) * 100
    test_precision = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    test_recall = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    # --- Log ---
    print(f"Validation Accuracy: {val_accuracy:.2f}% | Precision: {val_precision:.2f} | Recall: {val_recall:.2f} | F1: {val_f1:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}% | Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1: {test_f1:.2f}")

    return (
        [0.0], [0.0], [val_accuracy], [test_accuracy],
        [val_precision], [val_recall], [val_f1],
        [test_precision], [test_recall], [test_f1]
    )






# import torch
# import numpy as np
# #from sklearn.metrics import accuracy_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def train_model(
#     model,
#     train_loader,
#     val_loader,
#     test_loader,
#     criterion,
#     optimizer,
#     scheduler,
#     num_epochs,
#     device,
# ):
#     """
#     Train the model with logging of losses and accuracies
#     """
#     train_losses = []
#     val_losses = []
#     val_accuracies = []
#     test_accuracies = []

#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         total_train_loss = 0
#         for batch_features, batch_labels in train_loader:
#             batch_features, batch_labels = batch_features.to(device), batch_labels.to(
#                 device
#             )

#             optimizer.zero_grad()
#             outputs = model(batch_features)
#             loss = criterion(outputs, batch_labels)
#             loss.backward()
#             optimizer.step()

#             total_train_loss += loss.item()

#         avg_train_loss = total_train_loss / len(train_loader)
#         train_losses.append(avg_train_loss)

#         # Validation phase
#         model.eval()
#         total_val_loss = 0
#         val_correct = 0
#         test_correct = 0
#         val_total = 0
#         test_total = 0

#         with torch.no_grad():
#             # Validation
#             for batch_features, batch_labels in val_loader:
#                 batch_features, batch_labels = batch_features.to(
#                     device
#                 ), batch_labels.to(device)
#                 outputs = model(batch_features)
#                 loss = criterion(outputs, batch_labels)
#                 total_val_loss += loss.item()

#                 _, predicted = torch.max(outputs.data, 1)
#                 val_total += batch_labels.size(0)
#                 val_correct += (predicted == batch_labels).sum().item()

#             # Test
#             for batch_features, batch_labels in test_loader:
#                 batch_features, batch_labels = batch_features.to(
#                     device
#                 ), batch_labels.to(device)
#                 outputs = model(batch_features)
#                 _, predicted = torch.max(outputs.data, 1)
#                 test_total += batch_labels.size(0)
#                 test_correct += (predicted == batch_labels).sum().item()

#         # Calculate metrics
#         avg_val_loss = total_val_loss / len(val_loader)
#         val_accuracy = 100 * val_correct / val_total
#         test_accuracy = 100 * test_correct / test_total

#         # Store metrics
#         val_losses.append(avg_val_loss)
#         val_accuracies.append(val_accuracy)
#         test_accuracies.append(test_accuracy)

#         # Print epoch results
#         print(f"Epoch [{epoch+1}/{num_epochs}]")
#         print(f"Train Loss: {avg_train_loss:.4f}")
#         print(f"Val Loss: {avg_val_loss:.4f}")
#         print(f"Val Accuracy: {val_accuracy:.2f}%")
#         print(f"Test Accuracy: {test_accuracy:.2f}%")
#         print("-" * 50)

#         if scheduler:
#             scheduler.step(avg_val_loss)

#     # return train_losses, val_losses, val_accuracies, test_accuracies
#     return train_losses, val_losses, val_accuracies, test_accuracies, val_precisions, val_recalls, val_f1s, test_precisions, test_recalls, test_f1s

# def train_svm_model(model, train_loader, val_loader, test_loader, criterion=None, optimizer=None, num_epochs=None, device=None):
#     """
#     Train a scikit-learn SVM model using data from PyTorch DataLoaders
    
#     Args:
#         model: GestureSVM instance
#         train_loader: DataLoader for training data
#         val_loader: DataLoader for validation data
#         test_loader: DataLoader for test data
#         criterion, optimizer, num_epochs, device: Kept for API compatibility, not used
        
#     Returns:
#         train_losses: List of training losses (will contain placeholder values)
#         val_losses: List of validation losses (will contain placeholder values)
#         val_accuracies: List of validation accuracies
#         test_accuracies: List of test accuracies
#     """
#     print("Extracting data from dataloaders...")
    
#     # Extract data from data loaders
#     X_train = []
#     y_train = []
#     for features, labels in train_loader:
#         X_train.append(features.numpy())
#         y_train.append(labels.numpy())
    
#     X_train = np.vstack(X_train)
#     y_train = np.concatenate(y_train)
    
#     # Train SVM model
#     model.fit(X_train, y_train)
    
#     # Evaluate on validation set
#     X_val = []
#     y_val = []
#     for features, labels in val_loader:
#         X_val.append(features.numpy())
#         y_val.append(labels.numpy())
    
#     X_val = np.vstack(X_val)
#     y_val = np.concatenate(y_val)
#     val_predictions = model.predict(X_val)
#     val_accuracy = np.mean(val_predictions == y_val) * 100
    
#     # Evaluate on test set
#     X_test = []
#     y_test = []
#     for features, labels in test_loader:
#         X_test.append(features.numpy())
#         y_test.append(labels.numpy())
    
#     X_test = np.vstack(X_test)
#     y_test = np.concatenate(y_test)
#     test_predictions = model.predict(X_test)
#     test_accuracy = np.mean(test_predictions == y_test) * 100
    
#     print(f"Validation Accuracy: {val_accuracy:.2f}%")
#     print(f"Test Accuracy: {test_accuracy:.2f}%")
    
#     # Return dummy losses and actual accuracies
#     # This maintains compatibility with the CNN training interface
#     return [0.0], [0.0], [val_accuracy], [test_accuracy]

def train_sequence_model(model, train_loader, val_loader, test_loader, criterion, optimizer, 
                          scheduler, num_epochs, device, early_stop_patience=15, 
                          improvement_threshold=0.001, monitor_metric='accuracy',
                          l2_lambda=0.0001, l2_excluded_layers=('lstm',)):
    """
    Train a sequence-based model with masking, metrics tracking, and additional L2 regularization
    
    Args:
        model: The model to train
        train_loader, val_loader, test_loader: DataLoaders
        criterion: Loss function (can be standard CrossEntropyLoss or ConfidencePenaltyLoss)
        optimizer: Optimizer
        scheduler: Learning rate scheduler (supports OneCycleLR with per-batch stepping)
        num_epochs: Maximum number of training epochs
        device: Training device
        early_stop_patience: Patience for early stopping (default: 15)
        improvement_threshold: Minimum improvement required to reset patience (default: 0.001)
        monitor_metric: Metric to monitor for early stopping ('loss' or 'accuracy')
        l2_lambda: L2 regularization strength (default: 0.0001)
        l2_excluded_layers: Tuple of layer name substrings to exclude from L2 regularization
        
    Returns:
        Metrics and training history
    """
    # Initialize tracking variables
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    test_precisions = []
    test_recalls = []
    test_f1s = []
    
    # Check if scheduler is OneCycleLR
    is_one_cycle = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
    
    # Early stopping variables
    if monitor_metric == 'loss':
        best_metric = float('inf')  # Lower is better for loss
        is_better = lambda current, best: current < best - improvement_threshold
    else:  # 'accuracy'
        best_metric = 0.0  # Higher is better for accuracy
        is_better = lambda current, best: current > best + improvement_threshold
    
    patience_counter = 0
    best_model_state = None
    
    # Import metrics calculation functions
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import copy
    
    for epoch in range(num_epochs):
        # --- TRAIN ---
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)
        model.train()
        total_train_loss = 0
        train_preds, train_labels_list = [], []
        
        for batch_features, batch_masks, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features, batch_masks)
            
            # Use the provided criterion (could be standard or confidence penalty)
            # The criterion should handle the primary loss calculation
            primary_loss = criterion(outputs, batch_labels)
            
            # Calculate L2 regularization loss
            l2_reg = 0.0
            for name, param in model.named_parameters():
                # Only apply L2 regularization to weights and exclude specified layers
                if 'weight' in name and not any(excluded in name for excluded in l2_excluded_layers):
                    l2_reg += torch.norm(param, 2) ** 2
            
            # Combine primary loss with weighted L2 loss
            loss = primary_loss + l2_lambda * l2_reg
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Step the scheduler after each batch if using OneCycleLR
            if is_one_cycle:
                scheduler.step()
                
            total_train_loss += loss.item() * batch_labels.size(0)
            
            # Collect training predictions for accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().detach().numpy())
            train_labels_list.extend(batch_labels.cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Calculate training accuracy
        train_accuracy = accuracy_score(train_labels_list, train_preds) * 100
        train_accuracies.append(train_accuracy)
        
        # --- EVAL ---
        model.eval()
        total_val_loss = 0
        val_preds, val_labels_list = [], []
        test_preds, test_labels_list = [], []
        
        with torch.no_grad():
            # --- VALIDATION ---
            for batch_features, batch_masks, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_masks = batch_masks.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features, batch_masks)
                
                # Use the same criterion for validation
                loss = criterion(outputs, batch_labels)
                total_val_loss += loss.item() * batch_labels.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(batch_labels.cpu().numpy())
            
            # --- TEST ---
            for batch_features, batch_masks, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_masks = batch_masks.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features, batch_masks)
                _, predicted = torch.max(outputs.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels_list.extend(batch_labels.cpu().numpy())
        
        # --- METRICS CALCULATION ---
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Calculate accuracy, precision, recall, F1
        val_accuracy = accuracy_score(val_labels_list, val_preds) * 100
        val_precision = precision_score(val_labels_list, val_preds, average="weighted", zero_division=0)
        val_recall = recall_score(val_labels_list, val_preds, average="weighted", zero_division=0)
        val_f1 = f1_score(val_labels_list, val_preds, average="weighted", zero_division=0)
        
        test_accuracy = accuracy_score(test_labels_list, test_preds) * 100
        test_precision = precision_score(test_labels_list, test_preds, average="weighted", zero_division=0)
        test_recall = recall_score(test_labels_list, test_preds, average="weighted", zero_division=0)
        test_f1 = f1_score(test_labels_list, test_preds, average="weighted", zero_division=0)
        
        # Save metrics
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)
        
        # --- LOGGING ---
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        print(f"Val Metrics: Precision: {val_precision:.2f} | Recall: {val_recall:.2f} | F1: {val_f1:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}% | Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1: {test_f1:.2f}")
        
        # --- SCHEDULER STEP ---
        # Only step non-OneCycleLR schedulers here (e.g., ReduceLROnPlateau)
        if not is_one_cycle:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # --- EARLY STOPPING BASED ON SPECIFIED METRIC ---
        current_metric = avg_val_loss if monitor_metric == 'loss' else val_accuracy
        
        if is_better(current_metric, best_metric):
            print(f"Improvement found! {monitor_metric}: {best_metric:.4f} -> {current_metric:.4f}")
            best_metric = current_metric
            patience_counter = 0
            # Save best model
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            print(f"No improvement in {monitor_metric}. Patience: {patience_counter}/{early_stop_patience}")
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best {monitor_metric}: {best_metric:.4f}")
                break
        
        print("-" * 50)
    
    # Load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with {monitor_metric}: {best_metric:.4f}")
    
    return (
        model,
        train_losses,
        val_losses,
        val_accuracies,
        test_accuracies,
        val_precisions,
        val_recalls,
        val_f1s,
        test_precisions,
        test_recalls,
        test_f1s,
    )