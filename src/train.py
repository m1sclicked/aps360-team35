import torch
import numpy as np
from sklearn.metrics import accuracy_score

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
    Train the model with logging of losses and accuracies
    """
    train_losses = []
    val_losses = []
    val_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(
                device
            )

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct = 0
        test_correct = 0
        val_total = 0
        test_total = 0

        with torch.no_grad():
            # Validation
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(
                    device
                ), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

            # Test
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(
                    device
                ), batch_labels.to(device)
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()

        # Calculate metrics
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        test_accuracy = 100 * test_correct / test_total

        # Store metrics
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print("-" * 50)

        if scheduler:
            scheduler.step(avg_val_loss)

    return train_losses, val_losses, val_accuracies, test_accuracies

def train_svm_model(model, train_loader, val_loader, test_loader, criterion=None, optimizer=None, num_epochs=None, device=None):
    """
    Train a scikit-learn SVM model using data from PyTorch DataLoaders
    
    Args:
        model: GestureSVM instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        criterion, optimizer, num_epochs, device: Kept for API compatibility, not used
        
    Returns:
        train_losses: List of training losses (will contain placeholder values)
        val_losses: List of validation losses (will contain placeholder values)
        val_accuracies: List of validation accuracies
        test_accuracies: List of test accuracies
    """
    print("Extracting data from dataloaders...")
    
    # Extract data from data loaders
    X_train = []
    y_train = []
    for features, labels in train_loader:
        X_train.append(features.numpy())
        y_train.append(labels.numpy())
    
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    # Train SVM model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    X_val = []
    y_val = []
    for features, labels in val_loader:
        X_val.append(features.numpy())
        y_val.append(labels.numpy())
    
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    val_predictions = model.predict(X_val)
    val_accuracy = np.mean(val_predictions == y_val) * 100
    
    # Evaluate on test set
    X_test = []
    y_test = []
    for features, labels in test_loader:
        X_test.append(features.numpy())
        y_test.append(labels.numpy())
    
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test) * 100
    
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Return dummy losses and actual accuracies
    # This maintains compatibility with the CNN training interface
    return [0.0], [0.0], [val_accuracy], [test_accuracy]
