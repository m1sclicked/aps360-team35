import torch


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
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

    return train_losses, val_losses, val_accuracies, test_accuracies
