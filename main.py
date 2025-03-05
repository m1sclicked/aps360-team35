import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import GestureCNN
from src.dataset import prepare_dataset, GestureDataset, download_wlasl_data
from src.train import train_model

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def setup_results_dirs():
    """Create necessary directories for results"""
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("data/wlasl_data", exist_ok=True)


def save_training_plots(train_losses, val_losses, val_accuracies, test_accuracies):
    """Save training progress plots"""
    plt.figure(figsize=(15, 5))

    # Losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Losses")
    plt.legend()

    # Validation Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")

    # Test Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies)
    plt.title("Test Accuracy")

    plt.tight_layout()
    plt.savefig("results/plots/training_progress.png")
    plt.close()


def main():
    # Setup results and data directories
    setup_results_dirs()

    # Configuration
    config = {
        "num_classes": 10,
        "samples_per_class": 50,
        "batch_size": 32,
        "num_epochs": 25,
        "learning_rate": 0.001,
        "data_path": "data/wlasl_data",
    }

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Download/load dataset
    print("Step 1: Loading WLASL dataset...")
    wlasl_data = download_wlasl_data(config["data_path"], download=True)

    if wlasl_data is None:
        print("Failed to load or download dataset. Exiting.")
        return

    # Prepare dataset
    print(f"Step 2: Preparing dataset with {config['num_classes']} classes...")
    features, labels = prepare_dataset(
        wlasl_data,
        data_path=config["data_path"],
        num_classes=config["num_classes"],
        max_samples_per_class=config["samples_per_class"],
    )

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
    print(device)
    model = GestureCNN(num_classes=config["num_classes"]).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train model
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

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "final_test_accuracy": test_accuracies[-1],
        },
        "results/models/gesture_model.pth",
    )

    # Save scaler
    np.save("results/models/gesture_scaler.npy", scaler)

    # Save training progress plots
    save_training_plots(train_losses, val_losses, val_accuracies, test_accuracies)

    # Log results
    with open("results/logs/training_results.txt", "w") as f:
        f.write(
            f"""
        Training completed!
        Final Results:
        - Test Accuracy: {test_accuracies[-1]:.2f}%
        - Validation Accuracy: {val_accuracies[-1]:.2f}%
        - Training Loss: {train_losses[-1]:.4f}
        """
        )

    print("Training complete. Check results folder for outputs.")

    return model, scaler, config


if __name__ == "__main__":
    model, scaler, config = main()
