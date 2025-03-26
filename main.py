import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models  # still imported if needed elsewhere

# Import your models and utilities from the src folder
from src.cnn_model import GestureCNN
from src.svm_model import GestureSVM
from src.new_cnn_model import ImprovedGestureCNN
from src.new_resnet_cnn import ResNetGesture  # Updated ResNet-based model
from src.dataset import integrated_prepare_dataset, GestureDataset, download_wlasl_data
from src.train import train_model, train_svm_model
from src.augmenter import apply_data_augmentation
from src.save import save_results

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Create results directories
    os.makedirs("results/cnn", exist_ok=True)
    os.makedirs("results/svm", exist_ok=True)
    os.makedirs("results/resnet", exist_ok=True)

    # Configuration settings
    config = {
        "num_classes": 10,
        "batch_size": 64,
        "num_epochs": 100,           # Lower epochs for transfer learning
        "learning_rate": 0.0001,
        "data_path": "data/wlasl_data",
        "use_data_augmentation": True,
        "use_svm": False,           # Toggle SVM vs. CNN vs. ResNet
        "use_resnet": True,
        "new_resnet_cnn": True,         # Set True to use the ResNet transfer learning model
        "freeze_resnet": True,      # Freeze early layers for feature extraction
    }

    # Set random seeds for reproducibility
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Step 1: Load the WLASL dataset
    print("Step 1: Loading WLASL dataset...")
    wlasl_data = download_wlasl_data(config["data_path"], download=True)
    if wlasl_data is None:
        print("Failed to load dataset. Exiting.")
        return

    # Step 2: Prepare the dataset
    print(f"Step 2: Preparing dataset with {config['num_classes']} classes...")
    features, labels = integrated_prepare_dataset(
        data_path=config["data_path"],
        num_classes=config["num_classes"],
        min_videos_per_class=1,
    )

    # Split the dataset into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # Standardize the features (useful for SVM, and sometimes for CNN input)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply data augmentation if enabled
    if config["use_data_augmentation"]:
        print("Applying data augmentation to training set...")
        X_train_processed, y_train_processed = apply_data_augmentation(X_train, y_train)
        X_train_processed_scaled = scaler.transform(X_train_processed)
    else:
        print("Data augmentation disabled. Using original training set...")
        X_train_processed_scaled = X_train_scaled
        y_train_processed = y_train

    # Create PyTorch datasets
    train_dataset = GestureDataset(X_train_processed_scaled, y_train_processed)
    val_dataset = GestureDataset(X_val_scaled, y_val)
    test_dataset = GestureDataset(X_test_scaled, y_test)

    # Create DataLoaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # Setup device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # if config["new_resnet_cnn"]:
    #     print("Using New ResNet for Transfer Learning...")
    #     from src.new_resnet_cnn import ResNetGesture
    #     model = ResNetGesture(num_classes=config["num_classes"], pretrained=True, freeze_layers=config["freeze_resnet"]).to(device)
    # # ... rest of your training code ...


    # Select and train the model based on configuration
    if config["use_svm"]:
        # --- SVM MODE ---
        print("Using SVM model...")
        model = GestureSVM(num_classes=config["num_classes"])
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
        y_pred = model.predict(X_test)
        saved_paths = save_results(
            model=model, scaler=scaler, config=config, y_true=y_test, y_pred=y_pred
        )
    elif config["new_resnet_cnn"]:
        # --- RESNET MODE ---
        print("Using ResNet for Transfer Learning...")
        model = ResNetGesture(num_classes=config["num_classes"], pretrained=True, freeze_layers=config["freeze_resnet"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # train_losses, val_losses, val_accuracies, test_accuracies = train_model(
        #     model, train_loader, val_loader, test_loader,
        #     criterion, optimizer, scheduler, config["num_epochs"], device
        # )
        train_losses, val_losses, val_accuracies, test_accuracies, \
        val_precisions, val_recalls, val_f1s, \
        test_precisions, test_recalls, test_f1s = train_model(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler, config["num_epochs"], device
        )
        saved_paths = save_results(
            model=model, train_losses=train_losses, val_losses=val_losses,
            val_accuracies=val_accuracies, test_accuracies=test_accuracies,
            scaler=scaler, config=config
        )
    else:
        # --- IMPROVED CNN MODE ---
        print("Using Improved CNN model...")
        model = ImprovedGestureCNN(num_classes=config["num_classes"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        train_losses, val_losses, val_accuracies, test_accuracies = train_model(
            model, train_loader, val_loader, test_loader,
            criterion, optimizer, scheduler, config["num_epochs"], device
        )
        saved_paths = save_results(
            model=model, train_losses=train_losses, val_losses=val_losses,
            val_accuracies=val_accuracies, test_accuracies=test_accuracies,
            scaler=scaler, config=config
        )

    print("Training complete. Check results folder for outputs.")
    return model, scaler, config

if __name__ == "__main__":
    model, scaler, config = main()







# NEW BUT WEIRD
# def main():
#     # Create results directory
#     os.makedirs("results/resnet_transfer", exist_ok=True)

#     # Configuration settings
#     config = {
#         "num_classes": 10,
#         "batch_size": 32,
#         "num_epochs": 100,      
#         "learning_rate": 1e-5,  # Lower LR recommended for fine-tuning
#         "data_path": "data/wlasl_data",
#         "use_data_augmentation": True,

#         # Transfer-learning options
#         "use_resnet_transfer": True,
#         "freeze_mode": "none",  # 'full', 'partial', or 'none'
#     }

#     # Seed for reproducibility
#     torch.manual_seed(42)
#     torch.cuda.manual_seed_all(42)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     # Step 1: Download or load the dataset
#     print("Step 1: Loading WLASL dataset...")
#     wlasl_data = download_wlasl_data(config["data_path"], download=False)
#     if wlasl_data is None:
#         print("Failed to load dataset. Exiting.")
#         return

#     # Step 2: Prepare the dataset
#     print(f"Step 2: Preparing dataset with {config['num_classes']} classes...")
#     features, labels = integrated_prepare_dataset(
#         data_path=config["data_path"],
#         num_classes=config["num_classes"],
#         min_videos_per_class=1,
#     )

#     # Train/validation/test split
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         features, labels, test_size=0.2, random_state=42
#     )
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp, test_size=0.2, random_state=42
#     )

#     # Standardize the features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)

#     # Optional data augmentation
#     if config["use_data_augmentation"]:
#         print("Applying data augmentation to training set...")
#         X_train_aug, y_train_aug = apply_data_augmentation(X_train, y_train)
#         # Re-scale the augmented data
#         X_train_scaled = scaler.transform(X_train_aug)
#         y_train = y_train_aug

#     # Create datasets
#     train_dataset = GestureDataset(X_train_scaled, y_train)
#     val_dataset = GestureDataset(X_val_scaled, y_val)
#     test_dataset = GestureDataset(X_test_scaled, y_test)

#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
#     test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

#     # Determine device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Initialize ResNet transfer model
#     print("Using ResNet for transfer learning...")
#     model = ResNetGesture(
#         num_classes=config["num_classes"],
#         pretrained=True,
#         freeze_mode=config["freeze_mode"]
#     ).to(device)

#     # Define loss, optimizer, and LR scheduler
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(
#         filter(lambda p: p.requires_grad, model.parameters()), 
#         lr=config["learning_rate"]
#     )
#     # Example: step down LR every 15 epochs by factor of 0.5
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

#     # Train the model
#     train_losses, val_losses, val_accuracies, test_accuracies = train_model(
#         model, train_loader, val_loader, test_loader,
#         criterion, optimizer, scheduler, config["num_epochs"], device
#     )

#     # Save training results, confusion matrix, etc.
#     saved_paths = save_results(
#         model=model,
#         train_losses=train_losses,
#         val_losses=val_losses,
#         val_accuracies=val_accuracies,
#         test_accuracies=test_accuracies,
#         scaler=scaler,
#         config=config
#     )

#     print("Training complete. Check the results folder for outputs.")
#     return model, scaler, config


# if __name__ == "__main__":
#     model, scaler, config = main()









# OLD BUT SEMI-WEIRD
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import models # importing for ResNet


# from src.cnn_model import GestureCNN
# from src.svm_model import GestureSVM
# from src.new_cnn_model import ImprovedGestureCNN
# from src.dataset import integrated_prepare_dataset, GestureDataset, download_wlasl_data
# from src.train import train_model, train_svm_model
# from src.augmenter import KeypointAugmenter, apply_data_augmentation, augment_dataset
# from src.save import save_results

# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# def main():
    
#     # Also set up model-specific directories
#     os.makedirs("results/cnn", exist_ok=True)
#     os.makedirs("results/svm", exist_ok=True)
#     os.makedirs("results/unknown", exist_ok=True)

#     # Configuration
#     config = {
#         "num_classes": 10,
#         "min_samples_per_class": 1,
#         "max_samples_per_class": 1000,
#         "batch_size": 16,
#         "num_epochs": 200,
#         "learning_rate": 0.001,
#         "data_path": "data/wlasl_data",
#         "use_data_augmentation": True,
#     }

#     # Set random seeds for reproducibility
#     torch.cuda.manual_seed(42)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     # Download/load dataset
#     print("Step 1: Loading WLASL dataset...")
#     wlasl_data = download_wlasl_data(config["data_path"], download=True)

#     if wlasl_data is None:
#         print("Failed to load or download dataset. Exiting.")
#         return

#     # Prepare dataset
#     print(f"Step 2: Preparing dataset with {config['num_classes']} classes...")
#     features, labels = integrated_prepare_dataset(
#         data_path=config["data_path"],
#         num_classes=config["num_classes"],
#         min_videos_per_class=config["min_samples_per_class"],
#     )

    

#     # Get class names if available
#     try:
#         import json
#         with open(os.path.join(config["data_path"], "WLASL_v0.3.json"), 'r') as f:
#             wlasl_json = json.load(f)
        
#         # Extract class names based on the selected classes
#         unique_labels = sorted(list(set(labels)))
#         class_names = []
        
#         # First, create a mapping of indices to glosses
#         gloss_mapping = {}
#         for i, entry in enumerate(wlasl_json):
#             gloss_mapping[i] = entry['gloss']
        
#         # Now map the class labels to glosses
#         for label in unique_labels:
#             if label < len(gloss_mapping):
#                 class_names.append(gloss_mapping[label])
#             else:
#                 # If we couldn't find a name, use a placeholder
#                 class_names.append(f"Class_{label}")
        
#         config["class_names"] = class_names
#         print(f"Found class names: {class_names}")
#     except Exception as e:
#         print(f"Could not extract class names: {e}")
#         # Create generic class names
#         config["class_names"] = [f"Class_{i}" for i in range(config["num_classes"])]

#     # Split dataset
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         features, labels, test_size=0.2, random_state=42
#     )
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp, test_size=0.2, random_state=42
#     )
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)

#     if config["use_data_augmentation"]:
#         print("Applying data augmentation to training set...")
#         X_train_processed, y_train_processed = apply_data_augmentation(X_train, y_train)
#         X_train_processed_scaled = scaler.transform(X_train_processed)
#     else:
#         print("Data augmentation disabled. Using original training set...")
#         X_train_processed = X_train
#         y_train_processed = y_train
#         X_train_processed_scaled = X_train_scaled

#     train_dataset = GestureDataset(X_train_processed_scaled, y_train_processed)
#     val_dataset = GestureDataset(X_val_scaled, y_val)
#     test_dataset = GestureDataset(X_test_scaled, y_test)

#     train_loader = DataLoader(
#         train_dataset, batch_size=config["batch_size"], shuffle=True
#     )
#     val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
#     test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

#     # Setup model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Choose model type (SVM or CNN)
#     use_svm = False  # Set to False to use CNN instead
    
#     if use_svm:
#         print("Using SVM model...")
#         model = GestureSVM(num_classes=config["num_classes"]).to(device)
#         # For SVM, we don't need an optimizer or criterion
#         optimizer = None
#         criterion = None
#         # Perform grid search
#         model.grid_search(
#             X_train, y_train,
#             X_val=X_val, y_val=y_val,
#             X_test=X_test, y_test=y_test,
#             param_grid={
#                 'svm__C': [0.1, 1, 10, 100],
#                 'svm__gamma': ['scale', 0.001, 0.01, 0.1],
#                 'svm__kernel': ['rbf']
#             },
#             cv=5
#         )

#         # Get predictions for confusion matrix
#         y_true = y_test
#         y_pred = model.predict(X_test)

#         # Save results with the grid search visualizations
#         config = {
#             'model_type': 'svm',
#             'num_classes': 10,
#             'input_dim': X_train.shape[1],
#             'best_params': model.grid_search_results.best_params_,
#             'class_names': config.get("class_names", [f"Class_{i}" for i in range(config["num_classes"])])
#         }

#         train_losses = model.train_losses
#         val_losses = model.val_losses
#         val_accuracies = model.val_accuracies
#         test_accuracies = model.test_accuracies
        
#         # Save results with the grid search visualizations and confusion matrix
#         saved_paths = save_results(
#             model=model,
#             train_losses=train_losses,
#             val_losses=val_losses,
#             val_accuracies=val_accuracies,
#             test_accuracies=test_accuracies,
#             scaler=scaler,
#             config=config,
#             grid_search_results=model.grid_search_results,
#             y_true=y_true,
#             y_pred=y_pred
#         )
#     else:
#         print("Using CNN model...")
#         model = ImprovedGestureCNN(num_classes=config["num_classes"]).to(device)
        
#         # For CNN, we need optimizer and loss function
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
#         # Train CNN model
#         train_losses, val_losses, val_accuracies, test_accuracies = train_model(
#             model,
#             train_loader,
#             val_loader,
#             test_loader,
#             criterion,
#             optimizer,
#             scheduler,
#             config["num_epochs"],
#             device,
#         )
        
#         # Get predictions for confusion matrix
#         y_true = []
#         y_pred = []
        
#         model.eval()
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
                
#                 outputs = model(inputs)
#                 _, predictions = torch.max(outputs, 1)
                
#                 y_true.extend(labels.cpu().numpy())
#                 y_pred.extend(predictions.cpu().numpy())
        
#         # Save all results using our new function
#         saved_paths = save_results(
#             model=model,
#             train_losses=train_losses,
#             val_losses=val_losses,
#             val_accuracies=val_accuracies,
#             test_accuracies=test_accuracies,
#             scaler=scaler,
#             config=config,
#             y_true=y_true,
#             y_pred=y_pred
#         )

#     print("Training complete. Check results folder for outputs.")
#     return model, scaler, config


# if __name__ == "__main__":
#     model, scaler, config = main()
