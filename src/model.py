import torch
import torch.nn as nn


class GestureCNN(nn.Module):
    def __init__(self, num_classes=10, input_dim=126):
        super(GestureCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # first conv block
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(1),
            # ADDING NEW LAYER
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(1),
            # Adding ANOTHER new layer
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.AdaptiveAvgPool1d(1),
        )

        # fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.7), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)

        # Pass through the convolutional layers
        x = self.conv_layers(x)

        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc_layers(x)
        return x
