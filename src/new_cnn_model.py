import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedGestureCNN(nn.Module):
    def __init__(self, num_classes=10, input_dim=126, dropout_rate=0.5):
        super(ImprovedGestureCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 128, stride=2)
        self.res3 = ResidualBlock(128, 256, stride=2)
        self.res4 = ResidualBlock(256, 512, stride=2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Reshape input: [batch_size, features] -> [batch_size, 1, features]
        x = x.view(x.size(0), 1, -1)
        
        # Initial convolution block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # Apply attention
        weights = self.attention(x)
        x = x * weights
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x