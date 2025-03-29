import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStageTemporalCNNModel(nn.Module):
    """
    Multi-stage temporal CNN model for ASL sequence classification
    """
    def __init__(self, input_dim, num_classes, channels=[64, 128, 256], kernel_sizes=[5, 5, 3],
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # First stage - temporal convolutions
        self.stages = nn.ModuleList()
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            stage = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.stages.append(stage)
            in_channels = out_channels
        
        # Global average pooling + global max pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Transpose for 1D convolution: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
            x = x * mask.float()
        
        # Apply convolutional stages
        for stage in self.stages:
            x = stage(x)
            if mask is not None:
                # Downsample mask to match x
                mask = F.max_pool1d(mask.float(), kernel_size=2, stride=2) > 0
                x = x * mask.float()
        
        # Apply global pooling
        avg_pooled = self.gap(x).squeeze(-1)
        max_pooled = self.gmp(x).squeeze(-1)
        
        # Concatenate pooled features
        x = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Classification
        output = self.classifier(x)
        return output