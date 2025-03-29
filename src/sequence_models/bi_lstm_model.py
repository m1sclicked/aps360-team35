import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalDropout(nn.Module):
    """Applies dropout to entire timesteps in a sequence"""
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, features]
        """
        if not self.training or self.dropout_prob == 0:
            return x
            
        # Create dropout mask for timesteps
        batch_size, seq_length, _ = x.shape
        mask = torch.rand(batch_size, seq_length, 1, device=x.device) > self.dropout_prob
        mask = mask.float() / (1 - self.dropout_prob)  # Scale to maintain expected value
        
        return x * mask.expand_as(x)


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=2, dropout=0.5, temporal_dropout_prob=0.1):
        super().__init__()
        
        # Feature transformation with dropout
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)  # Apply dropout after activation
        )
        
        # Add temporal dropout before LSTM
        self.temporal_dropout = TemporalDropout(dropout_prob=temporal_dropout_prob)
        
        # BiLSTM with increased dropout between layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,  # Increased dropout between LSTM layers
            bidirectional=True
        )
        
        # Add dropout before attention
        self.pre_attention_dropout = nn.Dropout(dropout/2)  # Lighter dropout before attention
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Tanh()
        )
        
        # Classifier with increased dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Increased dropout
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Transform features
        x = self.feature_transform(x)
        
        # Apply temporal dropout before LSTM
        x = self.temporal_dropout(x)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout before attention
        lstm_out = self.pre_attention_dropout(lstm_out)
        
        # Apply attention mechanism
        attn_weights = self.attention(lstm_out).squeeze(-1)
        
        if mask is not None:
            # Handle mask logic as before
            if mask.size(1) != seq_len:
                if mask.size(1) < seq_len:
                    padding = torch.zeros(batch_size, seq_len - mask.size(1), device=mask.device, dtype=mask.dtype)
                    mask = torch.cat([mask, padding], dim=1)
                else:
                    mask = mask[:, :seq_len]
            
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classification
        output = self.classifier(context)
        
        return output