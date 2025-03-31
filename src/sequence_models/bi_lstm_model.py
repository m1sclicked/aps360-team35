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

class TemporalConvBlock(nn.Module):
    """Temporal convolution block for capturing multi-scale patterns"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2  # Same padding
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels // len(kernel_sizes), 
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels // len(kernel_sizes)),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, channels, seq_length]
        """
        # Apply each conv and concatenate results
        conv_outputs = [conv(x) for conv in self.convs]
        return torch.cat(conv_outputs, dim=1)



class ResidualBiLSTMModel(nn.Module):
    """BiLSTM with residual connections for better gradient flow"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Projection for residual connection if dimensions don't match
        self.projection = nn.Linear(input_size, hidden_size * 2) if input_size != hidden_size * 2 else nn.Identity()
        
        # Input projection layer (to handle varying input sizes)
        self.input_projection = None
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout after LSTM
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, hx=None):
        # Check input dimensions
        actual_input_size = x.shape[-1]
        
        if actual_input_size != self.input_size:
            # If input size is different from expected, use input projection
            if self.input_projection is None or self.input_projection.in_features != actual_input_size:
                self.input_projection = nn.Linear(actual_input_size, self.input_size).to(x.device)
            
            # Project input to expected size for LSTM
            x_for_lstm = self.input_projection(x)
            
            # Also adjust projection for residual connection if needed
            if not isinstance(self.projection, nn.Identity):
                self.projection = nn.Linear(actual_input_size, self.hidden_size * 2).to(x.device)
        else:
            # If input size matches expected, use as is
            x_for_lstm = x
        
        # Save input for residual connection
        residual = self.projection(x)
        
        # Apply BiLSTM
        out, hidden = self.lstm(x_for_lstm, hx)
        
        # Apply dropout
        out = self.dropout_layer(out)
        
        # Add residual connection
        out = out + residual
        
        return out, hidden


class MultiResolutionBiLSTMAttentionModel(nn.Module):
    """
    Enhanced BiLSTM model with:
    1. Multi-resolution temporal modeling
    2. Residual connections
    3. Improved attention mechanism
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=2, dropout=0.5, temporal_dropout_prob=0.1):
        super().__init__()
        
        # Feature transformation with dropout
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.temporal_conv_full = TemporalConvBlock(
            in_channels=hidden_dim, 
            out_channels=hidden_dim,
            kernel_sizes=[3, 5, 7]
        )
        
        self.temporal_conv_half = TemporalConvBlock(
            in_channels=hidden_dim, 
            out_channels=hidden_dim,
            kernel_sizes=[3, 5]
        )
        
        self.temporal_conv_quarter = TemporalConvBlock(
            in_channels=hidden_dim, 
            out_channels=hidden_dim,
            kernel_sizes=[3]
        )
        
        # Temporal dropout before LSTM
        self.temporal_dropout = TemporalDropout(dropout_prob=temporal_dropout_prob)
        
        # Multi-resolution processing paths
        self.original_lstm = ResidualBiLSTMModel(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.half_res_lstm = ResidualBiLSTMModel(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Smaller hidden dim for downsampled version
            num_layers=1,
            dropout=dropout
        )
        
        self.quarter_res_lstm = ResidualBiLSTMModel(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 4,  # Even smaller hidden dim for further downsampled
            num_layers=1,
            dropout=dropout
        )
        
        # Calculate combined output dimension from all resolutions
        self.combined_dim = hidden_dim * 2 + (hidden_dim // 2) * 2 + (hidden_dim // 4) * 2
        
        # Add pre-attention dropout
        self.pre_attention_dropout = nn.Dropout(dropout/2)
        
        # Attention mechanisms - one for each resolution
        self.attention_full = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Tanh()
        )
        
        self.attention_half = nn.Sequential(
            nn.Linear(hidden_dim * 1, 1),
            nn.Tanh()
        )
        
        self.attention_quarter = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        # Fusion layer to combine multi-resolution features
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier with increased dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Apply feature transformation
        x = self.feature_transform(x)
        
        # Apply temporal dropout
        x = self.temporal_dropout(x)
        
        # Create downsampled versions for multi-resolution processing
        # Half resolution: kernel_size=2, stride=2
        x_half = F.avg_pool1d(
            x.transpose(1, 2),  # [B, F, T]
            kernel_size=2, 
            stride=2,
            ceil_mode=True
        ).transpose(1, 2)  # [B, T/2, F]
        
        # Quarter resolution: kernel_size=4, stride=4
        x_quarter = F.avg_pool1d(
            x.transpose(1, 2),  # [B, F, T]
            kernel_size=4, 
            stride=4,
            ceil_mode=True
        ).transpose(1, 2)  # [B, T/4, F]
        
        # Apply temporal convolutions to each resolution
        x_full_conv = self.temporal_conv_full(x.transpose(1, 2)).transpose(1, 2)
        x_half_conv = self.temporal_conv_half(x_half.transpose(1, 2)).transpose(1, 2)
        x_quarter_conv = self.temporal_conv_quarter(x_quarter.transpose(1, 2)).transpose(1, 2)
        
        # Process each resolution through BiLSTMs
        lstm_out_full, _ = self.original_lstm(x_full_conv)
        lstm_out_half, _ = self.half_res_lstm(x_half_conv)
        lstm_out_quarter, _ = self.quarter_res_lstm(x_quarter_conv)
        
        # Apply pre-attention dropout
        lstm_out_full = self.pre_attention_dropout(lstm_out_full)
        lstm_out_half = self.pre_attention_dropout(lstm_out_half)
        lstm_out_quarter = self.pre_attention_dropout(lstm_out_quarter)
        
        # Apply attention to each resolution
        # Full resolution attention
        attn_weights_full = self.attention_full(lstm_out_full).squeeze(-1)
        
        if mask is not None:
            # Apply mask to attention weights
            if mask.size(1) != seq_len:
                if mask.size(1) < seq_len:
                    padding = torch.zeros(batch_size, seq_len - mask.size(1), device=mask.device, dtype=mask.dtype)
                    mask = torch.cat([mask, padding], dim=1)
                else:
                    mask = mask[:, :seq_len]
            attn_weights_full = attn_weights_full.masked_fill(~mask, float('-inf'))
        
        attn_weights_full = F.softmax(attn_weights_full, dim=1).unsqueeze(-1)
        context_full = torch.sum(lstm_out_full * attn_weights_full, dim=1)
        
        # Half resolution attention
        half_seq_len = lstm_out_half.size(1)
        half_mask = None
        if mask is not None:
            # Downsample mask to match half resolution
            half_mask = mask[:, ::2]
            if half_mask.size(1) < half_seq_len:
                padding = torch.zeros(batch_size, half_seq_len - half_mask.size(1), 
                                     device=half_mask.device, dtype=half_mask.dtype)
                half_mask = torch.cat([half_mask, padding], dim=1)
            else:
                half_mask = half_mask[:, :half_seq_len]
        
        attn_weights_half = self.attention_half(lstm_out_half).squeeze(-1)
        if half_mask is not None:
            attn_weights_half = attn_weights_half.masked_fill(~half_mask, float('-inf'))
        attn_weights_half = F.softmax(attn_weights_half, dim=1).unsqueeze(-1)
        context_half = torch.sum(lstm_out_half * attn_weights_half, dim=1)
        
        # Quarter resolution attention
        quarter_seq_len = lstm_out_quarter.size(1)
        quarter_mask = None
        if mask is not None:
            # Downsample mask to match quarter resolution
            quarter_mask = mask[:, ::4]
            if quarter_mask.size(1) < quarter_seq_len:
                padding = torch.zeros(batch_size, quarter_seq_len - quarter_mask.size(1), 
                                     device=quarter_mask.device, dtype=quarter_mask.dtype)
                quarter_mask = torch.cat([quarter_mask, padding], dim=1)
            else:
                quarter_mask = quarter_mask[:, :quarter_seq_len]
        
        attn_weights_quarter = self.attention_quarter(lstm_out_quarter).squeeze(-1)
        if quarter_mask is not None:
            attn_weights_quarter = attn_weights_quarter.masked_fill(~quarter_mask, float('-inf'))
        attn_weights_quarter = F.softmax(attn_weights_quarter, dim=1).unsqueeze(-1)
        context_quarter = torch.sum(lstm_out_quarter * attn_weights_quarter, dim=1)
        
        # Concatenate multi-resolution features
        multi_res_context = torch.cat([context_full, context_half, context_quarter], dim=1)
        
        # Apply fusion layer
        fused_context = self.fusion(multi_res_context)
        
        # Apply classifier
        output = self.classifier(fused_context)
        
        return output