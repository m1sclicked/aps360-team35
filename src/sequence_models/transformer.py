import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        return x + self.pe[:, :x.size(1)]
    
class ASLTransformerModel(nn.Module):
    """
    Transformer-based model for ASL sequence classification
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_heads=8, 
                 num_layers=4, dropout=0.3):
        super().__init__()
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Global attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Create attention mask for transformer
        if mask is not None:
            # Convert boolean mask to transformer attention mask
            # True means valid position, False means padding
            # For transformer: False -> not masked, True -> masked
            # So we need to invert and convert to appropriate format
            transformer_mask = ~mask
        else:
            transformer_mask = None
            
        # Embed input features
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        
        # Apply attention pooling
        attn_weights = self.attention(x).squeeze(-1)
        if mask is not None:
            # Set weights to -inf where mask is False (padding)
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)
        
        # Weighted sum of sequence features
        x = torch.sum(x * attn_weights, dim=1)
        
        # Classification
        output = self.classifier(x)
        return output