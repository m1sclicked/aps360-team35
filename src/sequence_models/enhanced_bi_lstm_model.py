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

class FeatureDropout(nn.Module):
    """Applies dropout to entire feature channels"""
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
            
        # Create dropout mask for features
        batch_size, seq_length, num_features = x.shape
        mask = torch.rand(batch_size, 1, num_features, device=x.device) > self.dropout_prob
        mask = mask.float() / (1 - self.dropout_prob)  # Scale to maintain expected value
        
        return x * mask.expand_as(x)

class GatedResidualConnection(nn.Module):
    """Gated residual connection for better gradient flow"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, residual):
        # Concatenate input and residual for gating decision
        gate_input = torch.cat([x, residual], dim=-1)
        gate = self.gate(gate_input)
        
        # Apply gate to control residual flow
        return gate * residual + (1 - gate) * x

class TemporalConvBlock(nn.Module):
    """Enhanced temporal convolution block with residual connections"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        
        self.out_channels = out_channels
        num_kernels = len(kernel_sizes)
        
        # Calculate channels per kernel, ensuring we use all channels
        base_channels = out_channels // num_kernels
        remainder = out_channels % num_kernels
        
        self.convs = nn.ModuleList()
        start_idx = 0
        
        for i, kernel_size in enumerate(kernel_sizes):
            # Allocate extra channel to early convolutions if there's a remainder
            current_channels = base_channels + (1 if i < remainder else 0)
            
            padding = kernel_size // 2  # Same padding
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, current_channels, 
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(current_channels),
                nn.GELU(),  # GELU activation often works better than ReLU
                nn.Dropout(0.1)
            ))
            start_idx += current_channels
            
        # Add residual connection if input and output dimensions match
        self.use_residual = (in_channels == out_channels)
        
        # Always create projection layer - we'll use it when needed
        self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, channels, seq_length]
        """
        # Save input for potential residual connection
        residual = x
        
        # Apply each conv and concatenate results
        conv_outputs = [conv(x) for conv in self.convs]
        output = torch.cat(conv_outputs, dim=1)
        
        # Verify output channel dimension matches expected
        assert output.size(1) == self.out_channels, f"Expected {self.out_channels} channels but got {output.size(1)}"
        
        # Check if the sequence length dimension has changed
        if output.size(2) != residual.size(2):
            # Adjust residual to match output sequence length using interpolation
            residual = F.interpolate(residual, size=output.size(2), mode='linear')
        
        # Add residual connection
        if self.use_residual:
            # Direct addition if channels match
            if residual.size(1) == output.size(1):
                output = output + residual
            else:
                # Project residual to match output channels
                projected_residual = self.projection(residual)
                output = output + projected_residual
        else:
            # Always project when not using direct residual
            projected_residual = self.projection(residual)
            # Check sequence length again after projection
            if output.size(2) != projected_residual.size(2):
                projected_residual = F.interpolate(projected_residual, size=output.size(2), mode='linear')
            output = output + projected_residual
            
        return output

class ResidualBiLSTMModel(nn.Module):
    """Enhanced BiLSTM with gated residual connections"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Residual projection (output of LSTM to residual connection)
        self.projection = nn.Linear(hidden_size * 2, hidden_size * 2) if hidden_size * 2 != hidden_size * 2 else nn.Identity()
        
        # Input projection layer (to handle varying input sizes)
        self.input_projection = None  # Will be created if needed during forward pass
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Gated residual connection
        self.gated_residual = GatedResidualConnection(hidden_size)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Dropout after LSTM
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, hx=None):
        # Check input dimensions and create input projection if needed
        actual_input_size = x.shape[-1]
        
        if actual_input_size != self.input_size:
            if self.input_projection is None or self.input_projection.in_features != actual_input_size:
                print(f"Creating input projection: {actual_input_size} -> {self.input_size}")
                self.input_projection = nn.Linear(actual_input_size, self.input_size).to(x.device)
            
            # Apply input projection
            x_projected = self.input_projection(x)
        else:
            x_projected = x
        
        # Apply BiLSTM
        out, hidden = self.lstm(x_projected, hx)
        
        # Apply dropout
        out = self.dropout_layer(out)
        
        # Save projected output for residual connection
        residual = x_projected if actual_input_size == self.hidden_size * 2 else self.projection(out)
        
        # Apply gated residual connection (splitting for each direction)
        batch_size, seq_len, _ = out.shape
        fwd_out = out[:, :, :self.hidden_size]
        bwd_out = out[:, :, self.hidden_size:]
        
        fwd_residual = residual[:, :, :self.hidden_size] if residual.size(-1) == self.hidden_size * 2 else torch.zeros_like(fwd_out)
        bwd_residual = residual[:, :, self.hidden_size:] if residual.size(-1) == self.hidden_size * 2 else torch.zeros_like(bwd_out)
        
        fwd_gated = self.gated_residual(fwd_out, fwd_residual)
        bwd_gated = self.gated_residual(bwd_out, bwd_residual)
        
        # Recombine outputs
        out = torch.cat([fwd_gated, bwd_gated], dim=2)
        
        # Apply layer normalization
        out = self.layer_norm(out)
        
        return out, hidden

class MultiHeadTemporalAttention(nn.Module):
    """Multi-head attention mechanism for temporal sequences"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_dim]
            mask: Optional mask tensor of shape [batch_size, seq_length]
        """
        batch_size, seq_length, _ = x.shape
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose to get dimensions [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention dimensions
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
            scores = scores.masked_fill(~expanded_mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Apply output projection
        output = self.output_proj(context)
        
        return output

class CrossResolutionAttention(nn.Module):
    """Cross-attention between different temporal resolutions"""
    def __init__(self, query_dim, key_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert query_dim % num_heads == 0, "Query dimension must be divisible by number of heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)  # Project key to query dimension
        self.v_proj = nn.Linear(key_dim, query_dim)  # Project value to query dimension
        
        # Output projection
        self.output_proj = nn.Linear(query_dim, query_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, query_mask=None, kv_mask=None):
        """
        Args:
            query: Query tensor of shape [batch_size, query_seq_length, query_dim]
            key_value: Key/Value tensor of shape [batch_size, kv_seq_length, key_dim]
            query_mask: Optional mask for query of shape [batch_size, query_seq_length]
            kv_mask: Optional mask for key/value of shape [batch_size, kv_seq_length]
        """
        batch_size, query_seq_length, _ = query.shape
        _, kv_seq_length, _ = key_value.shape
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, query_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Apply masks if provided
        if kv_mask is not None:
            # Expand key/value mask to match attention dimensions
            expanded_kv_mask = kv_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, kv_seq_length]
            scores = scores.masked_fill(~expanded_kv_mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        context = context.transpose(1, 2).contiguous().view(batch_size, query_seq_length, q.size(-1) * self.num_heads)
        
        # Apply output projection
        output = self.output_proj(context)
        
        return output

class MultiResolutionBiLSTMAttentionModelEnhanced(nn.Module):
    """
    Enhanced BiLSTM model with:
    1. Multi-resolution temporal modeling
    2. Multi-head attention
    3. Cross-resolution attention
    4. Gated residual connections
    5. Feature and temporal dropout
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=2, 
                 dropout=0.5, temporal_dropout_prob=0.1, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Feature transformation with dropout
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Temporal convolutional blocks for different resolutions
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
        
        # Feature dropout for regularization
        self.feature_dropout = FeatureDropout(dropout_prob=dropout/2)
        
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
        
        # Multi-head attention for each resolution
        self.attention_full = MultiHeadTemporalAttention(
            hidden_dim=hidden_dim * 2,  # BiLSTM output has 2x hidden_dim
            num_heads=num_heads,
            dropout=dropout/2
        )
        
        self.attention_half = MultiHeadTemporalAttention(
            hidden_dim=hidden_dim,  # BiLSTM output has 2x (hidden_dim/2)
            num_heads=num_heads//2,
            dropout=dropout/2
        )
        
        self.attention_quarter = MultiHeadTemporalAttention(
            hidden_dim=hidden_dim//2,  # BiLSTM output has 2x (hidden_dim/4)
            num_heads=max(1, num_heads//4),
            dropout=dropout/2
        )
        
        # Cross-resolution attention mechanisms
        self.cross_attn_half_to_full = CrossResolutionAttention(
            query_dim=hidden_dim * 2,
            key_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout/2
        )
        
        self.cross_attn_quarter_to_full = CrossResolutionAttention(
            query_dim=hidden_dim * 2,
            key_dim=hidden_dim//2,
            num_heads=num_heads,
            dropout=dropout/2
        )
        
        # Calculate combined output dimension from all resolutions
        self.combined_dim = (hidden_dim * 2) + hidden_dim + (hidden_dim // 2)
        
        # Add pre-attention dropout
        self.pre_attention_dropout = nn.Dropout(dropout/2)
        
        # Fusion layer to combine multi-resolution features with layer norm
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Classifier with increased dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Apply feature transformation
        x = self.feature_transform(x)
        
        # Apply feature and temporal dropout for regularization
        x = self.feature_dropout(x)
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
        # Save original sequence lengths before convolution
        orig_full_len = x.size(1)
        orig_half_len = x_half.size(1)
        orig_quarter_len = x_quarter.size(1)
        
        # Apply temporal convolutions with proper handling of dimensions
        x_full_conv = self.temporal_conv_full(x.transpose(1, 2))
        x_half_conv = self.temporal_conv_half(x_half.transpose(1, 2))
        x_quarter_conv = self.temporal_conv_quarter(x_quarter.transpose(1, 2))
        
        # Check if temporal convolution changed sequence lengths
        if x_full_conv.size(2) != orig_full_len:
            # Adjust sequence length using interpolation
            x_full_conv = F.interpolate(x_full_conv, size=orig_full_len, mode='linear')
        
        if x_half_conv.size(2) != orig_half_len:
            x_half_conv = F.interpolate(x_half_conv, size=orig_half_len, mode='linear')
        
        if x_quarter_conv.size(2) != orig_quarter_len:
            x_quarter_conv = F.interpolate(x_quarter_conv, size=orig_quarter_len, mode='linear')
        
        # Transpose back to [batch_size, seq_len, features]
        x_full_conv = x_full_conv.transpose(1, 2)
        x_half_conv = x_half_conv.transpose(1, 2)
        x_quarter_conv = x_quarter_conv.transpose(1, 2)
        
        # Process each resolution through BiLSTMs
        lstm_out_full, _ = self.original_lstm(x_full_conv)
        lstm_out_half, _ = self.half_res_lstm(x_half_conv)
        lstm_out_quarter, _ = self.quarter_res_lstm(x_quarter_conv)
        
        # Apply pre-attention dropout
        lstm_out_full = self.pre_attention_dropout(lstm_out_full)
        lstm_out_half = self.pre_attention_dropout(lstm_out_half)
        lstm_out_quarter = self.pre_attention_dropout(lstm_out_quarter)
        
        # Prepare masks for different resolutions
        half_seq_len = lstm_out_half.size(1)
        quarter_seq_len = lstm_out_quarter.size(1)
        
        half_mask = None
        quarter_mask = None
        
        if mask is not None:
            # Downsample masks to match each resolution
            if mask.size(1) != seq_len:
                if mask.size(1) < seq_len:
                    padding = torch.zeros(batch_size, seq_len - mask.size(1), device=mask.device, dtype=mask.dtype)
                    mask = torch.cat([mask, padding], dim=1)
                else:
                    mask = mask[:, :seq_len]
            
            # Create half resolution mask
            half_mask = mask[:, ::2]
            if half_mask.size(1) < half_seq_len:
                padding = torch.zeros(batch_size, half_seq_len - half_mask.size(1), 
                                    device=half_mask.device, dtype=half_mask.dtype)
                half_mask = torch.cat([half_mask, padding], dim=1)
            else:
                half_mask = half_mask[:, :half_seq_len]
            
            # Create quarter resolution mask
            quarter_mask = mask[:, ::4]
            if quarter_mask.size(1) < quarter_seq_len:
                padding = torch.zeros(batch_size, quarter_seq_len - quarter_mask.size(1), 
                                    device=quarter_mask.device, dtype=quarter_mask.dtype)
                quarter_mask = torch.cat([quarter_mask, padding], dim=1)
            else:
                quarter_mask = quarter_mask[:, :quarter_seq_len]
        
        # Apply multi-head self-attention to each resolution
        attn_full = self.attention_full(lstm_out_full, mask)
        attn_half = self.attention_half(lstm_out_half, half_mask)
        attn_quarter = self.attention_quarter(lstm_out_quarter, quarter_mask)
        
        # Apply cross-resolution attention to enrich full resolution with information from others
        cross_half_to_full = self.cross_attn_half_to_full(lstm_out_full, lstm_out_half, mask, half_mask)
        cross_quarter_to_full = self.cross_attn_quarter_to_full(lstm_out_full, lstm_out_quarter, mask, quarter_mask)
        
        # Combine self-attention and cross-attention for full resolution
        combined_full = attn_full + cross_half_to_full + cross_quarter_to_full
        
        # Get global context vectors via mean pooling
        if mask is not None:
            # Create expanded masks for proper masked pooling
            expanded_mask = mask.unsqueeze(-1).float()
            expanded_half_mask = half_mask.unsqueeze(-1).float()
            expanded_quarter_mask = quarter_mask.unsqueeze(-1).float()
            
            # Apply masked mean pooling
            mask_sum = expanded_mask.sum(dim=1, keepdim=True) + 1e-8
            half_mask_sum = expanded_half_mask.sum(dim=1, keepdim=True) + 1e-8
            quarter_mask_sum = expanded_quarter_mask.sum(dim=1, keepdim=True) + 1e-8
            
            context_full = (combined_full * expanded_mask).sum(dim=1) / mask_sum.squeeze(1)
            context_half = (attn_half * expanded_half_mask).sum(dim=1) / half_mask_sum.squeeze(1)
            context_quarter = (attn_quarter * expanded_quarter_mask).sum(dim=1) / quarter_mask_sum.squeeze(1)
        else:
            # Simple mean pooling if no masks
            context_full = combined_full.mean(dim=1)
            context_half = attn_half.mean(dim=1)
            context_quarter = attn_quarter.mean(dim=1)
        
        # Concatenate multi-resolution features
        multi_res_context = torch.cat([context_full, context_half, context_quarter], dim=1)
        
        # Apply fusion layer to combine multi-resolution features
        fused_context = self.fusion(multi_res_context)
        
        # Apply final layer normalization
        fused_context = self.final_layer_norm(fused_context)
        
        # Apply classifier
        output = self.classifier(fused_context)
        
        return output