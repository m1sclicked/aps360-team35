import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

class TemporalResNetGesture(nn.Module):
    """
    ResNet-based model adapted for temporal gesture recognition.
    This model handles sequence data by using a CNN backbone with temporal pooling.
    """
    def __init__(self, num_classes=10, input_dim=126, seq_length=150, pretrained=True, freeze_layers=True):
        super(TemporalResNetGesture, self).__init__()
        # Save input parameters
        self.input_dim = input_dim
        self.seq_length = seq_length
        
        # Load a pre-trained ResNet model with updated weights syntax
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Modify the first convolution layer to accept our temporal feature input
        # Instead of [batch, 3, H, W], we'll reshape our input to [batch, 3, seq_length, input_dim//3]
        # This requires the input_dim to be divisible by 3 to maintain the RGB channel structure
        self.input_reshape_dim = input_dim // 3
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=(7, 7), 
            stride=(2, 2), padding=(3, 3), bias=False
        )
        
        if freeze_layers:
            # Freeze all parameters first
            for param in self.resnet.parameters():
                param.requires_grad = False
            # Unfreeze later layers for fine-tuning
            for param in self.resnet.layer3.parameters():
                param.requires_grad = True
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        # Add an attention-based temporal pooling layer
        self.temporal_attention = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        """
        Forward pass for temporal gesture recognition
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            mask: Boolean mask of shape [batch_size, seq_length]
                  indicating valid (1) and padding (0) positions
        
        Returns:
            Class scores of shape [batch_size, num_classes]
        """
        batch_size, seq_length, input_dim = x.shape
        
        # Handle the case where seq_length is less than the expected seq_length
        if seq_length < self.seq_length:
            padding = torch.zeros(batch_size, self.seq_length - seq_length, input_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)
            
            # Update mask if provided
            if mask is not None:
                mask_padding = torch.zeros(batch_size, self.seq_length - seq_length, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, mask_padding], dim=1)
        
        # Reshape input to [batch, 3, seq_length, input_dim//3]
        # This treats the temporal dimension as the height and splits features into 3 channels
        x = x.reshape(batch_size, self.seq_length, 3, self.input_reshape_dim)
        x = x.permute(0, 2, 1, 3)  # [batch, 3, seq_length, input_dim//3]
        
        # First pass through ResNet up to the final global pooling
        # Get the features from each layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # At this point, x has shape [batch, channels, height, width]
        # Use adaptive pooling to get a fixed-size output regardless of input dimensions
        x = self.resnet.avgpool(x)  # [batch, channels, 1, 1]
        features = torch.flatten(x, 1)  # [batch, channels]
        
        # Apply final classification layer
        outputs = self.resnet.fc(features)
        
        return outputs

class TemporalResNetAttention(nn.Module):
    """
    ResNet-based model with explicit temporal attention for gesture recognition.
    This model extracts features from individual frames, then applies attention over time.
    """
    def __init__(self, num_classes=10, input_dim=126, seq_length=150, pretrained=True, freeze_layers=True):
        super(TemporalResNetAttention, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        
        # Load and modify ResNet for feature extraction
        self.resnet_base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Modify first conv to accept our input
        self.resnet_base.conv1 = nn.Conv2d(
            3, 64, kernel_size=(7, 7), 
            stride=(2, 2), padding=(3, 3), bias=False
        )
        
        # Remove the final classification layer
        self.feature_dim = self.resnet_base.fc.in_features
        self.resnet_base = nn.Sequential(*list(self.resnet_base.children())[:-1])
        
        if freeze_layers:
            # We can selectively freeze/unfreeze specific layers
            for i, child in enumerate(self.resnet_base.children()):
                # Freeze early layers (0-6), unfreeze later ones (7-8)
                if i < 7:  # Freeze up to layer3
                    for param in child.parameters():
                        param.requires_grad = False
        
        # Temporal attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Final classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x, mask=None):
        """
        Forward pass with temporal attention
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            mask: Boolean mask of shape [batch_size, seq_length]
                  indicating valid (1) and padding (0) positions
        
        Returns:
            Class scores of shape [batch_size, num_classes]
        """
        batch_size, seq_length, input_dim = x.shape
        
        # Reshape for frame-by-frame processing
        x_reshaped = x.reshape(batch_size * seq_length, input_dim)
        
        # Further reshape for ResNet input [batch*seq, 3, H, W]
        resnet_input = x_reshaped.view(batch_size * seq_length, 3, input_dim // 3, 1)
        
        # Extract features for each frame
        frame_features = self.resnet_base(resnet_input)
        frame_features = frame_features.view(batch_size * seq_length, self.feature_dim)
        
        # Reshape back to [batch, seq, features]
        frame_features = frame_features.view(batch_size, seq_length, self.feature_dim)
        
        # Apply attention over the temporal dimension
        attention_scores = self.attention(frame_features)  # [batch, seq, 1]
        
        # Apply mask if provided (set padding attention to very negative values)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # [batch, seq, 1]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(frame_features * attention_weights, dim=1)  # [batch, features]
        
        # Classification
        outputs = self.classifier(context_vector)
        
        return outputs