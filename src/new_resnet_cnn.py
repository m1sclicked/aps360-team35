import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


# class ResNetGesture(nn.Module):
#     """
#     Transfer-learning model that can freeze or partially freeze layers in ResNet18.
#     Supported freeze_mode options:
#       - 'full': Freezes all conv layers, only final FC is trainable
#       - 'layer2': Freezes layers [conv1, bn1, layer1], unfreezes [layer2, layer3, layer4, fc]
#       - 'layer3': Freezes layers [conv1, bn1, layer1, layer2], unfreezes [layer3, layer4, fc]
#       - 'layer4': Freezes layers [conv1, bn1, layer1, layer2, layer3], unfreezes [layer4, fc]
#       - 'none': No freezing at all, everything is trainable (fully fine-tuned)
#     """
#     def __init__(self, num_classes=10, pretrained=True, freeze_mode='none'):
#         super(ResNetGesture, self).__init__()
#         # Load pre-trained ResNet18 (ImageNet) weights if requested
#         self.resnet = models.resnet18(
#             weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#         )

#         # Adjust the first convolution layer for single-dimension input:
#         #  - kernel_size=(7,3), stride=(2,1), padding=(3,1) treats
#         #    your feature dimension like "image height" and a single "image width"
#         self.resnet.conv1 = nn.Conv2d(
#             in_channels=3,
#             out_channels=64,
#             kernel_size=(7, 3),
#             stride=(2, 1),
#             padding=(3, 1),
#             bias=False
#         )

#         # 1) Freeze everything initially
#         for param in self.resnet.parameters():
#             param.requires_grad = False

#         # 2) Unfreeze selected layers based on freeze_mode
#         def unfreeze_module(module):
#             for param in module.parameters():
#                 param.requires_grad = True

#         # Unfreeze logic
#         if freeze_mode == 'full':
#             # Only final fully connected layer is trainable
#             pass  # everything else stays frozen
#         elif freeze_mode == 'layer2':
#             unfreeze_module(self.resnet.layer2)
#             unfreeze_module(self.resnet.layer3)
#             unfreeze_module(self.resnet.layer4)
#         elif freeze_mode == 'layer3':
#             unfreeze_module(self.resnet.layer2)
#             unfreeze_module(self.resnet.layer3)
#             unfreeze_module(self.resnet.layer4)
#             # layer2 and beyond are trainable
#         elif freeze_mode == 'layer4':
#             unfreeze_module(self.resnet.layer4)
#         elif freeze_mode == 'none':
#             # fully unfreeze everything
#             for param in self.resnet.parameters():
#                 param.requires_grad = True
#         else:
#             # In case someone passes 'partial' or something else
#             # we unfreeze only layer3, layer4, and fc as your original code did
#             unfreeze_module(self.resnet.layer3)
#             unfreeze_module(self.resnet.layer4)

#         # Always unfreeze the final FC, so we can learn new classes
#         unfreeze_module(self.resnet.fc)

#         # Replace the final fully connected layer
#         in_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         """
#         Input shape: [batch_size, features]
#         Steps to convert into a 4D tensor suitable for ResNet:
#           1) unsqueeze(1) -> [batch_size, 1, features]
#           2) repeat -> [batch_size, 3, features]
#           3) unsqueeze(-1) -> [batch_size, 3, features, 1]
#         """
#         x = x.unsqueeze(1)       # shape: [B, 1, features]
#         x = x.repeat(1, 3, 1)    # shape: [B, 3, features]
#         x = x.unsqueeze(-1)      # shape: [B, 3, features, 1]
#         return self.resnet(x)


class ResNetGesture(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, freeze_layers=True):
        super(ResNetGesture, self).__init__()
        # Load a pre-trained ResNet model with updated weights syntax.
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Modify the first convolution layer to accept our 1D feature input.
        # Kernel size (7,3) adapts the filter to work with our data (features treated as "height").
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1), bias=False)
        
        if freeze_layers:
            # Freeze all parameters first.
            for param in self.resnet.parameters():
                param.requires_grad = False
            # Unfreeze later layers (layer3, layer4, and the classifier) for fine-tuning.
            for param in self.resnet.layer3.parameters():
                param.requires_grad = True
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        
        # Replace the final fully connected layer with one that outputs num_classes.
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Expected input: x with shape [batch_size, features]
        Steps to convert into a 4D tensor:
          1. unsqueeze(1): [batch_size, 1, features]
          2. repeat across channel dimension: [batch_size, 3, features]
          3. unsqueeze(-1): [batch_size, 3, features, 1]
        """
        x = x.unsqueeze(1)       # Shape: [batch_size, 1, features]
        x = x.repeat(1, 3, 1)      # Shape: [batch_size, 3, features]
        x = x.unsqueeze(-1)       # Shape: [batch_size, 3, features, 1]
        return self.resnet(x)









# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision.models import ResNet18_Weights

# class ResNetGesture(nn.Module):
#     def __init__(self, num_classes=10, pretrained=True, freeze_layers=True):
#         super(ResNetGesture, self).__init__()
#         self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
#         self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1), bias=False)
        
#         if freeze_layers:
#             # Freeze all layers...
#             for param in self.resnet.parameters():
#                 param.requires_grad = False
#             # ...except unfreeze layer3 and layer4 for fine-tuning
#             for param in self.resnet.layer3.parameters():
#                 param.requires_grad = True
#             for param in self.resnet.layer4.parameters():
#                 param.requires_grad = True

#         in_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         x = x.unsqueeze(1)  # [batch_size, 1, features]
#         x = x.repeat(1, 3, 1)  # [batch_size, 3, features]
#         x = x.unsqueeze(-1)   # [batch_size, 3, features, 1]
#         return self.resnet(x)









# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision.models import ResNet18_Weights

# class ResNetGesture(nn.Module):
#     def __init__(self, num_classes=10, pretrained=True, freeze_layers=True):
#         super(ResNetGesture, self).__init__()
#         # Load a pre-trained ResNet model using the new weights parameter
#         self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
#         # Modify the first convolution to accept our 1D feature input:
#         # We adapt the kernel to work with a "height" equal to the number of features and a width of 1.
#         self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1), bias=False)
        
#         # Optionally freeze all layers except the final classifier
#         if freeze_layers:
#             for param in self.resnet.parameters():
#                 param.requires_grad = False

#         # Replace the final fully connected layer to match the number of gesture classes
#         in_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         """
#         Input: x of shape [batch_size, features]
#         We need to convert it into a 4D tensor:
#           1. unsqueeze(1): [batch_size, 1, features]
#           2. repeat to get 3 channels: [batch_size, 3, features]
#           3. unsqueeze(-1): [batch_size, 3, features, 1]
#         """
#         x = x.unsqueeze(1)       # [batch_size, 1, features]
#         x = x.repeat(1, 3, 1)      # [batch_size, 3, features]
#         x = x.unsqueeze(-1)       # [batch_size, 3, features, 1]
#         return self.resnet(x)
