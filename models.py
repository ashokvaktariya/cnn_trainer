import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from monai.networks.nets import EfficientNet
except ImportError:
    # Fallback to lib if direct import fails
    from lib.monai.networks.nets import EfficientNet
import numpy as np

# =============================================================================
# BINARY CLASSIFICATION MODELS FOR FRACTURE DETECTION
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in binary classification"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=2).float()
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BinaryEfficientNet(nn.Module):
    """Binary Classification Model using EfficientNet-B7 for fracture detection"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3, pretrained=True):
        super().__init__()
        
        # Use EfficientNet-B7 backbone
        self.backbone = EfficientNet(
            model_name="efficientnet-b7",
            spatial_dims=2,
            in_channels=3,  # RGB images
            pretrained=pretrained,
            dropout_prob=dropout_rate
        )
        
        # Get feature dimension from EfficientNet-B7
        feature_dim = 2560  # EfficientNet-B7 feature dimension
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224) - single image
        if x.dim() == 5:
            # If batch contains multiple images, take the first one
            x = x[:, 0]  # (batch_size, 3, 224, 224)
        
        # Extract features using EfficientNet backbone
        features = self.backbone(x)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits
    
    def get_attention_maps(self, x):
        """Get attention maps for visualization"""
        if x.dim() == 5:
            x = x[:, 0]
        
        # Get intermediate features for attention visualization
        features = self.backbone.features(x)
        attention_maps = F.adaptive_avg_pool2d(features, (1, 1))
        
        return attention_maps

class BinaryDenseNet(nn.Module):
    """Alternative Binary Classification Model using DenseNet-121"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3, pretrained=True):
        super().__init__()
        
        # Use DenseNet-121 backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # DenseNet blocks (simplified)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224)
        if x.dim() == 5:
            x = x[:, 0]  # Take first image if multiple provided
        
        # Extract features
        features = self.backbone(x)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits

# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(model_type="BinaryEfficientNet", num_classes=2, **kwargs):
    """Factory function to create models"""
    
    if model_type == "BinaryEfficientNet":
        return BinaryEfficientNet(num_classes=num_classes, **kwargs)
    elif model_type == "BinaryDenseNet":
        return BinaryDenseNet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_loss_function(use_focal_loss=True, **kwargs):
    """Create loss function for binary classification"""
    
    if use_focal_loss:
        return FocalLoss(**kwargs)
    else:
        return nn.CrossEntropyLoss()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_summary(model, input_size=(1, 3, 224, 224)):
    """Get model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Summary:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024
    }

if __name__ == "__main__":
    # Test model creation
    model = create_model("BinaryEfficientNet", num_classes=2)
    get_model_summary(model)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"✅ Forward pass successful: {output.shape}")