import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.monai.networks.nets import DenseNet121, EfficientNet, ResNet
from transformers import AutoModel, AutoTokenizer
import numpy as np

# =============================================================================
# MODEL 1 & 2: IMAGE-ONLY CNNs
# =============================================================================

class ImageOnlyDenseNet(nn.Module):
    """Model 1: DenseNet121 for image-only classification"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        # Use DenseNet121
        self.backbone = DenseNet121(
            spatial_dims=2,
            in_channels=3,  # RGB images
            out_channels=num_classes,
            dropout_prob=dropout_rate
        )
        
        # Add custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 3, 3, 224, 224) - 3 images per study
        batch_size, num_images, channels, height, width = x.shape
        
        # Process each image separately
        image_features = []
        for i in range(num_images):
            img = x[:, i]  # (batch_size, channels, height, width)
            features = self.backbone.features(img)
            features = self.classifier(features)
            image_features.append(features)
        
        # Average features across images
        combined_features = torch.stack(image_features, dim=1).mean(dim=1)
        
        return combined_features

class ImageOnlyEfficientNet(nn.Module):
    """Model 2: EfficientNet for image-only classification"""
    
    def __init__(self, num_classes=2, model_name="efficientnet-b0"):
        super().__init__()
        
        # Use EfficientNet
        self.backbone = EfficientNet(
            model_name=model_name,
            spatial_dims=2,
            in_channels=3,
            num_classes=num_classes,
            pretrained=True
        )
        
        # Custom classifier for multiple images
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 512),  # EfficientNet-B0 feature size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_images, channels, height, width = x.shape
        
        # Process each image
        image_features = []
        for i in range(num_images):
            img = x[:, i]
            features = self.backbone._conv_head(img)
            features = self.backbone._bn1(features)
            features = F.relu(features, inplace=True)
            features = self.classifier(features)
            image_features.append(features)
        
        # Average features
        combined_features = torch.stack(image_features, dim=1).mean(dim=1)
        
        return combined_features

# =============================================================================
# MODEL 3 & 4: MULTIMODAL (IMAGE + TEXT)
# =============================================================================

class MultimodalDenseNet(nn.Module):
    """Model 3: DenseNet121 + BERT for multimodal classification"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        # Image branch (DenseNet121)
        self.image_backbone = DenseNet121(
            spatial_dims=2,
            in_channels=3,
            out_channels=num_classes,
            dropout_prob=dropout_rate
        )
        
        self.image_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Text branch (BERT)
        self.text_model = AutoModel.from_pretrained("bert-base-uncased")
        self.text_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),  # 512 + 512
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, text_input_ids, text_attention_mask):
        # Image processing
        batch_size, num_images, channels, height, width = images.shape
        
        image_features = []
        for i in range(num_images):
            img = images[:, i]
            features = self.image_backbone.features(img)
            features = self.image_classifier(features)
            image_features.append(features)
        
        # Average image features
        img_features = torch.stack(image_features, dim=1).mean(dim=1)
        
        # Text processing
        text_outputs = self.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        text_features = self.text_classifier(text_features)
        
        # Fusion
        combined = torch.cat([img_features, text_features], dim=1)
        output = self.fusion(combined)
        
        return output

class MultimodalEfficientNet(nn.Module):
    """Model 4: EfficientNet + BERT for multimodal classification"""
    
    def __init__(self, num_classes=2, model_name="efficientnet-b0"):
        super().__init__()
        
        # Image branch (EfficientNet)
        self.image_backbone = EfficientNet(
            model_name=model_name,
            spatial_dims=2,
            in_channels=3,
            num_classes=num_classes,
            pretrained=True
        )
        
        self.image_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Text branch (BERT)
        self.text_model = AutoModel.from_pretrained("bert-base-uncased")
        self.text_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, text_input_ids, text_attention_mask):
        # Image processing
        batch_size, num_images, channels, height, width = images.shape
        
        image_features = []
        for i in range(num_images):
            img = images[:, i]
            features = self.image_backbone._conv_head(img)
            features = self.image_backbone._bn1(features)
            features = F.relu(features, inplace=True)
            features = self.image_classifier(features)
            image_features.append(features)
        
        # Average image features
        img_features = torch.stack(image_features, dim=1).mean(dim=1)
        
        # Text processing
        text_outputs = self.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        text_features = self.text_classifier(text_features)
        
        # Fusion
        combined = torch.cat([img_features, text_features], dim=1)
        output = self.fusion(combined)
        
        return output

# =============================================================================
# MODEL 5: ENSEMBLE
# =============================================================================

class MedicalEnsemble(nn.Module):
    """Model 5: Ensemble of all 4 models"""
    
    def __init__(self, model_paths, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load trained models
        self.models = nn.ModuleDict()
        
        # Model 1: Image-only DenseNet
        self.models['image_densenet'] = ImageOnlyDenseNet()
        self.models['image_densenet'].load_state_dict(
            torch.load(model_paths['image_densenet'], map_location=device)
        )
        
        # Model 2: Image-only EfficientNet
        self.models['image_efficientnet'] = ImageOnlyEfficientNet()
        self.models['image_efficientnet'].load_state_dict(
            torch.load(model_paths['image_efficientnet'], map_location=device)
        )
        
        # Model 3: Multimodal DenseNet
        self.models['multimodal_densenet'] = MultimodalDenseNet()
        self.models['multimodal_densenet'].load_state_dict(
            torch.load(model_paths['multimodal_densenet'], map_location=device)
        )
        
        # Model 4: Multimodal EfficientNet
        self.models['multimodal_efficientnet'] = MultimodalEfficientNet()
        self.models['multimodal_efficientnet'].load_state_dict(
            torch.load(model_paths['multimodal_efficientnet'], map_location=device)
        )
        
        # Freeze all models
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Meta-classifier
        self.meta_classifier = nn.Sequential(
            nn.Linear(8, 32),  # 4 models × 2 classes
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2)
        )
    
    def forward(self, images, text_input_ids=None, text_attention_mask=None):
        predictions = []
        probabilities = []
        
        # Get predictions from all models
        with torch.no_grad():
            # Model 1: Image-only DenseNet
            pred1 = self.models['image_densenet'](images)
            prob1 = F.softmax(pred1, dim=1)
            predictions.append(pred1)
            probabilities.append(prob1)
            
            # Model 2: Image-only EfficientNet
            pred2 = self.models['image_efficientnet'](images)
            prob2 = F.softmax(pred2, dim=1)
            predictions.append(pred2)
            probabilities.append(prob2)
            
            # Model 3: Multimodal DenseNet
            if text_input_ids is not None and text_attention_mask is not None:
                pred3 = self.models['multimodal_densenet'](images, text_input_ids, text_attention_mask)
            else:
                # Fallback to image-only prediction
                pred3 = self.models['image_densenet'](images)
            prob3 = F.softmax(pred3, dim=1)
            predictions.append(pred3)
            probabilities.append(prob3)
            
            # Model 4: Multimodal EfficientNet
            if text_input_ids is not None and text_attention_mask is not None:
                pred4 = self.models['multimodal_efficientnet'](images, text_input_ids, text_attention_mask)
            else:
                # Fallback to image-only prediction
                pred4 = self.models['image_efficientnet'](images)
            prob4 = F.softmax(pred4, dim=1)
            predictions.append(pred4)
            probabilities.append(prob4)
        
        # Stack all predictions
        all_preds = torch.stack(predictions, dim=1)  # (batch, 4, 2)
        all_probs = torch.stack(probabilities, dim=1)  # (batch, 4, 2)
        
        # Method 1: Weighted average
        weights = F.softmax(self.ensemble_weights, dim=0)
        weighted_pred = torch.sum(all_preds * weights.view(1, 4, 1), dim=1)
        
        # Method 2: Meta-classifier
        meta_features = all_probs.view(all_probs.size(0), -1)  # (batch, 8)
        meta_pred = self.meta_classifier(meta_features)
        
        # Combine both methods
        final_pred = 0.7 * weighted_pred + 0.3 * meta_pred
        
        return final_pred, all_probs, weights

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_model(model_type, **kwargs):
    """Factory function to create models"""
    if model_type == 'image_densenet':
        return ImageOnlyDenseNet(**kwargs)
    elif model_type == 'image_efficientnet':
        return ImageOnlyEfficientNet(**kwargs)
    elif model_type == 'multimodal_densenet':
        return MultimodalDenseNet(**kwargs)
    elif model_type == 'multimodal_efficientnet':
        return MultimodalEfficientNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    batch_size = 2
    images = torch.randn(batch_size, 3, 3, 224, 224)  # 3 images per study
    text_input_ids = torch.randint(0, 1000, (batch_size, 512))
    text_attention_mask = torch.ones(batch_size, 512)
    
    # Test image-only models
    print("Testing Image-Only Models:")
    
    model1 = ImageOnlyDenseNet()
    output1 = model1(images)
    print(f"DenseNet output shape: {output1.shape}")
    
    model2 = ImageOnlyEfficientNet()
    output2 = model2(images)
    print(f"EfficientNet output shape: {output2.shape}")
    
    # Test multimodal models
    print("\nTesting Multimodal Models:")
    
    model3 = MultimodalDenseNet()
    output3 = model3(images, text_input_ids, text_attention_mask)
    print(f"Multimodal DenseNet output shape: {output3.shape}")
    
    model4 = MultimodalEfficientNet()
    output4 = model4(images, text_input_ids, text_attention_mask)
    print(f"Multimodal EfficientNet output shape: {output4.shape}")
    
    print("\nAll models created successfully!")
