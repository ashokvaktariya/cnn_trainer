#!/usr/bin/env python3
"""
Configuration file for Binary Medical Image Classification
H200 GPU Server Setup - Fracture Detection
"""

import os
from pathlib import Path

# =============================================================================
# SERVER PATHS CONFIGURATION
# =============================================================================

# Server data paths - H100 GPU Server
SERVER_DATA_ROOT = "/sharedata01/CNN_data"
GLEAMER_DATA_ROOT = "/sharedata01/CNN_data/gleamer/gleamer"

# Output directories (define first)
DATA_ROOT = SERVER_DATA_ROOT  # For our processed data
OUTPUT_ROOT = os.path.join(DATA_ROOT, "medical_classification")

# Dataset paths - use preprocessed data
CSV_FILE = os.path.join(OUTPUT_ROOT, "preprocessed", "binary_medical_dataset.csv")
EXISTING_IMAGES_DIR = GLEAMER_DATA_ROOT  # Where images are already stored

print(f"✅ Using server data path: {DATA_ROOT}")
print(f"✅ Using existing images from: {EXISTING_IMAGES_DIR}")
print(f"✅ CSV file location: {CSV_FILE}")
CHECKPOINTS_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
LOGS_DIR = os.path.join(OUTPUT_ROOT, "logs")
PREPROCESSED_DIR = os.path.join(OUTPUT_ROOT, "preprocessed")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")

# Create directories if they don't exist
for dir_path in [OUTPUT_ROOT, CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, PREPROCESSED_DIR, IMAGES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Binary Classification Model
MODELS_TO_TRAIN = {
    "binary_classifier": True,  # Single binary classifier
}

# Binary Classification Model Configuration
MODEL_CONFIGS = {
    "binary_classifier": {
        "model_type": "BinaryEfficientNet",
        "model_name": "efficientnet-b7",  # Best performance model
        "pretrained": True,
        "freeze_backbone": False,
        "dropout_rate": 0.3,
        "num_classes": 2,  # POSITIVE, NEGATIVE
        "class_names": ["NEGATIVE", "POSITIVE"],
        "use_focal_loss": True,  # Handle class imbalance
        "focal_alpha": 0.25,
        "focal_gamma": 2.0
    }
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Binary Classification Training Configuration
TRAINING_CONFIG = {
    # Data parameters
    "batch_size": 32,  # Optimized for binary classification
    "num_workers": 8,  # H200 server can handle more workers
    "use_valid_images_only": True,  # Filter out blank images
    "exclude_doubt_cases": True,  # Remove DOUBT cases
    
    # Training parameters
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 5,
    
    # Class balancing
    "balance_classes": True,  # Balance POSITIVE/NEGATIVE
    "class_weights": [1.0, 1.2],  # Weight NEGATIVE slightly more
    "augment_positive_class": True,  # Heavy augmentation for POSITIVE
    
    # Optimization
    "use_amp": True,  # Automatic Mixed Precision for H200
    "gradient_clipping": 1.0,
    "scheduler": "cosine",  # cosine, step, plateau
    
    # Early stopping
    "patience": 10,
    "min_delta": 0.001,
    
    # Validation
    "val_split": 0.2,
    "test_split": 0.1,
    
    # Checkpointing
    "save_every_n_epochs": 5,
    "save_best_only": True,
    "monitor_metric": "val_accuracy",  # Monitor accuracy for binary classification
    "mode": "max"  # max for accuracy
}

# H200 GPU specific settings
GPU_CONFIG = {
    "device": "cuda:0",  # H200 GPU
    "mixed_precision": True,
    "compile_model": True,  # PyTorch 2.0 compilation for H200
    "num_gpus": 1,
    "memory_fraction": 0.9
}

# =============================================================================
# DATA PREPROCESSING CONFIGURATION
# =============================================================================

PREPROCESSING_CONFIG = {
    # Image preprocessing
    "image_size": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet stats
    "normalize_std": [0.229, 0.224, 0.225],
    
    # Augmentation settings
    "use_augmentation": True,
    "augmentation_prob": 0.5,
    
    # Data validation
    "min_image_size": (32, 32),
    "max_image_size": (1024, 1024),
    "skip_corrupt_images": True,
    "max_retries": 3,
    
    # Caching
    "cache_preprocessed": True,
    "cache_dir": PREPROCESSED_DIR,
    
    # Use existing images (skip download)
    "use_existing_images": True
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVALUATION_CONFIG = {
    "metrics": ["accuracy", "auc", "precision", "recall", "f1"],
    "confidence_threshold": 0.5,
    "save_predictions": True,
    "save_probabilities": True,
    "create_confusion_matrix": True,
    "create_roc_curve": True
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_to_file": True,
    "log_file": os.path.join(LOGS_DIR, "training.log"),
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "tensorboard_logs": True,
    "tensorboard_dir": os.path.join(LOGS_DIR, "tensorboard")
}

# =============================================================================
# BINARY CLASSIFICATION SPECIFIC CONFIG
# =============================================================================

BINARY_CONFIG = {
    # Label mapping
    "label_mapping": {
        "NEGATIVE": 0,
        "POSITIVE": 1
    },
    "exclude_labels": ["DOUBT"],  # Exclude uncertain cases
    
    # Performance targets
    "target_accuracy": 0.85,  # 85% accuracy target
    "target_precision": 0.80,  # 80% precision for POSITIVE
    "target_recall": 0.80,    # 80% recall for POSITIVE
    
    # Inference settings
    "confidence_threshold": 0.5,
    "top_k_predictions": 2,
    "save_predictions": True
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_path(model_name, epoch=None, best=False):
    """Get model checkpoint path"""
    if best:
        return os.path.join(CHECKPOINTS_DIR, f"{model_name}_best.pth")
    elif epoch:
        return os.path.join(CHECKPOINTS_DIR, f"{model_name}_epoch_{epoch}.pth")
    else:
        return os.path.join(CHECKPOINTS_DIR, f"{model_name}_latest.pth")

def get_results_path(model_name, metric_name):
    """Get results file path"""
    return os.path.join(RESULTS_DIR, f"{model_name}_{metric_name}.json")

def print_config():
    """Print current configuration"""
    print("🔧 Binary Medical Image Classification Configuration")
    print("=" * 60)
    print(f"📁 Data Root: {DATA_ROOT}")
    print(f"📊 CSV File: {CSV_FILE}")
    print(f"💾 Output Root: {OUTPUT_ROOT}")
    print(f"🖥️  Device: {GPU_CONFIG['device']}")
    print(f"📦 Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"🔄 Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"🎯 Model: {MODEL_CONFIGS['binary_classifier']['model_name']}")
    print(f"🏷️  Classes: {MODEL_CONFIGS['binary_classifier']['num_classes']} (POSITIVE/NEGATIVE)")
    print(f"📊 Target Accuracy: {BINARY_CONFIG['target_accuracy']*100}%")
    print(f"⚖️  Class Balancing: {TRAINING_CONFIG['balance_classes']}")
    print(f"🚫 Exclude DOUBT: {TRAINING_CONFIG['exclude_doubt_cases']}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
