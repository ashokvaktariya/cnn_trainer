#!/usr/bin/env python3
"""
Configuration file for Medical Image Classification
H200 GPU Server Setup
"""

import os
from pathlib import Path

# =============================================================================
# SERVER PATHS CONFIGURATION
# =============================================================================

# Server data paths - assume we're running on server
SERVER_DATA_ROOT = "/sharedata01/CNN_data"
GLEAMER_DATA_ROOT = "/sharedata01/CNN_data/gleamer/gleamer"

# Dataset paths - use existing gleamer data
CSV_FILE = os.path.join(GLEAMER_DATA_ROOT, "dicom_image_url_file.csv")
DATA_ROOT = SERVER_DATA_ROOT  # For our processed data
EXISTING_IMAGES_DIR = GLEAMER_DATA_ROOT  # Where images are already stored

print(f"✅ Using server data path: {DATA_ROOT}")
print(f"✅ Using existing images from: {EXISTING_IMAGES_DIR}")
print(f"✅ CSV file location: {CSV_FILE}")

# Output directories
OUTPUT_ROOT = os.path.join(DATA_ROOT, "medical_classification")
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

# Model types to train
MODELS_TO_TRAIN = {
    "image_densenet": True,
    "image_efficientnet": True,
    "multimodal_densenet": True,
    "multimodal_efficientnet": True,
    "ensemble": True
}

# Model-specific configurations
MODEL_CONFIGS = {
    "image_densenet": {
        "model_type": "ImageOnlyDenseNet",
        "pretrained": True,
        "freeze_backbone": False,
        "dropout_rate": 0.3,
        "feature_dim": 1024
    },
    "image_efficientnet": {
        "model_type": "ImageOnlyEfficientNet", 
        "model_name": "efficientnet-b0",
        "pretrained": True,
        "freeze_backbone": False,
        "dropout_rate": 0.3,
        "feature_dim": 1280
    },
    "multimodal_densenet": {
        "model_type": "MultimodalDenseNet",
        "text_model": "bert-base-uncased",
        "pretrained": True,
        "freeze_backbone": False,
        "freeze_text_model": False,
        "dropout_rate": 0.3,
        "fusion_dim": 512
    },
    "multimodal_efficientnet": {
        "model_type": "MultimodalEfficientNet",
        "text_model": "bert-base-uncased",
        "model_name": "efficientnet-b0",
        "pretrained": True,
        "freeze_backbone": False,
        "freeze_text_model": False,
        "dropout_rate": 0.3,
        "fusion_dim": 512
    }
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# General training parameters
TRAINING_CONFIG = {
    # Data parameters
    "batch_size": 16,  # Reduced for H200 memory optimization
    "num_workers": 8,  # H200 server can handle more workers
    "max_images_per_study": 3,
    "text_max_length": 512,
    
    # Training parameters
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 5,
    
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
    "monitor_metric": "val_auc",
    "mode": "max"  # max for AUC, min for loss
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
# ENSEMBLE CONFIGURATION
# =============================================================================

ENSEMBLE_CONFIG = {
    "methods": ["voting", "averaging", "weighted", "stacking"],
    "voting_method": "soft",  # soft or hard
    "stacking_cv_folds": 5,
    "weight_optimization": True,
    "meta_learner": "logistic_regression"
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
    print("🔧 Medical Image Classification Configuration")
    print("=" * 50)
    print(f"📁 Data Root: {DATA_ROOT}")
    print(f"📊 CSV File: {CSV_FILE}")
    print(f"💾 Output Root: {OUTPUT_ROOT}")
    print(f"🖥️  Device: {GPU_CONFIG['device']}")
    print(f"📦 Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"🔄 Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"📚 Models to Train: {list(MODELS_TO_TRAIN.keys())}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
