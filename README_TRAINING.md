# Medical Fracture Detection - Training & Inference System

## 🏥 Project Overview
Complete medical fracture detection system with binary classification, optimized for H100 GPU server with public API access.

## 🚀 Quick Start

### 1. Server Setup (H100)
```bash
# Install dependencies
pip install -r requirements_training.txt

# Run setup
python setup_training.py

# Analyze data
python data_overview.py

# Train model
python train_model.py
```

### 2. API Deployment
```bash
# Deploy for public access
python deploy_api.py

# Or run locally
python inference_api.py
```

## 📁 File Structure

```
MONAI/
├── config_training.yaml          # Training configuration
├── requirements_training.txt     # Python dependencies
├── data_overview.py             # Data analysis script
├── train_model.py               # Main training script
├── inference_api.py             # FastAPI inference server
├── deploy_api.py                # Deployment script
├── setup_training.py            # Setup script
└── README_TRAINING.md           # This file
```

## ⚙️ Configuration

### Server Configuration (config_training.yaml)
```yaml
# Data paths for H100 server
data:
  csv_path: "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/processed_dicom_image_url_file.csv"
  image_root: "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/"
  output_dir: "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/training_outputs/"

# H100 optimizations
hardware:
  device: "cuda"
  mixed_precision: true
  compile_model: true
  use_torch_compile: true
```

## 📊 Data Analysis

The `data_overview.py` script provides comprehensive analysis:

### Features:
- ✅ CSV file validation and loading
- ✅ Column analysis and data types
- ✅ Label distribution analysis
- ✅ Image path validation
- ✅ Dataset statistics and visualizations
- ✅ Comprehensive report generation

### Usage:
```bash
python data_overview.py
```

### Output:
- `training_outputs/data_analysis_report.txt` - Detailed analysis report
- `training_outputs/visualizations/` - Charts and plots
- Dataset statistics and recommendations

## 🎯 Model Training

The `train_model.py` script provides H100-optimized training:

### Features:
- ✅ EfficientNet-B0 architecture
- ✅ Mixed precision training (AMP)
- ✅ Data augmentation
- ✅ Early stopping and checkpointing
- ✅ Wandb integration
- ✅ Comprehensive evaluation

### Usage:
```bash
python train_model.py
```

### Output:
- `training_outputs/best_model.pth` - Best model weights
- `training_outputs/checkpoints/` - Training checkpoints
- `training_outputs/logs/` - Training logs

## 🌐 API Inference

The `inference_api.py` provides a complete FastAPI server:

### Endpoints:
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch prediction (max 10 images)
- `POST /predict_base64` - Base64 encoded image
- `GET /model_info` - Model information
- `GET /config` - Configuration

### Usage:
```bash
# Local development
python inference_api.py

# Production
uvicorn inference_api:app --host 0.0.0.0 --port 8000
```

## 🚀 Public Deployment

The `deploy_api.py` script supports multiple platforms:

### Supported Platforms:
1. **Railway** - Modern cloud platform
2. **Heroku** - Popular PaaS
3. **Render** - Simple deployment
4. **Local + ngrok** - Public tunnel

### Usage:
```bash
# Interactive deployment
python deploy_api.py

# Direct deployment
python deploy_api.py railway
python deploy_api.py heroku
python deploy_api.py render
python deploy_api.py local
```

## 📋 API Usage Examples

### Single Image Prediction
```bash
curl -X POST "http://your-api-url/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

### Batch Prediction
```bash
curl -X POST "http://your-api-url/predict_batch" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

### Base64 Image
```bash
curl -X POST "http://your-api-url/predict_base64" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "image_data=base64_encoded_string"
```

## 📊 Response Format

```json
{
  "predicted_class": 0,
  "predicted_label": "NEGATIVE",
  "confidence": 0.513,
  "probabilities": {
    "negative": 0.513,
    "positive": 0.487
  },
  "filename": "image.jpg",
  "file_size": 12345,
  "content_type": "image/jpeg"
}
```

## 🔧 Hardware Requirements

### Training (H100 Server):
- **GPU**: NVIDIA H100 (95GB VRAM)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for dataset
- **CUDA**: 12.0+

### Inference:
- **GPU**: Any CUDA-compatible GPU (optional)
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for model

## 📈 Performance Metrics

### Training Performance (H100):
- **Batch Size**: 32
- **Training Time**: ~2-3 hours for 50 epochs
- **Memory Usage**: ~40GB VRAM
- **Throughput**: ~1000 images/minute

### Inference Performance:
- **Single Image**: ~1-2 seconds
- **Batch (10 images)**: ~5-10 seconds
- **Model Size**: ~100MB
- **Memory Usage**: ~2GB VRAM

## 🛠️ Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Model Loading Failed**
   - Check model path in config
   - Verify model architecture matches
   - Ensure model file exists

3. **Data Loading Errors**
   - Validate CSV file path
   - Check image file permissions
   - Verify image formats

4. **API Deployment Issues**
   - Check platform requirements
   - Verify environment variables
   - Ensure all dependencies installed

## 📞 Support

For issues or questions:
1. Check the generated data analysis report
2. Review training logs in `training_outputs/logs/`
3. Verify configuration in `config_training.yaml`
4. Test API endpoints with sample images

## 🎉 Success Indicators

✅ **Training Complete When:**
- Best model saved to `training_outputs/best_model.pth`
- Validation accuracy > 80%
- Training loss decreasing
- No early stopping triggered

✅ **API Ready When:**
- Health check returns 200
- Model loaded successfully
- Predictions working
- Public URL accessible

---

**Ready to detect fractures in medical images! 🏥✨**
