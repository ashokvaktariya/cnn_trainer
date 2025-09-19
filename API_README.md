# 🏥 Medical Fracture Detection API

## Overview
This FastAPI application provides binary classification for detecting fractures in medical images. It supports both local and Hugging Face Hub model loading with a comprehensive web interface.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Download Model (Optional - API will auto-download)
```bash
python download_model.py
```

### 3. Run the API
```bash
python run_api.py
```

**Expected Output:**
```
🚀 Starting Medical Fracture Detection API...
📱 Hugging Face Repository: 5techlab-research/medical-fracture-detection-v1
🌐 API will be available at: http://localhost:8000
📤 Upload interface: http://localhost:8000/upload_interface
📚 API Documentation: http://localhost:8000/api_docs
============================================================
🔍 Checking model availability...
✅ Model found locally: 748.73 MB
🚀 Starting API server...
✅ Model loaded successfully from local directory!
✅ API ready for inference!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 4. Access the Interface
- **Web Interface**: http://localhost:8000/upload_interface
- **API Documentation**: http://localhost:8000/api_docs
- **Health Check**: http://localhost:8000/health
- **FastAPI Docs**: http://localhost:8000/docs

## 🔧 Configuration

### Model Storage
- **Local Directory**: `./models/`
- **Model File**: `binary_classifier_best.pth`
- **Full Path**: `./models/binary_classifier_best.pth`

### Hugging Face Hub Settings
```python
HF_USERNAME = "5techlab-research"
HF_REPO_NAME = "medical-fracture-detection-v1"
HF_REPO_ID = "5techlab-research/medical-fracture-detection-v1"
HF_TOKEN = os.getenv('HF_TOKEN', '')  # Get token from environment variable
```

**Setup Environment Variable:**
```bash
# Windows
set HF_TOKEN=your_hugging_face_token_here

# Linux/Mac
export HF_TOKEN=your_hugging_face_token_here
```

Get your token from: https://huggingface.co/settings/tokens

The API automatically downloads models from Hugging Face Hub to the local `./models/` directory for easy team sharing.

## 📡 API Endpoints

### Core Endpoints

#### `GET /`
- **Description**: Root endpoint with API status
- **Response**: API information and model status

#### `GET /health`
- **Description**: Health check endpoint
- **Response**: System health and model status

#### `GET /model_info`
- **Description**: Get detailed model information
- **Response**: Model source, device, HF repo info

#### `POST /load_model`
- **Description**: Load model from specified source
- **Parameters**: 
  - `source`: "auto", "local", or "huggingface"
- **Response**: Loading status and model details

### Inference Endpoints

#### `POST /predict_local`
- **Description**: Local inference on uploaded image
- **Input**: Image file (multipart/form-data)
- **Response**: Prediction results with local inference metadata

#### `POST /predict_global`
- **Description**: Global inference via HF Hub
- **Input**: Image file (multipart/form-data)
- **Response**: Prediction results with global inference metadata

#### `POST /predict`
- **Description**: Original single image prediction
- **Input**: Image file (multipart/form-data)
- **Response**: Standard prediction results

#### `POST /predict_batch`
- **Description**: Batch prediction (max 10 images)
- **Input**: Multiple image files
- **Response**: Batch prediction results

#### `POST /predict_base64`
- **Description**: Base64 encoded image prediction
- **Input**: Base64 encoded image string
- **Response**: Prediction results

### Interface Endpoints

#### `GET /upload_interface`
- **Description**: HTML web interface for image upload
- **Response**: Interactive web page for testing

#### `GET /api_docs`
- **Description**: Complete API documentation
- **Response**: JSON documentation of all endpoints

### Testing Endpoints

#### `GET /test_images`
- **Description**: List available test images
- **Response**: Available test image categories

#### `POST /test_inference`
- **Description**: Run inference on all test images
- **Response**: Comprehensive test results with accuracy metrics

## 🖼️ Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## 📊 Response Format

### Successful Prediction Response
```json
{
  "predicted_class": 1,
  "predicted_label": "POSITIVE",
  "confidence": 0.95,
  "probabilities": {
    "negative": 0.05,
    "positive": 0.95
  },
  "filename": "image.jpg",
  "file_size": 1024000,
  "content_type": "image/jpeg",
  "inference_type": "local",
  "model_source": "huggingface"
}
```

### Error Response
```json
{
  "detail": "Error message description"
}
```

## 🌐 Usage Examples

### 1. Using the Web Interface
1. Navigate to http://localhost:8000/upload_interface
2. Select an image file
3. Click "🔍 Local Inference" or "🌐 Global Inference"
4. View results in the interface

### 2. Using cURL
```bash
# Local inference
curl -X POST "http://localhost:8000/predict_local" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"

# Global inference
curl -X POST "http://localhost:8000/predict_global" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"
```

### 3. Using Python
```python
import requests

# Upload image for inference
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict_local', files=files)
    result = response.json()
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

## 🔄 Model Loading

The API supports three model loading modes:

1. **Auto** (default): Checks local directory first, downloads from HF Hub if needed
2. **Hugging Face**: Downloads from HF Hub to local directory
3. **Local**: Loads from local `./models/` directory

### Switching Model Sources
```bash
# Load from Hugging Face Hub
curl -X POST "http://localhost:8000/load_model?source=huggingface"

# Load from local file
curl -X POST "http://localhost:8000/load_model?source=local"

# Auto-load (HF first, then local)
curl -X POST "http://localhost:8000/load_model?source=auto"
```

## 🏗️ Architecture

### Model Architecture
- **Backbone**: EfficientNet-B7 (from efficientnet-pytorch)
- **Classes**: 2 (NEGATIVE, POSITIVE)
- **Input Size**: 224x224 pixels
- **Normalization**: ImageNet statistics
- **Model Size**: ~748 MB
- **Feature Dimension**: 2560

### Processing Pipeline
1. Image upload and validation
2. Preprocessing (resize, normalize)
3. Model inference
4. Post-processing (softmax, confidence)
5. Response formatting

## 🚨 Error Handling

The API includes comprehensive error handling for:
- Invalid file types
- Corrupted images
- Model loading failures
- Network issues
- Server errors

## 📈 Performance

- **Inference Time**: <1 second per image (CPU)
- **Batch Processing**: Up to 10 images
- **Model Size**: ~748 MB
- **Memory Usage**: Optimized for CPU (GPU support available)
- **Model Loading**: Automatic from HF Hub cache

## 🔒 Security Notes

- API runs on all interfaces (0.0.0.0)
- No authentication implemented
- Input validation for file types
- Error messages don't expose sensitive information

## 🛠️ Development

### Running in Development Mode
```bash
python run_api.py
```

### API Documentation
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **Custom Docs**: http://localhost:8000/api_docs

### Testing
```bash
# Test all endpoints
curl http://localhost:8000/health
curl http://localhost:8000/model_info
curl http://localhost:8000/api_docs

# Test inference with sample image
curl -X POST "http://localhost:8000/predict_local" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_images/test_images/sample.jpg"
```

### Troubleshooting
- **Model Loading Issues**: Check if `efficientnet-pytorch` is installed
- **HF Hub Access**: Verify internet connection for initial model download
- **Memory Issues**: Model requires ~1GB RAM for loading
- **Team Sharing**: Model is stored in `./models/binary_classifier_best.pth` for easy sharing
- **Manual Download**: Use `python download_model.py` to download model separately

## 📞 Support

For issues or questions:
1. Check the API documentation at `/api_docs`
2. Verify model loading at `/model_info`
3. Test with the web interface at `/upload_interface`
4. Check server logs for detailed error messages

---

**Happy Fracture Detection! 🎯**
