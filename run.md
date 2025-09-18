# 🚀 Binary Medical Image Classification - Server Run Guide

## 📋 Overview
This guide provides step-by-step commands to run the complete binary classification pipeline on your H200 GPU server for fracture detection in medical images.

## 🎯 Target Performance
- **Accuracy**: 85-90%
- **Training Time**: 2-3 hours on H200
- **Model Size**: ~100MB
- **Classes**: POSITIVE (fracture) / NEGATIVE (no fracture)

---

## 🔧 Server Setup & Preparation

### 1. **Clone Repository**
```bash
cd /sharedata01/CNN_data/
git clone https://github.com/ashokvaktariya/cnn_trainer.git
cd cnn_trainer
```

### 2. **Install Dependencies**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai[all]
pip install transformers
pip install efficientnet-pytorch
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install opencv-python
pip install pillow
pip install pandas
pip install numpy
pip install tqdm
pip install tensorboard
```

### 3. **Verify Data Structure**
```bash
# Check data paths
ls -la /sharedata01/CNN_data/gleamer/gleamer/
ls -la /sharedata01/CNN_data/gleamer/gleamer/dicom_image_url_file.csv

# Expected structure:
# /sharedata01/CNN_data/gleamer/gleamer/
# ├── dicom_image_url_file.csv
# ├── 2.25.98756531213685189148264762540282984870.jpg
# ├── 2.25.98125395115665805448452401418999617883.jpg
# └── ... (70,000+ images)
```

---

## 📊 Data Preparation

### 1. **Run Data Preparation**
```bash
# Full dataset processing (recommended)
python3 data_preparation.py

# Or test with sample first
python3 data_preparation.py --sample_size 1000
```

**Expected Output:**
- Clean dataset: `/sharedata01/CNN_data/medical_classification/preprocessed/binary_medical_dataset.csv`
- Statistics: `/sharedata01/CNN_data/medical_classification/preprocessed/binary_medical_dataset_stats.json`
- Dataset size: ~11,000 balanced records

### 2. **Verify Data Preparation**
```bash
# Check output files
ls -la /sharedata01/CNN_data/medical_classification/preprocessed/

# View statistics
cat /sharedata01/CNN_data/medical_classification/preprocessed/binary_medical_dataset_stats.json
```

---

## 🏋️ Model Training

### 1. **Start Training**
```bash
# Full training with monitoring
python3 train_models.py

# Training with custom config
python3 train_models.py --config config.py
```

**Expected Training Process:**
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Model**: EfficientNet-B7
- **Loss**: Focal Loss (handles class imbalance)
- **Monitoring**: Validation accuracy

### 2. **Monitor Training**
```bash
# View training logs
tail -f /sharedata01/CNN_data/medical_classification/logs/training.log

# Check TensorBoard (if enabled)
tensorboard --logdir=/sharedata01/CNN_data/medical_classification/logs/
```

### 3. **Training Outputs**
```bash
# Check model checkpoints
ls -la /sharedata01/CNN_data/medical_classification/models/

# Expected files:
# ├── binary_classifier_best_model.pth
# ├── binary_classifier_final_model.pth
# └── binary_classifier_training_results.json
```

---

## 📈 Model Evaluation

### 1. **Run Evaluation**
```bash
# Evaluate best model
python3 evaluate_model.py --model_path /sharedata01/CNN_data/medical_classification/models/binary_classifier_best_model.pth

# Evaluate specific checkpoint
python3 evaluate_model.py --model_path /sharedata01/CNN_data/medical_classification/models/binary_classifier_final_model.pth
```

### 2. **Evaluation Outputs**
```bash
# Check evaluation results
ls -la /sharedata01/CNN_data/medical_classification/results/

# Expected files:
# ├── binary_classifier_evaluation_results.json
# ├── confusion_matrix.png
# ├── roc_curve.png
# └── predictions.csv
```

### 3. **View Results**
```bash
# View evaluation metrics
cat /sharedata01/CNN_data/medical_classification/results/binary_classifier_evaluation_results.json

# View confusion matrix
display /sharedata01/CNN_data/medical_classification/results/confusion_matrix.png
```

---

## 🔮 Model Inference

### 1. **Single Image Inference**
```bash
# Test on single image
python3 inference.py --image_path /path/to/test/image.jpg --model_path /sharedata01/CNN_data/medical_classification/models/binary_classifier_best_model.pth

# Example with actual image
python3 inference.py --image_path /sharedata01/CNN_data/gleamer/gleamer/2.25.98756531213685189148264762540282984870.jpg --model_path /sharedata01/CNN_data/medical_classification/models/binary_classifier_best_model.pth
```

### 2. **Batch Inference**
```bash
# Process multiple images
python3 inference.py --batch_mode --input_dir /path/to/test/images/ --model_path /sharedata01/CNN_data/medical_classification/models/binary_classifier_best_model.pth --output_dir /sharedata01/CNN_data/medical_classification/inference_results/
```

### 3. **Export to ONNX**
```bash
# Export model for deployment
python3 inference.py --export_onnx --model_path /sharedata01/CNN_data/medical_classification/models/binary_classifier_best_model.pth --onnx_path /sharedata01/CNN_data/medical_classification/models/binary_classifier.onnx
```

---

## 🚀 Complete Pipeline Run

### **One-Command Execution**
```bash
# Run complete pipeline
bash -c '
echo "🚀 Starting Binary Classification Pipeline..."
echo "📊 Step 1: Data Preparation"
python3 data_preparation.py
echo "🏋️ Step 2: Model Training"
python3 train_models.py
echo "📈 Step 3: Model Evaluation"
python3 evaluate_model.py --model_path /sharedata01/CNN_data/medical_classification/models/binary_classifier_best_model.pth
echo "🔮 Step 4: Test Inference"
python3 inference.py --image_path /sharedata01/CNN_data/gleamer/gleamer/2.25.98756531213685189148264762540282984870.jpg --model_path /sharedata01/CNN_data/medical_classification/models/binary_classifier_best_model.pth
echo "✅ Pipeline Complete!"
'
```

---

## 📁 Directory Structure After Run

```
/sharedata01/CNN_data/medical_classification/
├── preprocessed/
│   ├── binary_medical_dataset.csv
│   └── binary_medical_dataset_stats.json
├── models/
│   ├── binary_classifier_best_model.pth
│   ├── binary_classifier_final_model.pth
│   ├── binary_classifier_training_results.json
│   └── binary_classifier.onnx
├── results/
│   ├── binary_classifier_evaluation_results.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── predictions.csv
├── logs/
│   └── training.log
└── inference_results/
    └── batch_predictions.csv
```

---

## 🔍 Troubleshooting

### **Common Issues:**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config.py
   TRAINING_CONFIG['batch_size'] = 16
   ```

2. **Data Loading Errors**
   ```bash
   # Check image paths
   python3 -c "import pandas as pd; df = pd.read_csv('/sharedata01/CNN_data/gleamer/gleamer/dicom_image_url_file.csv'); print(df.head())"
   ```

3. **Model Loading Issues**
   ```bash
   # Verify model file
   ls -la /sharedata01/CNN_data/medical_classification/models/
   ```

4. **Permission Errors**
   ```bash
   # Fix permissions
   chmod -R 755 /sharedata01/CNN_data/medical_classification/
   ```

---

## 📊 Expected Results

### **Training Metrics:**
- **Training Accuracy**: 90-95%
- **Validation Accuracy**: 85-90%
- **Training Loss**: <0.1
- **Validation Loss**: <0.2

### **Evaluation Metrics:**
- **Test Accuracy**: 85-90%
- **Precision**: 0.85-0.90
- **Recall**: 0.85-0.90
- **F1-Score**: 0.85-0.90
- **ROC AUC**: 0.90-0.95

### **Inference Performance:**
- **Single Image**: <1 second
- **Batch Processing**: 100+ images/minute
- **Model Size**: ~100MB

---

## 🎯 Success Criteria

✅ **Pipeline Complete When:**
- [ ] Data preparation finished (11,000+ records)
- [ ] Training completed (50 epochs)
- [ ] Evaluation accuracy >85%
- [ ] Inference working on test images
- [ ] Model exported to ONNX

✅ **Ready for Production When:**
- [ ] Test accuracy >85%
- [ ] Confusion matrix shows good separation
- [ ] ROC curve AUC >0.90
- [ ] Model size <200MB
- [ ] Inference time <2 seconds per image

---

## 📞 Support

**For Issues:**
1. Check logs: `/sharedata01/CNN_data/medical_classification/logs/`
2. Verify data paths in `config.py`
3. Ensure GPU memory availability
4. Check file permissions

**Happy Training! 🎉**
