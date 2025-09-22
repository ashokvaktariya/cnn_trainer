# Data Preparation Script - Server Ready

## 🎯 **Updated Features:**

### **1. Image Path Configuration:**
- **Server Path**: `/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/gleamer-images`
- **Automatic Search**: Recursively searches in gleamer-images folder
- **Filename Matching**: Extracts exact filenames from download_urls

### **2. Dataset Processing:**
- **Input**: `training_dataset.csv` (filtered X-ray dataset)
- **Output**: `train_dataset.csv` and `val_dataset.csv` (85:15 split)
- **No Class Balancing**: Uses all available samples
- **Confidence Filtering**: Optional (if confidence column exists)

### **3. URL Parsing:**
- **JSON Array**: `["url1", "url2", "url3"]`
- **Python List**: `['url1', 'url2', 'url3']`
- **Comma Separated**: `url1, url2, url3`
- **Filename Extraction**: `url.split('/')[-1]`

### **4. Image Validation:**
- **Blank Detection**: Removes images with low variance
- **Format Support**: JPG, JPEG, PNG, DCM, DICOM
- **Error Handling**: Graceful handling of corrupted images

## 🚀 **Server Execution:**

```bash
# On H100 server
cd /path/to/project
python data_preparation.py
```

## 📊 **Expected Output:**
- `train_dataset.csv` - Training data (85%)
- `val_dataset.csv` - Validation data (15%)
- `dataset_stats.json` - Processing statistics
- `data_preparation_report.png` - Visualization (if matplotlib available)

## 🔧 **Configuration:**
- **CSV Path**: `training_dataset.csv`
- **Image Root**: `/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images`
- **Output Dir**: `./training_outputs/`
- **Train Ratio**: 85%
- **Val Ratio**: 15%

## ✅ **Ready for Training:**
The script will create properly formatted train/validation datasets ready for your fracture detection model training on H100!
