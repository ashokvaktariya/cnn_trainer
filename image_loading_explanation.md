# How Images Are Loaded Using CSV Files

## Overview
The system uses CSV files to map image filenames to labels and metadata, then loads the actual image files from the filesystem.

## CSV Structure

### Example CSV Row:
```csv
image_path,label,sop_instance_uid,StudyDescription,SeriesDescription,Modality,BodyPartExamined,Manufacturer,is_report,binary_label,file_path
1.2.840.113619.2.203.4.2147483647.1756929960.26353.jpg,POSITIVE,1.2.840.113619.2.203.4.2147483647.1756929960.26353,WRIST LEFT,Wrist,DX,WRIST,"GE Healthcare",False,1,1.2.840.113619.2.203.4.2147483647.1756929960.26353.jpg
```

### Key Columns:
- **`image_path`**: The filename of the image (e.g., `1.2.840.113619.2.203.4.2147483647.1756929960.26353.jpg`)
- **`label`**: Text label (POSITIVE/NEGATIVE)
- **`binary_label`**: Numeric label (1/0)
- **`file_path`**: Duplicate of image_path for compatibility

## Image Loading Process

### Step 1: CSV Reading
```python
# In BinaryMedicalDataset.__init__()
self.data = pd.read_csv(csv_file)  # Load CSV into DataFrame
```

### Step 2: Data Processing
```python
# In _process_data()
if 'label' in self.data.columns:
    self.data = self.data[self.data['label'].isin(['POSITIVE', 'NEGATIVE'])]
    self.label_column = 'label'
    self.image_column = 'image_path'  # Column containing image filenames
```

### Step 3: Image Loading (per sample)
```python
# In __getitem__(idx)
row = self.data.iloc[idx]  # Get row from CSV

# Get image filename from CSV
image_filename = str(row[self.image_column])  # e.g., "1.2.840.113619.2.203.4.2147483647.1756929960.26353.jpg"

# Find actual image file on filesystem
image_path = self._find_image_file_by_filename(image_filename)
```

### Step 4: File System Search
```python
def _find_image_file_by_filename(self, filename):
    base_dir = config['data']['image_root']  # e.g., "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/"
    
    # Check main directory
    image_path = os.path.join(base_dir, filename)
    if os.path.exists(image_path):
        return image_path
    
    # Check subdirectories
    for subdir in ['images', 'data', 'positive', 'negative', 'Negetive', 'Negative 2', 'fracture', 'normal']:
        image_path = os.path.join(base_dir, subdir, filename)
        if os.path.exists(image_path):
            return image_path
    
    return None
```

### Step 5: Image Validation
```python
def _is_valid_image(self, image_path):
    # Check if image is not blank/corrupted
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Check variance - very low variance indicates blank image
        variance = np.var(img_array)
        if variance < 10:
            return False
        
        return True
```

### Step 6: Image Loading and Preprocessing
```python
# Load image
with Image.open(image_path) as img:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    image = np.array(img)

# Apply transforms (resize, normalize, etc.)
if self.transform:
    image = Image.fromarray(image.astype(np.uint8), 'RGB')
    image = self.transform(image)  # Convert to tensor, normalize, etc.
```

## Complete Flow Diagram

```
CSV File → DataFrame → Row Selection → Image Filename → File System Search → Image Loading → Validation → Preprocessing → Tensor
```

### Detailed Steps:

1. **CSV Reading**: `pd.read_csv()` loads the CSV file
2. **Row Selection**: `self.data.iloc[idx]` gets specific row
3. **Filename Extraction**: `row['image_path']` gets image filename
4. **File Search**: `_find_image_file_by_filename()` searches filesystem
5. **Validation**: `_is_valid_image()` checks if image is valid
6. **Loading**: `Image.open()` loads the image
7. **Preprocessing**: Transforms convert to tensor format

## Configuration

### Image Root Directory:
```yaml
# In config_training.yaml
data:
  image_root: "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/"
```

### Search Locations:
The system searches for images in:
- Main directory: `/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/`
- Subdirectories: `images/`, `data/`, `positive/`, `negative/`, `Negetive/`, `Negative 2/`, `fracture/`, `normal/`

## Error Handling

### Missing Images:
```python
if image_path is None or not self._is_valid_image(image_path):
    # Skip this sample and try next one
    return self.__getitem__((idx + 1) % len(self.data))
```

### Invalid Images:
- Blank images (all pixels same color)
- Low variance images (variance < 10)
- Corrupted files

## Example Walkthrough

### Input CSV Row:
```csv
1.2.840.113619.2.203.4.2147483647.1756929960.26353.jpg,POSITIVE,1.2.840.113619.2.203.4.2147483647.1756929960.26353,WRIST LEFT,Wrist,DX,WRIST,"GE Healthcare",False,1
```

### Processing:
1. **Extract filename**: `1.2.840.113619.2.203.4.2147483647.1756929960.26353.jpg`
2. **Search path**: `/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/1.2.840.113619.2.203.4.2147483647.1756929960.26353.jpg`
3. **Load image**: `Image.open(image_path)`
4. **Validate**: Check if not blank/corrupted
5. **Preprocess**: Resize to 600x600, normalize, convert to tensor
6. **Return**: `{'image': tensor, 'label': 1, 'uid': filename, 'gleamer_finding': 'POSITIVE'}`

## Key Benefits

1. **Flexibility**: CSV can be easily modified without changing code
2. **Metadata**: Rich metadata stored alongside image paths
3. **Validation**: Built-in image validation prevents bad data
4. **Error Recovery**: Skips invalid images automatically
5. **Scalability**: Can handle large datasets efficiently

This approach separates the data organization (CSV) from the actual image storage (filesystem), making it easy to manage and modify datasets.
