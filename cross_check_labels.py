import pandas as pd
import numpy as np

print("=== CROSS-CHECKING LABELS BETWEEN DATASETS ===")

# Load both datasets
final_dataset = pd.read_csv('final_dataset.csv')
final_dataset_cnn = pd.read_csv('test/data/final_dataset_cnn.csv')

print(f"Final dataset shape: {final_dataset.shape}")
print(f"Final dataset CNN shape: {final_dataset_cnn.shape}")

# Check column names
print(f"\nFinal dataset columns: {list(final_dataset.columns)}")
print(f"Final dataset CNN columns: {list(final_dataset_cnn.columns)}")

# Check label distribution in final_dataset
print(f"\nFinal dataset label distribution:")
label_counts = final_dataset['class'].value_counts()
print(label_counts)

# Check label distribution in final_dataset_cnn
print(f"\nFinal dataset CNN label distribution:")
gleamer_counts = final_dataset_cnn['gleamer_finding'].value_counts()
print(gleamer_counts)

# Create mapping from final_dataset
final_dataset_mapping = dict(zip(final_dataset['image_path'], final_dataset['class']))
print(f"\nCreated mapping for {len(final_dataset_mapping)} images from final_dataset")

# Check how many images from final_dataset_cnn are in final_dataset
cnn_images = set(final_dataset_cnn['jpg_filename'])
final_images = set(final_dataset['image_path'])

common_images = cnn_images.intersection(final_images)
print(f"\nCommon images between datasets: {len(common_images)}")
print(f"Images only in CNN dataset: {len(cnn_images - final_images)}")
print(f"Images only in final dataset: {len(final_images - cnn_images)}")

# Check for label differences
print(f"\n=== CHECKING FOR LABEL DIFFERENCES ===")
differences = []
updated_count = 0

for idx, row in final_dataset_cnn.iterrows():
    image_name = row['jpg_filename']
    current_label = row['gleamer_finding']
    
    if image_name in final_dataset_mapping:
        final_label = final_dataset_mapping[image_name]
        
        # Convert final_label to match gleamer_finding format
        if final_label == 'positive':
            final_label_formatted = 'POSITIVE'
        elif final_label == 'negative':
            final_label_formatted = 'NEGATIVE'
        else:
            final_label_formatted = final_label.upper()
        
        if current_label != final_label_formatted:
            differences.append({
                'image': image_name,
                'current_label': current_label,
                'final_label': final_label_formatted
            })
            updated_count += 1

print(f"Found {len(differences)} label differences")

if differences:
    print(f"\nFirst 10 differences:")
    for i, diff in enumerate(differences[:10]):
        print(f"  {i+1}. {diff['image']}: {diff['current_label']} -> {diff['final_label']}")
    
    # Update the labels in final_dataset_cnn
    print(f"\n=== UPDATING LABELS ===")
    
    for idx, row in final_dataset_cnn.iterrows():
        image_name = row['jpg_filename']
        
        if image_name in final_dataset_mapping:
            final_label = final_dataset_mapping[image_name]
            
            # Convert final_label to match gleamer_finding format
            if final_label == 'positive':
                final_label_formatted = 'POSITIVE'
            elif final_label == 'negative':
                final_label_formatted = 'NEGATIVE'
            else:
                final_label_formatted = final_label.upper()
            
            # Update the label
            final_dataset_cnn.at[idx, 'gleamer_finding'] = final_label_formatted
            
            # Update binary_label accordingly
            if final_label_formatted == 'POSITIVE':
                final_dataset_cnn.at[idx, 'binary_label'] = 1
            elif final_label_formatted == 'NEGATIVE':
                final_dataset_cnn.at[idx, 'binary_label'] = 0
    
    # Save updated dataset
    final_dataset_cnn.to_csv('test/data/final_dataset_cnn_updated.csv', index=False)
    print(f"Updated dataset saved to: test/data/final_dataset_cnn_updated.csv")
    
    # Show updated distribution
    print(f"\nUpdated label distribution:")
    updated_counts = final_dataset_cnn['gleamer_finding'].value_counts()
    print(updated_counts)
    
    # Update train and validation files as well
    print(f"\n=== UPDATING TRAIN/VAL FILES ===")
    
    # Load train and val files
    train_data = pd.read_csv('test/data/final_dataset_cnn_train.csv')
    val_data = pd.read_csv('test/data/final_dataset_cnn_val.csv')
    
    # Update train data
    for idx, row in train_data.iterrows():
        image_name = row['jpg_filename']
        if image_name in final_dataset_mapping:
            final_label = final_dataset_mapping[image_name]
            if final_label == 'positive':
                final_label_formatted = 'POSITIVE'
            elif final_label == 'negative':
                final_label_formatted = 'NEGATIVE'
            else:
                final_label_formatted = final_label.upper()
            
            train_data.at[idx, 'gleamer_finding'] = final_label_formatted
            if final_label_formatted == 'POSITIVE':
                train_data.at[idx, 'binary_label'] = 1
            elif final_label_formatted == 'NEGATIVE':
                train_data.at[idx, 'binary_label'] = 0
    
    # Update val data
    for idx, row in val_data.iterrows():
        image_name = row['jpg_filename']
        if image_name in final_dataset_mapping:
            final_label = final_dataset_mapping[image_name]
            if final_label == 'positive':
                final_label_formatted = 'POSITIVE'
            elif final_label == 'negative':
                final_label_formatted = 'NEGATIVE'
            else:
                final_label_formatted = final_label.upper()
            
            val_data.at[idx, 'gleamer_finding'] = final_label_formatted
            if final_label_formatted == 'POSITIVE':
                val_data.at[idx, 'binary_label'] = 1
            elif final_label_formatted == 'NEGATIVE':
                val_data.at[idx, 'binary_label'] = 0
    
    # Save updated train and val files
    train_data.to_csv('test/data/final_dataset_cnn_train_updated.csv', index=False)
    val_data.to_csv('test/data/final_dataset_cnn_val_updated.csv', index=False)
    
    print(f"Updated train dataset saved to: test/data/final_dataset_cnn_train_updated.csv")
    print(f"Updated val dataset saved to: test/data/final_dataset_cnn_val_updated.csv")
    
    # Show updated distributions
    print(f"\nUpdated train distribution:")
    train_counts = train_data['gleamer_finding'].value_counts()
    print(train_counts)
    
    print(f"\nUpdated val distribution:")
    val_counts = val_data['gleamer_finding'].value_counts()
    print(val_counts)
    
else:
    print("No label differences found. Datasets are consistent.")

print(f"\n=== CROSS-CHECK COMPLETE ===")
