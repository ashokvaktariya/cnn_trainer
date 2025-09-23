#!/usr/bin/env python3
import os
from PIL import Image
import cv2
import numpy as np

def read_images_from_directory(directory):
    """Read all images from a directory"""
    print(f"Reading images from: {directory}")
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    image_files = []
    image_data = []
    
    # Get all image files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm', '.dicom')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files")
    
    # Read each image
    for i, image_path in enumerate(image_files):
        try:
            if image_path.lower().endswith(('.dcm', '.dicom')):
                # Read DICOM file
                import pydicom
                ds = pydicom.dcmread(image_path)
                pixel_array = ds.pixel_array
                image_data.append({
                    'path': image_path,
                    'data': pixel_array,
                    'shape': pixel_array.shape,
                    'type': 'DICOM',
                    'modality': getattr(ds, 'Modality', 'Unknown'),
                    'study_description': getattr(ds, 'StudyDescription', 'Unknown')
                })
                print(f"Image {i+1}: {os.path.basename(image_path)} - DICOM {pixel_array.shape}")
            else:
                # Read regular image
                image = Image.open(image_path)
                image_array = np.array(image)
                image_data.append({
                    'path': image_path,
                    'data': image_array,
                    'shape': image_array.shape,
                    'type': 'Regular',
                    'mode': image.mode,
                    'format': image.format
                })
                print(f"Image {i+1}: {os.path.basename(image_path)} - {image_array.shape} {image.mode}")
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
    
    return image_data

def main():
    print("Reading all images from positive_images and random_dicom_samples...")
    print("=" * 60)
    
    # Read from positive_images directory
    positive_images = read_images_from_directory("positive_images")
    print(f"\nPositive images: {len(positive_images)} files")
    
    # Read from random_dicom_samples directory
    random_images = read_images_from_directory("random_dicom_samples")
    print(f"\nRandom DICOM samples: {len(random_images)} files")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Total images read: {len(positive_images) + len(random_images)}")
    print(f"Positive images: {len(positive_images)}")
    print(f"Random DICOM samples: {len(random_images)}")
    
    # Show some statistics
    if positive_images:
        print(f"\nPositive images shapes:")
        for img in positive_images[:5]:  # Show first 5
            print(f"  {os.path.basename(img['path'])}: {img['shape']}")
    
    if random_images:
        print(f"\nRandom DICOM shapes:")
        for img in random_images[:5]:  # Show first 5
            print(f"  {os.path.basename(img['path'])}: {img['shape']} ({img.get('modality', 'Unknown')})")

if __name__ == "__main__":
    main()
