#!/usr/bin/env python3
import os
import csv
import pandas as pd
from pathlib import Path
import pydicom

def extract_dicom_to_csv():
    """Extract DICOM metadata from .dcm files and save to CSV"""
    
    # Paths
    dicom_path = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-dicom/"
    output_csv = "dicom_metadata.csv"
    
    print(f"Scanning DICOM files in: {dicom_path}")
    
    # Find all .dcm files
    dicom_files = []
    for root, dirs, files in os.walk(dicom_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    print(f"Found {len(dicom_files)} .dcm files")
    
    # Extract metadata from each file
    all_data = []
    
    for i, file_path in enumerate(dicom_files):
        print(f"Processing {i+1}/{len(dicom_files)}: {os.path.basename(file_path)}")
        
        try:
            # Read DICOM file without pixel data for speed
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            # Extract common metadata
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'PatientID': str(ds.get('PatientID', '')),
                'PatientName': str(ds.get('PatientName', '')),
                'StudyInstanceUID': str(ds.get('StudyInstanceUID', '')),
                'SeriesInstanceUID': str(ds.get('SeriesInstanceUID', '')),
                'SOPInstanceUID': str(ds.get('SOPInstanceUID', '')),
                'StudyDate': str(ds.get('StudyDate', '')),
                'StudyTime': str(ds.get('StudyTime', '')),
                'Modality': str(ds.get('Modality', '')),
                'BodyPartExamined': str(ds.get('BodyPartExamined', '')),
                'Manufacturer': str(ds.get('Manufacturer', '')),
                'Rows': str(ds.get('Rows', '')),
                'Columns': str(ds.get('Columns', '')),
                'PixelSpacing': str(ds.get('PixelSpacing', '')),
                'SliceThickness': str(ds.get('SliceThickness', '')),
                'WindowCenter': str(ds.get('WindowCenter', '')),
                'WindowWidth': str(ds.get('WindowWidth', '')),
                'InstitutionName': str(ds.get('InstitutionName', '')),
                'SeriesDescription': str(ds.get('SeriesDescription', '')),
                'StudyDescription': str(ds.get('StudyDescription', ''))
            }
            
            all_data.append(metadata)
            
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            # Add error record
            all_data.append({
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'error': str(e)
            })
    
    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"✅ Data saved to {output_csv}")
        print(f"Total records: {len(df)}")
    else:
        print("❌ No data to save")

if __name__ == "__main__":
    extract_dicom_to_csv()
