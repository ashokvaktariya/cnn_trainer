#!/usr/bin/env python3
"""
Image Alignment System
Maintains proper alignment between downloaded images and CSV columns
"""

import os
import json
import hashlib
from pathlib import Path
import pandas as pd

class ImageAlignmentSystem:
    """System to maintain alignment between images and CSV data"""
    
    def __init__(self, data_root, images_dir):
        self.data_root = data_root
        self.images_dir = images_dir
        self.alignment_file = os.path.join(data_root, "image_alignment.json")
        self.alignment_data = self.load_alignment_data()
    
    def load_alignment_data(self):
        """Load existing alignment data"""
        if os.path.exists(self.alignment_file):
            with open(self.alignment_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_alignment_data(self):
        """Save alignment data to file"""
        os.makedirs(os.path.dirname(self.alignment_file), exist_ok=True)
        with open(self.alignment_file, 'w') as f:
            json.dump(self.alignment_data, f, indent=2)
    
    def create_image_id(self, row_data, image_index):
        """Create unique image ID based on CSV row data"""
        # Use multiple identifiers for better uniqueness
        accession = row_data.get('ACCESSION_NUMBER', 'unknown')
        study_uid = row_data.get('STUDY_INSTANCE_UID', 'unknown')
        sop_uids = row_data.get('SOP_INSTANCE_UID_ARRAY', [])
        
        # Create unique ID combining multiple fields
        if isinstance(sop_uids, list) and len(sop_uids) > image_index:
            sop_uid = sop_uids[image_index]
            # Clean SOP UID for filename
            sop_uid_clean = "".join(c for c in sop_uid if c.isalnum() or c in "._-")
            image_id = f"{accession}_{sop_uid_clean}"
        else:
            image_id = f"{accession}_{image_index}"
        
        return image_id
    
    def create_alignment_entry(self, csv_row_index, row_data, image_index, image_url, local_path):
        """Create alignment entry for an image"""
        image_id = self.create_image_id(row_data, image_index)
        
        alignment_entry = {
            'csv_row_index': csv_row_index,
            'image_index': image_index,
            'image_id': image_id,
            'local_path': local_path,
            'original_url': image_url,
            'accession_number': row_data.get('ACCESSION_NUMBER'),
            'study_instance_uid': row_data.get('STUDY_INSTANCE_UID'),
            'sop_instance_uid': row_data.get('SOP_INSTANCE_UID_ARRAY', [])[image_index] if isinstance(row_data.get('SOP_INSTANCE_UID_ARRAY', []), list) and len(row_data.get('SOP_INSTANCE_UID_ARRAY', [])) > image_index else None,
            'body_part': row_data.get('BODY_PART_ARRAY'),
            'label': 1 if row_data.get('GLEAMER_FINDING') == 'POSITIVE' else 0,
            'downloaded_at': None,  # Will be set when downloaded
            'file_size': None,      # Will be set when downloaded
            'image_dimensions': None  # Will be set when downloaded
        }
        
        return image_id, alignment_entry
    
    def register_image(self, csv_row_index, row_data, image_index, image_url, local_path):
        """Register an image in the alignment system"""
        image_id, alignment_entry = self.create_alignment_entry(
            csv_row_index, row_data, image_index, image_url, local_path
        )
        
        self.alignment_data[image_id] = alignment_entry
        return image_id
    
    def update_image_info(self, image_id, file_size=None, dimensions=None):
        """Update image information after download"""
        if image_id in self.alignment_data:
            if file_size is not None:
                self.alignment_data[image_id]['file_size'] = file_size
            if dimensions is not None:
                self.alignment_data[image_id]['image_dimensions'] = dimensions
            self.alignment_data[image_id]['downloaded_at'] = str(pd.Timestamp.now())
    
    def get_image_info(self, image_id):
        """Get information for a specific image"""
        return self.alignment_data.get(image_id)
    
    def get_images_for_csv_row(self, csv_row_index):
        """Get all images associated with a CSV row"""
        images = []
        for image_id, info in self.alignment_data.items():
            if info['csv_row_index'] == csv_row_index:
                images.append((image_id, info))
        return sorted(images, key=lambda x: x[1]['image_index'])
    
    def get_alignment_statistics(self):
        """Get statistics about image alignment"""
        total_images = len(self.alignment_data)
        unique_csv_rows = len(set(info['csv_row_index'] for info in self.alignment_data.values()))
        
        # Count images per row
        images_per_row = {}
        for info in self.alignment_data.values():
            row_idx = info['csv_row_index']
            images_per_row[row_idx] = images_per_row.get(row_idx, 0) + 1
        
        stats = {
            'total_images': total_images,
            'unique_csv_rows': unique_csv_rows,
            'avg_images_per_row': total_images / unique_csv_rows if unique_csv_rows > 0 else 0,
            'images_per_row_distribution': {
                str(k): v for k, v in sorted(images_per_row.items())
            }
        }
        
        return stats
    
    def validate_alignment(self, csv_file):
        """Validate that alignment is correct"""
        df = pd.read_csv(csv_file)
        validation_results = {
            'total_csv_rows': len(df),
            'total_aligned_images': len(self.alignment_data),
            'missing_images': [],
            'orphaned_images': [],
            'alignment_issues': []
        }
        
        # Check for missing images
        for csv_idx in range(len(df)):
            images = self.get_images_for_csv_row(csv_idx)
            if len(images) == 0:
                validation_results['missing_images'].append(csv_idx)
        
        # Check for orphaned images (images without corresponding CSV row)
        valid_csv_indices = set(range(len(df)))
        for image_id, info in self.alignment_data.items():
            if info['csv_row_index'] not in valid_csv_indices:
                validation_results['orphaned_images'].append(image_id)
        
        return validation_results
    
    def create_image_filename(self, image_id, original_url=None):
        """Create consistent filename for image"""
        # Clean image_id for filename
        clean_id = "".join(c for c in image_id if c.isalnum() or c in "._-")
        
        # Try to get extension from URL
        extension = ".jpg"  # Default
        if original_url and '.' in original_url.split('/')[-1]:
            url_extension = original_url.split('/')[-1].split('.')[-1]
            if url_extension.lower() in ['jpg', 'jpeg', 'png', 'bmp']:
                extension = f".{url_extension.lower()}"
        
        filename = f"{clean_id}{extension}"
        return filename
    
    def get_image_path(self, image_id, original_url=None):
        """Get full path for an image"""
        filename = self.create_image_filename(image_id, original_url)
        return os.path.join(self.images_dir, filename)

def demonstrate_alignment_system():
    """Demonstrate how the alignment system works"""
    print("🔍 Image Alignment System Demonstration")
    print("=" * 50)
    
    # Example CSV row data
    example_row = {
        'ACCESSION_NUMBER': '1883942.001SRMC',
        'STUDY_INSTANCE_UID': '1.2.410.200010.1005146.4746.418.1954055',
        'SOP_INSTANCE_UID_ARRAY': [
            '1.2.840.113619.2.203.4.2147483647.1754035138.302895.1',
            '1.2.840.113619.2.203.4.2147483647.1754035175.251619.1',
            '1.2.250.1.439.5.78.20250801080320113.62779384'
        ],
        'BODY_PART_ARRAY': ['FOREARM'],
        'GLEAMER_FINDING': 'NEGATIVE'
    }
    
    example_urls = [
        'https://ciprodciviedicom01.blob.core.windows.net/gleamer/1.2.840.113619.2.203.4.2147483647.1754035138.302895.1.jpg',
        'https://ciprodciviedicom01.blob.core.windows.net/gleamer/1.2.840.113619.2.203.4.2147483647.1754035175.251619.1.jpg',
        'https://ciprodciviedicom01.blob.core.windows.net/gleamer/1.2.250.1.439.5.78.20250801080320113.62779384.jpg'
    ]
    
    # Initialize alignment system
    alignment_system = ImageAlignmentSystem(
        data_root="/sharedata01/CNN_data",
        images_dir="/sharedata01/CNN_data/images"
    )
    
    print("📊 Example CSV Row:")
    print(f"   Accession: {example_row['ACCESSION_NUMBER']}")
    print(f"   Study UID: {example_row['STUDY_INSTANCE_UID']}")
    print(f"   SOP UIDs: {example_row['SOP_INSTANCE_UID_ARRAY']}")
    print(f"   Label: {example_row['GLEAMER_FINDING']}")
    
    print("\n🖼️ Image Alignment Process:")
    for i, url in enumerate(example_urls):
        # Create image ID
        image_id = alignment_system.create_image_id(example_row, i)
        image_path = alignment_system.get_image_path(image_id, url)
        
        print(f"\n   Image {i+1}:")
        print(f"   Image ID: {image_id}")
        print(f"   Local Path: {image_path}")
        print(f"   Original URL: {url}")
        
        # Register image (simulate)
        alignment_system.register_image(
            csv_row_index=0,  # First row
            row_data=example_row,
            image_index=i,
            image_url=url,
            local_path=image_path
        )
    
    print("\n📈 Alignment Statistics:")
    stats = alignment_system.get_alignment_statistics()
    print(f"   Total Images: {stats['total_images']}")
    print(f"   Unique CSV Rows: {stats['unique_csv_rows']}")
    print(f"   Avg Images per Row: {stats['avg_images_per_row']:.2f}")
    
    print("\n✅ Alignment system demonstration completed!")

if __name__ == "__main__":
    demonstrate_alignment_system()
