#!/usr/bin/env python3
"""
Data Overview and Analysis Script
Analyzes the medical imaging dataset for fracture detection
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from PIL import Image
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, config_path="config_training.yaml"):
        """Initialize data analyzer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.csv_path = self.config['data']['csv_path']
        self.image_root = self.config['data']['image_root']
        self.output_dir = self.config['data']['output_dir']
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🔍 Data Analyzer initialized")
        print(f"📁 CSV Path: {self.csv_path}")
        print(f"🖼️ Image Root: {self.image_root}")
        print(f"📊 Output Dir: {self.output_dir}")
    
    def load_and_validate_csv(self):
        """Load and validate the CSV file"""
        print("\n" + "="*60)
        print("📋 LOADING AND VALIDATING CSV FILE")
        print("="*60)
        
        try:
            # Load CSV
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ CSV loaded successfully!")
            print(f"📊 Dataset shape: {self.df.shape}")
            
            # Basic info
            print(f"\n📋 Dataset Information:")
            print(f"   • Total rows: {len(self.df):,}")
            print(f"   • Total columns: {len(self.df.columns)}")
            print(f"   • Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Column information
            print(f"\n📝 Column Information:")
            for i, col in enumerate(self.df.columns):
                dtype = self.df[col].dtype
                null_count = self.df[col].isnull().sum()
                null_pct = (null_count / len(self.df)) * 100
                print(f"   {i+1:2d}. {col:<30} | {str(dtype):<10} | Nulls: {null_count:>6} ({null_pct:>5.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
    
    def analyze_columns(self):
        """Analyze each column in detail"""
        print("\n" + "="*60)
        print("🔍 DETAILED COLUMN ANALYSIS")
        print("="*60)
        
        for col in self.df.columns:
            print(f"\n📊 Column: {col}")
            print("-" * 40)
            
            # Data type and basic stats
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            unique_count = self.df[col].nunique()
            
            print(f"   Type: {dtype}")
            print(f"   Null values: {null_count} ({(null_count/len(self.df)*100):.1f}%)")
            print(f"   Unique values: {unique_count}")
            
            # Categorical analysis
            if dtype == 'object' or unique_count < 20:
                print(f"   📈 Value distribution:")
                value_counts = self.df[col].value_counts().head(10)
                for value, count in value_counts.items():
                    pct = (count / len(self.df)) * 100
                    print(f"      '{value}': {count:,} ({pct:.1f}%)")
            
            # Numerical analysis
            elif dtype in ['int64', 'float64']:
                print(f"   📊 Statistical summary:")
                stats = self.df[col].describe()
                print(f"      Mean: {stats['mean']:.2f}")
                print(f"      Std:  {stats['std']:.2f}")
                print(f"      Min:  {stats['min']:.2f}")
                print(f"      Max:  {stats['max']:.2f}")
                print(f"      Median: {stats['50%']:.2f}")
    
    def analyze_labels(self):
        """Analyze label distribution"""
        print("\n" + "="*60)
        print("🏷️ LABEL ANALYSIS")
        print("="*60)
        
        # Find label columns
        label_columns = []
        for col in self.df.columns:
            if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower():
                label_columns.append(col)
        
        if not label_columns:
            print("⚠️ No obvious label columns found. Checking for binary columns...")
            for col in self.df.columns:
                if self.df[col].nunique() == 2:
                    label_columns.append(col)
        
        print(f"📋 Found potential label columns: {label_columns}")
        
        for col in label_columns:
            print(f"\n🏷️ Label Distribution - {col}:")
            print("-" * 40)
            
            value_counts = self.df[col].value_counts()
            total = len(self.df)
            
            for label, count in value_counts.items():
                pct = (count / total) * 100
                print(f"   {label}: {count:,} ({pct:.1f}%)")
            
            # Check for class imbalance
            if len(value_counts) == 2:
                ratio = value_counts.iloc[0] / value_counts.iloc[1]
                if ratio > 2 or ratio < 0.5:
                    print(f"   ⚠️ Class imbalance detected! Ratio: {ratio:.2f}")
                else:
                    print(f"   ✅ Classes are relatively balanced. Ratio: {ratio:.2f}")
    
    def analyze_images(self):
        """Analyze image data and paths"""
        print("\n" + "="*60)
        print("🖼️ IMAGE DATA ANALYSIS")
        print("="*60)
        
        # Find image path columns
        image_columns = []
        for col in self.df.columns:
            if 'image' in col.lower() or 'path' in col.lower() or 'url' in col.lower():
                image_columns.append(col)
        
        print(f"📁 Found potential image columns: {image_columns}")
        
        for col in image_columns:
            print(f"\n🖼️ Image Analysis - {col}:")
            print("-" * 40)
            
            # Check for valid paths
            valid_paths = 0
            invalid_paths = 0
            image_sizes = []
            file_extensions = []
            
            sample_size = min(1000, len(self.df))  # Sample for performance
            
            for idx, path in enumerate(self.df[col].dropna().head(sample_size)):
                try:
                    # Check if path exists
                    full_path = os.path.join(self.image_root, str(path))
                    if os.path.exists(full_path):
                        valid_paths += 1
                        
                        # Get image info
                        try:
                            with Image.open(full_path) as img:
                                image_sizes.append(img.size)
                                file_extensions.append(Path(full_path).suffix.lower())
                        except Exception as e:
                            pass
                    else:
                        invalid_paths += 1
                        
                except Exception as e:
                    invalid_paths += 1
            
            print(f"   📊 Path validation (sample of {sample_size}):")
            print(f"      Valid paths: {valid_paths}")
            print(f"      Invalid paths: {invalid_paths}")
            print(f"      Success rate: {(valid_paths/(valid_paths+invalid_paths)*100):.1f}%")
            
            if image_sizes:
                print(f"   📐 Image dimensions:")
                sizes_array = np.array(image_sizes)
                print(f"      Width:  {sizes_array[:, 0].min()} - {sizes_array[:, 0].max()} (avg: {sizes_array[:, 0].mean():.0f})")
                print(f"      Height: {sizes_array[:, 1].min()} - {sizes_array[:, 1].max()} (avg: {sizes_array[:, 1].mean():.0f})")
                
                # Most common sizes
                size_counts = Counter([(w, h) for w, h in image_sizes])
                print(f"   📏 Most common sizes:")
                for (w, h), count in size_counts.most_common(5):
                    print(f"      {w}x{h}: {count} images")
            
            if file_extensions:
                ext_counts = Counter(file_extensions)
                print(f"   📄 File extensions:")
                for ext, count in ext_counts.most_common():
                    print(f"      {ext}: {count} images")
    
    def generate_visualizations(self):
        """Generate data visualization plots"""
        print("\n" + "="*60)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(fig_dir, exist_ok=True)
        
        # 1. Dataset overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # Missing values heatmap
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            axes[0, 0].bar(range(len(missing_data)), missing_data.values)
            axes[0, 0].set_title('Missing Values by Column')
            axes[0, 0].set_ylabel('Number of Missing Values')
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            axes[0, 0].set_title('Missing Values by Column')
        
        # Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Data Types Distribution')
        
        # Dataset size over time (if there's a date column)
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            try:
                date_col = date_cols[0]
                self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
                daily_counts = self.df[date_col].dt.date.value_counts().sort_index()
                axes[1, 0].plot(daily_counts.index, daily_counts.values)
                axes[1, 0].set_title(f'Data Collection Over Time ({date_col})')
                axes[1, 0].tick_params(axis='x', rotation=45)
            except:
                axes[1, 0].text(0.5, 0.5, 'No valid date column', ha='center', va='center')
                axes[1, 0].set_title('Data Collection Over Time')
        else:
            axes[1, 0].text(0.5, 0.5, 'No date column found', ha='center', va='center')
            axes[1, 0].set_title('Data Collection Over Time')
        
        # Memory usage by column
        memory_usage = self.df.memory_usage(deep=True) / 1024**2  # Convert to MB
        axes[1, 1].bar(range(len(memory_usage)), memory_usage.values)
        axes[1, 1].set_title('Memory Usage by Column (MB)')
        axes[1, 1].set_ylabel('Memory Usage (MB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Label distribution plots
        label_columns = []
        for col in self.df.columns:
            if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower():
                label_columns.append(col)
        
        if label_columns:
            n_labels = len(label_columns)
            fig, axes = plt.subplots(1, min(n_labels, 3), figsize=(5*min(n_labels, 3), 5))
            if n_labels == 1:
                axes = [axes]
            
            for i, col in enumerate(label_columns[:3]):
                value_counts = self.df[col].value_counts()
                axes[i].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                axes[i].set_title(f'Label Distribution - {col}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to: {fig_dir}")
    
    def generate_report(self):
        """Generate comprehensive data report"""
        print("\n" + "="*60)
        print("📝 GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        report_path = os.path.join(self.output_dir, 'data_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("MEDICAL FRACTURE DETECTION - DATA ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"CSV Path: {self.csv_path}\n")
            f.write(f"Image Root: {self.image_root}\n")
            f.write(f"Dataset Shape: {self.df.shape}\n")
            f.write(f"Total Rows: {len(self.df):,}\n")
            f.write(f"Total Columns: {len(self.df.columns)}\n")
            f.write(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
            
            # Column analysis
            f.write("COLUMN ANALYSIS\n")
            f.write("-" * 15 + "\n")
            for col in self.df.columns:
                dtype = self.df[col].dtype
                null_count = self.df[col].isnull().sum()
                null_pct = (null_count / len(self.df)) * 100
                unique_count = self.df[col].nunique()
                
                f.write(f"{col}:\n")
                f.write(f"  Type: {dtype}\n")
                f.write(f"  Null values: {null_count} ({null_pct:.1f}%)\n")
                f.write(f"  Unique values: {unique_count}\n")
                
                if dtype == 'object' or unique_count < 20:
                    f.write(f"  Top values:\n")
                    value_counts = self.df[col].value_counts().head(5)
                    for value, count in value_counts.items():
                        pct = (count / len(self.df)) * 100
                        f.write(f"    '{value}': {count:,} ({pct:.1f}%)\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            # Check for missing values
            missing_cols = self.df.columns[self.df.isnull().any()].tolist()
            if missing_cols:
                f.write(f"• Handle missing values in: {missing_cols}\n")
            
            # Check for class imbalance
            label_columns = [col for col in self.df.columns if 'label' in col.lower()]
            for col in label_columns:
                if self.df[col].nunique() == 2:
                    value_counts = self.df[col].value_counts()
                    ratio = value_counts.iloc[0] / value_counts.iloc[1]
                    if ratio > 2 or ratio < 0.5:
                        f.write(f"• Consider class balancing for {col} (ratio: {ratio:.2f})\n")
            
            # Image path validation
            image_cols = [col for col in self.df.columns if 'image' in col.lower()]
            if image_cols:
                f.write(f"• Validate image paths in: {image_cols}\n")
                f.write(f"• Ensure images are accessible from: {self.image_root}\n")
            
            f.write("\n")
            f.write("TRAINING RECOMMENDATIONS\n")
            f.write("-" * 22 + "\n")
            f.write("• Use data augmentation to increase dataset diversity\n")
            f.write("• Implement stratified sampling for train/val/test splits\n")
            f.write("• Consider mixed precision training for H100 GPU\n")
            f.write("• Use early stopping to prevent overfitting\n")
            f.write("• Monitor class distribution during training\n")
        
        print(f"✅ Report saved to: {report_path}")
    
    def run_full_analysis(self):
        """Run complete data analysis"""
        print("🚀 Starting comprehensive data analysis...")
        
        # Load and validate data
        if not self.load_and_validate_csv():
            return False
        
        # Run analysis steps
        self.analyze_columns()
        self.analyze_labels()
        self.analyze_images()
        self.generate_visualizations()
        self.generate_report()
        
        print("\n" + "="*60)
        print("✅ DATA ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 Results saved to: {self.output_dir}")
        print(f"📈 Visualizations: {os.path.join(self.output_dir, 'visualizations')}")
        print(f"📝 Report: {os.path.join(self.output_dir, 'data_analysis_report.txt')}")
        
        return True

def main():
    """Main function to run data analysis"""
    print("🏥 Medical Fracture Detection - Data Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = DataAnalyzer()
    
    # Run full analysis
    success = analyzer.run_full_analysis()
    
    if success:
        print("\n🎉 Data analysis completed successfully!")
        print("📋 Next steps:")
        print("   1. Review the generated report")
        print("   2. Check visualizations")
        print("   3. Validate image paths")
        print("   4. Proceed with training setup")
    else:
        print("\n❌ Data analysis failed!")
        print("Please check the CSV path and file format.")

if __name__ == "__main__":
    main()
