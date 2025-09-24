#!/usr/bin/env python3
"""
DICOM CSV Analyzer

Analyzes the converted DICOM CSV data to understand the dataset structure
and identify key fields for labeling and mapping.
"""

import pandas as pd
import json
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DicomCsvAnalyzer:
    """Analyze DICOM CSV data"""
    
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = None
        
        logger.info("🔍 DICOM CSV Analyzer initialized")
    
    def load_csv(self):
        """Load CSV data"""
        logger.info(f"📋 Loading CSV data from: {self.csv_path}")
        
        if not self.csv_path.exists():
            logger.error(f"CSV file not found: {self.csv_path}")
            return False
        
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"📊 Loaded {len(self.df)} records")
            return True
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return False
    
    def analyze_basic_info(self):
        """Analyze basic information"""
        logger.info("📊 BASIC INFORMATION")
        logger.info("=" * 50)
        
        logger.info(f"Total records: {len(self.df)}")
        logger.info(f"Total columns: {len(self.df.columns)}")
        
        # Check read success
        if 'read_success' in self.df.columns:
            success_count = self.df['read_success'].sum()
            logger.info(f"Successfully read: {success_count} ({success_count/len(self.df)*100:.1f}%)")
            failed_count = len(self.df) - success_count
            logger.info(f"Failed to read: {failed_count} ({failed_count/len(self.df)*100:.1f}%)")
        
        # File size analysis
        if 'file_size' in self.df.columns:
            total_size = self.df['file_size'].sum()
            avg_size = self.df['file_size'].mean()
            logger.info(f"Total file size: {total_size / (1024*1024*1024):.2f} GB")
            logger.info(f"Average file size: {avg_size / (1024*1024):.2f} MB")
    
    def analyze_uid_patterns(self):
        """Analyze UID patterns"""
        logger.info("\n🔍 UID PATTERN ANALYSIS")
        logger.info("=" * 50)
        
        # SOPInstanceUID patterns
        if 'SOPInstanceUID' in self.df.columns:
            uid_patterns = {}
            for uid in self.df['SOPInstanceUID'].dropna():
                if pd.notna(uid) and len(str(uid)) > 10:
                    pattern = '.'.join(str(uid).split('.')[:3])
                    uid_patterns[pattern] = uid_patterns.get(pattern, 0) + 1
            
            logger.info("Top SOPInstanceUID patterns:")
            sorted_patterns = sorted(uid_patterns.items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_patterns[:10]:
                logger.info(f"  {pattern}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # StudyInstanceUID patterns
        if 'StudyInstanceUID' in self.df.columns:
            study_patterns = {}
            for uid in self.df['StudyInstanceUID'].dropna():
                if pd.notna(uid) and len(str(uid)) > 10:
                    pattern = '.'.join(str(uid).split('.')[:3])
                    study_patterns[pattern] = study_patterns.get(pattern, 0) + 1
            
            logger.info("\nTop StudyInstanceUID patterns:")
            sorted_patterns = sorted(study_patterns.items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_patterns[:5]:
                logger.info(f"  {pattern}: {count} ({count/len(self.df)*100:.1f}%)")
    
    def analyze_modality(self):
        """Analyze modality distribution"""
        logger.info("\n🔬 MODALITY ANALYSIS")
        logger.info("=" * 50)
        
        if 'Modality' in self.df.columns:
            modality_counts = self.df['Modality'].value_counts()
            logger.info("Modality distribution:")
            for modality, count in modality_counts.items():
                logger.info(f"  {modality}: {count} ({count/len(self.df)*100:.1f}%)")
    
    def analyze_body_parts(self):
        """Analyze body parts"""
        logger.info("\n🦴 BODY PART ANALYSIS")
        logger.info("=" * 50)
        
        if 'BodyPartExamined' in self.df.columns:
            body_part_counts = self.df['BodyPartExamined'].value_counts()
            logger.info("Body part distribution:")
            for body_part, count in body_part_counts.head(15).items():
                logger.info(f"  {body_part}: {count} ({count/len(self.df)*100:.1f}%)")
    
    def analyze_clinical_fields(self):
        """Analyze clinical fields"""
        logger.info("\n🏥 CLINICAL FIELDS ANALYSIS")
        logger.info("=" * 50)
        
        clinical_fields = ['ClinicalIndication', 'Findings', 'Impression', 'Diagnosis']
        
        for field in clinical_fields:
            if field in self.df.columns:
                non_empty = self.df[field].notna().sum()
                logger.info(f"{field}: {non_empty} non-empty ({non_empty/len(self.df)*100:.1f}%)")
                
                # Show sample values
                sample_values = self.df[field].dropna().head(3)
                for i, value in enumerate(sample_values, 1):
                    logger.info(f"  Sample {i}: {str(value)[:100]}...")
    
    def analyze_study_descriptions(self):
        """Analyze study descriptions"""
        logger.info("\n📋 STUDY DESCRIPTION ANALYSIS")
        logger.info("=" * 50)
        
        if 'StudyDescription' in self.df.columns:
            study_desc_counts = self.df['StudyDescription'].value_counts()
            logger.info("Top study descriptions:")
            for desc, count in study_desc_counts.head(10).items():
                logger.info(f"  {desc}: {count} ({count/len(self.df)*100:.1f}%)")
    
    def analyze_manufacturer(self):
        """Analyze manufacturer distribution"""
        logger.info("\n🏭 MANUFACTURER ANALYSIS")
        logger.info("=" * 50)
        
        if 'Manufacturer' in self.df.columns:
            manufacturer_counts = self.df['Manufacturer'].value_counts()
            logger.info("Manufacturer distribution:")
            for manufacturer, count in manufacturer_counts.head(10).items():
                logger.info(f"  {manufacturer}: {count} ({count/len(self.df)*100:.1f}%)")
    
    def find_fracture_keywords(self):
        """Find fracture-related keywords"""
        logger.info("\n🔍 FRACTURE KEYWORD ANALYSIS")
        logger.info("=" * 50)
        
        fracture_keywords = ['fracture', 'broken', 'crack', 'break', 'fissure', 'dislocation', 'avulsion']
        
        # Check StudyDescription
        if 'StudyDescription' in self.df.columns:
            logger.info("Fracture keywords in StudyDescription:")
            for keyword in fracture_keywords:
                count = self.df['StudyDescription'].str.contains(keyword, case=False, na=False).sum()
                if count > 0:
                    logger.info(f"  '{keyword}': {count} cases")
        
        # Check Findings
        if 'Findings' in self.df.columns:
            logger.info("\nFracture keywords in Findings:")
            for keyword in fracture_keywords:
                count = self.df['Findings'].str.contains(keyword, case=False, na=False).sum()
                if count > 0:
                    logger.info(f"  '{keyword}': {count} cases")
        
        # Check ClinicalIndication
        if 'ClinicalIndication' in self.df.columns:
            logger.info("\nFracture keywords in ClinicalIndication:")
            for keyword in fracture_keywords:
                count = self.df['ClinicalIndication'].str.contains(keyword, case=False, na=False).sum()
                if count > 0:
                    logger.info(f"  '{keyword}': {count} cases")
    
    def find_negative_keywords(self):
        """Find negative keywords"""
        logger.info("\n❌ NEGATIVE KEYWORD ANALYSIS")
        logger.info("=" * 50)
        
        negative_keywords = ['no acute', 'no fracture', 'normal', 'unremarkable', 'negative for fracture', 'no evidence of fracture']
        
        # Check Findings
        if 'Findings' in self.df.columns:
            logger.info("Negative keywords in Findings:")
            for keyword in negative_keywords:
                count = self.df['Findings'].str.contains(keyword, case=False, na=False).sum()
                if count > 0:
                    logger.info(f"  '{keyword}': {count} cases")
    
    def analyze_missing_data(self):
        """Analyze missing data patterns"""
        logger.info("\n❓ MISSING DATA ANALYSIS")
        logger.info("=" * 50)
        
        # Key fields for analysis
        key_fields = ['SOPInstanceUID', 'StudyDescription', 'Findings', 'ClinicalIndication', 'Modality', 'BodyPartExamined']
        
        logger.info("Missing data in key fields:")
        for field in key_fields:
            if field in self.df.columns:
                missing = self.df[field].isna().sum()
                logger.info(f"  {field}: {missing} missing ({missing/len(self.df)*100:.1f}%)")
    
    def generate_summary_report(self):
        """Generate summary report"""
        logger.info("\n📊 SUMMARY REPORT")
        logger.info("=" * 50)
        
        # Key insights
        insights = []
        
        # Check if we have UIDs
        if 'SOPInstanceUID' in self.df.columns:
            uid_count = self.df['SOPInstanceUID'].notna().sum()
            insights.append(f"✅ {uid_count} records have SOPInstanceUID")
        
        # Check clinical fields
        clinical_fields = ['Findings', 'ClinicalIndication']
        for field in clinical_fields:
            if field in self.df.columns:
                non_empty = self.df[field].notna().sum()
                if non_empty > 0:
                    insights.append(f"✅ {non_empty} records have {field} data")
                else:
                    insights.append(f"❌ No {field} data found")
        
        # Check for fracture keywords
        if 'StudyDescription' in self.df.columns:
            fracture_count = self.df['StudyDescription'].str.contains('fracture', case=False, na=False).sum()
            if fracture_count > 0:
                insights.append(f"✅ {fracture_count} records mention 'fracture' in StudyDescription")
        
        logger.info("Key insights:")
        for insight in insights:
            logger.info(f"  {insight}")
    
    def run_full_analysis(self):
        """Run complete analysis"""
        if not self.load_csv():
            return False
        
        self.analyze_basic_info()
        self.analyze_uid_patterns()
        self.analyze_modality()
        self.analyze_body_parts()
        self.analyze_clinical_fields()
        self.analyze_study_descriptions()
        self.analyze_manufacturer()
        self.find_fracture_keywords()
        self.find_negative_keywords()
        self.analyze_missing_data()
        self.generate_summary_report()
        
        logger.info("\n🎉 Analysis completed!")
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze DICOM CSV data')
    parser.add_argument('--csv-path', default='dicom_metadata.csv',
                       help='Path to DICOM CSV file')
    
    args = parser.parse_args()
    
    # Create analyzer instance
    analyzer = DicomCsvAnalyzer(csv_path=args.csv_path)
    
    # Run analysis
    success = analyzer.run_full_analysis()
    
    if success:
        logger.info("✅ Analysis completed successfully!")
    else:
        logger.error("❌ Analysis failed!")

if __name__ == "__main__":
    main()
