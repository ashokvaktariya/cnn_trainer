#!/usr/bin/env python3
"""
Run All Training Steps
Executes the complete medical image classification pipeline
H200 GPU Server Optimized
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

def run_step(step_name, script_name, args=None):
    """Run a single step with error handling"""
    print(f"\n{'='*60}")
    print(f"🚀 Running {step_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Prepare command
        cmd = [sys.executable, script_name]
        if args:
            cmd.extend(args)
        
        # Run the script
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ {step_name} completed successfully!")
        print(f"⏱️  Time taken: {elapsed_time/60:.1f} minutes")
        
        # Print any output
        if result.stdout:
            print("📝 Output:")
            print(result.stdout[-1000:])  # Last 1000 characters
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        print(f"❌ {step_name} failed!")
        print(f"⏱️  Time taken: {elapsed_time/60:.1f} minutes")
        print(f"🔍 Error code: {e.returncode}")
        
        if e.stdout:
            print("📝 Output:")
            print(e.stdout[-1000:])
        
        if e.stderr:
            print("❌ Error:")
            print(e.stderr[-1000:])
        
        return False

def check_prerequisites():
    """Check if prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check if CSV file exists
    from config import CSV_FILE
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file not found: {CSV_FILE}")
        print("Please ensure the dataset is available at the specified path.")
        return False
    
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available. Training will be slower on CPU.")
        else:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed. Please install requirements first.")
        return False
    
    print("✅ Prerequisites check passed!")
    return True

def main():
    """Main function to run all steps"""
    parser = argparse.ArgumentParser(description='Run complete medical image classification pipeline')
    parser.add_argument('--step', type=str, choices=[
        'preprocessing', 'image_models', 'multimodal_models', 
        'ensemble', 'evaluation', 'all'
    ], default='all', help='Which step(s) to run')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip preprocessing if already done')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Use sample of data for testing (default: full dataset)')
    
    args = parser.parse_args()
    
    print("🏥 Medical Image Classification - Complete Pipeline")
    print("=" * 70)
    print(f"🎯 Target: H200 GPU Server")
    print(f"📁 Data Path: /sharedata01/CNN_data")
    print(f"🔧 Step: {args.step}")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    total_start_time = time.time()
    success_count = 0
    total_steps = 0
    
    # Define steps
    steps = [
        {
            'name': 'Step 1: Dataset Preprocessing',
            'script': 'step1_preprocessing.py',
            'args': ['--sample-size', str(args.sample_size)] if args.sample_size else None
        },
        {
            'name': 'Step 2: Train Image-Only Models',
            'script': 'step2_train_image_models.py',
            'args': None
        },
        {
            'name': 'Step 3: Train Multimodal Models',
            'script': 'step3_train_multimodal_models.py',
            'args': None
        },
        {
            'name': 'Step 4: Create Ensemble Models',
            'script': 'step4_create_ensemble.py',
            'args': None
        },
        {
            'name': 'Step 5: Final Evaluation',
            'script': 'step5_final_evaluation.py',
            'args': None
        }
    ]
    
    # Determine which steps to run
    if args.step == 'all':
        steps_to_run = steps
    elif args.step == 'preprocessing':
        steps_to_run = [steps[0]]
    elif args.step == 'image_models':
        steps_to_run = [steps[1]]
    elif args.step == 'multimodal_models':
        steps_to_run = [steps[2]]
    elif args.step == 'ensemble':
        steps_to_run = [steps[3]]
    elif args.step == 'evaluation':
        steps_to_run = [steps[4]]
    
    # Skip preprocessing if requested
    if args.skip_preprocessing and steps[0] in steps_to_run:
        print(f"\n⏭️  Skipping preprocessing as requested...")
        steps_to_run.remove(steps[0])
    
    # Run steps
    for i, step in enumerate(steps_to_run):
        total_steps += 1
        
        print(f"\n📋 Progress: Step {i+1}/{len(steps_to_run)}")
        
        if run_step(step['name'], step['script'], step['args']):
            success_count += 1
        else:
            print(f"\n❌ Pipeline stopped due to failure in {step['name']}")
            break
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*70}")
    print(f"🏁 PIPELINE COMPLETED")
    print(f"{'='*70}")
    print(f"✅ Successful steps: {success_count}/{total_steps}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    if success_count == total_steps:
        print(f"🎉 All steps completed successfully!")
        print(f"📁 Results saved in: /sharedata01/CNN_data/medical_classification/")
        print(f"📊 Check final_evaluation/ directory for comprehensive results")
    else:
        print(f"⚠️  Pipeline completed with some failures.")
        print(f"🔧 Please check the logs and fix issues before continuing.")
    
    return success_count == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
