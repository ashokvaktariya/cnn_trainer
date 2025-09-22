#!/usr/bin/env python3
"""
Deployment script for Medical Fracture Detection API
Supports multiple deployment platforms for public URL access
"""

import os
import subprocess
import sys
import json
from pathlib import Path

class APIDeployer:
    def __init__(self):
        self.platforms = {
            'railway': self.deploy_railway,
            'heroku': self.deploy_heroku,
            'render': self.deploy_render,
            'local': self.run_local
        }
    
    def create_requirements(self):
        """Create requirements.txt for deployment"""
        requirements = [
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0",
            "python-multipart==0.0.6",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "Pillow>=10.0.0",
            "numpy>=1.24.0",
            "pyyaml>=6.0",
            "python-dotenv>=1.0.0"
        ]
        
        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(requirements))
        
        print("✅ requirements.txt created")
    
    def create_procfile(self):
        """Create Procfile for Heroku/Railway"""
        procfile_content = "web: uvicorn inference_api:app --host 0.0.0.0 --port $PORT"
        
        with open('Procfile', 'w') as f:
            f.write(procfile_content)
        
        print("✅ Procfile created")
    
    def create_runtime_txt(self):
        """Create runtime.txt for Python version"""
        runtime_content = "python-3.11.0"
        
        with open('runtime.txt', 'w') as f:
            f.write(runtime_content)
        
        print("✅ runtime.txt created")
    
    def create_dockerfile(self):
        """Create Dockerfile for containerized deployment"""
        dockerfile_content = """FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY inference_api.py .
COPY config_training.yaml .

# Create directory for models
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        print("✅ Dockerfile created")
    
    def create_env_file(self):
        """Create .env file template"""
        env_content = """# Environment variables for Medical Fracture Detection API
PORT=8000
PYTHONPATH=/app
CUDA_VISIBLE_DEVICES=0

# Model configuration
MODEL_PATH=/app/models/best_model.pth
CONFIG_PATH=/app/config_training.yaml

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created")
    
    def deploy_railway(self):
        """Deploy to Railway"""
        print("🚂 Deploying to Railway...")
        
        # Check if Railway CLI is installed
        try:
            subprocess.run(['railway', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Railway CLI not found. Please install it first:")
            print("   npm install -g @railway/cli")
            return False
        
        # Create Railway project
        try:
            subprocess.run(['railway', 'login'], check=True)
            subprocess.run(['railway', 'init'], check=True)
            subprocess.run(['railway', 'up'], check=True)
            
            # Get the URL
            result = subprocess.run(['railway', 'domain'], capture_output=True, text=True)
            if result.returncode == 0:
                domain = result.stdout.strip()
                print(f"✅ Deployed to Railway: https://{domain}")
                return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Railway deployment failed: {e}")
            return False
    
    def deploy_heroku(self):
        """Deploy to Heroku"""
        print("🟣 Deploying to Heroku...")
        
        # Check if Heroku CLI is installed
        try:
            subprocess.run(['heroku', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Heroku CLI not found. Please install it first:")
            print("   https://devcenter.heroku.com/articles/heroku-cli")
            return False
        
        # Create Heroku app
        try:
            app_name = f"medical-fracture-detection-{os.getenv('USER', 'user')}"
            subprocess.run(['heroku', 'create', app_name], check=True)
            subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Deploy medical fracture detection API'], check=True)
            subprocess.run(['git', 'push', 'heroku', 'main'], check=True)
            
            print(f"✅ Deployed to Heroku: https://{app_name}.herokuapp.com")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Heroku deployment failed: {e}")
            return False
    
    def deploy_render(self):
        """Deploy to Render"""
        print("🎨 Deploying to Render...")
        
        # Create render.yaml
        render_config = {
            "services": [
                {
                    "type": "web",
                    "name": "medical-fracture-detection",
                    "env": "python",
                    "plan": "free",
                    "buildCommand": "pip install -r requirements.txt",
                    "startCommand": "uvicorn inference_api:app --host 0.0.0.0 --port $PORT",
                    "envVars": [
                        {"key": "PORT", "value": "8000"},
                        {"key": "PYTHONPATH", "value": "/app"}
                    ]
                }
            ]
        }
        
        with open('render.yaml', 'w') as f:
            json.dump(render_config, f, indent=2)
        
        print("✅ render.yaml created")
        print("📋 Manual steps for Render:")
        print("   1. Go to https://render.com")
        print("   2. Connect your GitHub repository")
        print("   3. Select 'Web Service'")
        print("   4. Use the settings from render.yaml")
        print("   5. Deploy!")
        
        return True
    
    def run_local(self):
        """Run locally with public access"""
        print("🏠 Running locally with public access...")
        
        try:
            # Install ngrok for public access
            try:
                subprocess.run(['ngrok', 'version'], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ ngrok not found. Please install it first:")
                print("   https://ngrok.com/download")
                return False
            
            # Start the API server
            print("🚀 Starting API server...")
            api_process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 
                'inference_api:app', 
                '--host', '0.0.0.0', 
                '--port', '8000'
            ])
            
            # Start ngrok tunnel
            print("🌐 Starting ngrok tunnel...")
            ngrok_process = subprocess.Popen(['ngrok', 'http', '8000'])
            
            print("✅ API running locally with public access!")
            print("📱 Check ngrok dashboard for public URL: http://localhost:4040")
            print("🛑 Press Ctrl+C to stop")
            
            try:
                api_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping servers...")
                api_process.terminate()
                ngrok_process.terminate()
            
            return True
            
        except Exception as e:
            print(f"❌ Local deployment failed: {e}")
            return False
    
    def deploy(self, platform='local'):
        """Deploy to specified platform"""
        print(f"🚀 Deploying Medical Fracture Detection API to {platform.upper()}")
        print("=" * 60)
        
        # Create deployment files
        self.create_requirements()
        self.create_procfile()
        self.create_runtime_txt()
        self.create_dockerfile()
        self.create_env_file()
        
        # Deploy to platform
        if platform in self.platforms:
            success = self.platforms[platform]()
            if success:
                print(f"\n🎉 Successfully deployed to {platform.upper()}!")
                print("📋 API endpoints:")
                print("   • GET  / - API information")
                print("   • GET  /health - Health check")
                print("   • POST /predict - Single image prediction")
                print("   • POST /predict_batch - Batch prediction")
                print("   • POST /predict_base64 - Base64 image prediction")
                print("   • GET  /model_info - Model information")
                print("   • GET  /config - Configuration")
                print("\n📖 API documentation available at: /docs")
            else:
                print(f"❌ Deployment to {platform} failed!")
        else:
            print(f"❌ Unknown platform: {platform}")
            print(f"Available platforms: {list(self.platforms.keys())}")

def main():
    """Main deployment function"""
    print("🏥 Medical Fracture Detection API - Deployment")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        platform = sys.argv[1].lower()
    else:
        print("Available deployment platforms:")
        print("  1. local  - Run locally with ngrok tunnel")
        print("  2. railway - Deploy to Railway")
        print("  3. heroku - Deploy to Heroku")
        print("  4. render - Deploy to Render")
        
        choice = input("\nSelect platform (1-4): ").strip()
        
        platforms = {'1': 'local', '2': 'railway', '3': 'heroku', '4': 'render'}
        platform = platforms.get(choice, 'local')
    
    deployer = APIDeployer()
    deployer.deploy(platform)

if __name__ == "__main__":
    main()
