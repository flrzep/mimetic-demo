#!/usr/bin/env python3
"""
Deployment script for Modal services
Run this to deploy both CV inference and WebRTC streaming to Modal
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, cwd: str) -> bool:
    """Run a command and return success status"""
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def deploy_modal_services():
    """Deploy all Modal services"""
    script_dir = Path(__file__).parent
    
    print("🚀 Deploying Modal Services...")
    print("=" * 50)
    
    # Deploy main CV inference service
    print("\n📸 Deploying CV Inference Service...")
    if not run_command("modal deploy modal_service.py", cwd=script_dir):
        print("❌ Failed to deploy CV inference service")
        return False
    
    # Deploy WebRTC streaming service
    print("\n📹 Deploying WebRTC Streaming Service...")
    if not run_command("modal deploy webrtc_service.py", cwd=script_dir):
        print("❌ Failed to deploy WebRTC streaming service")
        return False
    
    print("\n✅ All Modal services deployed successfully!")
    print("\n📋 Next steps:")
    print("1. Get your Modal app URLs:")
    print("   modal app list")
    print("2. Update your .env file with:")
    print("   USE_MOCK_MODAL=false")
    print("   MODAL_ENDPOINT_URL=<your-cv-inference-url>")
    print("   MODAL_WEBRTC_URL=<your-webrtc-streaming-url>")
    print("3. Restart your backend")
    
    return True

if __name__ == "__main__":
    if not deploy_modal_services():
        sys.exit(1)
