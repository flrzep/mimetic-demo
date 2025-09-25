#!/usr/bin/env python3
"""
Model management script for Modal CV service
Use this script to download, upload, and manage models in Modal storage
"""

import os
import shutil
import json
from pathlib import Path

import modal
from huggingface_hub import hf_hub_download

# Use the same app configuration as the main service
def get_app_name():
    github_ref = os.getenv("GITHUB_REF", "")
    if github_ref:
        if github_ref.startswith("refs/heads/"):
            branch = github_ref[11:]
            if branch == "main" or branch.startswith("deploy/"):
                return "mimetic-demo"
            else:
                safe_branch = branch.replace("/", "-").replace("_", "-").replace(".", "-")
                return f"mimetic-demo-{safe_branch}"
    return os.getenv("MODAL_APP_NAME", "mimetic-demo-dev")

app_name = get_app_name()

# Lightweight image for model management (no GPU needed)
management_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "huggingface-hub",
        "torch", 
        "transformers"
    ])
    .add_local_dir("models", remote_path="/app/models")
    .add_local_file("model_config.json", remote_path="/app/model_config.json")
)

# Create management app
management_app = modal.App(f"{app_name}-models", image=management_image)

# Use the same volume as the main app
model_volume = modal.Volume.from_name("mimetic-models", create_if_missing=True)

MODEL_DIR = Path("/models")

@management_app.function(
    image=management_image,
    volumes={"/models": model_volume},
    timeout=3600
)
def download_yolo_model():
    """Download YOLO model weights to Modal storage"""
    print("Downloading YOLO model weights...")
    
    # Download directly to models directory
    models_yolo_dir = MODEL_DIR / "yolo"
    models_yolo_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = hf_hub_download(
        repo_id="onnx-community/yolov10n",
        filename="onnx/model.onnx",
        cache_dir=str(models_yolo_dir),
    )
    
    # Copy the downloaded model to a consistent name
    destination = models_yolo_dir / "yolov10n.onnx"
    shutil.copy2(model_file, destination)
    
    # Copy the class names file from the local models directory to the volume
    local_classes_file = Path("/app/models/yolo/yolo_classes.txt")
    if local_classes_file.exists():
        shutil.copy2(local_classes_file, models_yolo_dir / "yolo_classes.txt")
        print(f"Class names copied from {local_classes_file}")
    else:
        print("Warning: /app/models/yolo/yolo_classes.txt not found")
    
    print(f"✅ YOLO model weights saved to {destination}")
    print(f"✅ Class names saved to {models_yolo_dir / 'yolo_classes.txt'}")
    return str(destination)

@management_app.function(
    image=management_image,
    volumes={"/models": model_volume},
    timeout=3600
)
def download_custom_model(repo_id: str, filename: str, model_name: str):
    """Download any custom model from HuggingFace to Modal storage"""
    print(f"Downloading {model_name} from {repo_id}...")
    
    # Download directly to models directory
    models_dir = MODEL_DIR / model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(models_dir),
    )
    
    # Copy the downloaded model to a consistent name
    destination = models_dir / Path(filename).name
    shutil.copy2(model_file, destination)
    
    print(f"✅ {model_name} model saved to {destination}")
    return str(destination)

@management_app.function(
    image=management_image,
    volumes={"/models": model_volume}
)
def list_models():
    """List all models in Modal storage"""
    print("📋 Listing models in Modal storage...")
    
    results = {"models": {}}
    
    # Check /models volume
    if MODEL_DIR.exists():
        for model_dir in MODEL_DIR.iterdir():
            if model_dir.is_dir():
                files = []
                total_size = 0
                for file in model_dir.rglob("*"):
                    if file.is_file():
                        size = file.stat().st_size
                        files.append({
                            "name": file.name,
                            "path": str(file.relative_to(MODEL_DIR)),
                            "size_mb": round(size / (1024 * 1024), 2)
                        })
                        total_size += size
                
                results["models"][model_dir.name] = {
                    "files": files,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }
                
                print(f"📂 {model_dir.name}:")
                for file_info in files:
                    print(f"  📄 {file_info['name']} ({file_info['size_mb']} MB)")
    
    return results

@management_app.function(
    image=management_image,
    volumes={"/models": model_volume}
)
def remove_model(model_name: str):
    """Remove a specific model from storage"""
    print(f"🗑️ Removing model: {model_name}")
    
    model_path = MODEL_DIR / model_name
    if model_path.exists():
        shutil.rmtree(model_path)
        print(f"✅ Removed {model_name}")
    else:
        print(f"❌ Model {model_name} not found")


@management_app.function(
    image=management_image,
    volumes={"/models": model_volume},
    timeout=3600
)
def upload_local_models():
    """Upload all local model files to Modal storage"""
    print("📤 Uploading local models to Modal storage...")
    
    local_models_dir = Path("/app/models")
    if not local_models_dir.exists():
        print("❌ Local models directory not found")
        return False
    
    success_count = 0
    total_count = 0
    
    # Upload each model folder
    for model_folder in local_models_dir.iterdir():
        if not model_folder.is_dir() or model_folder.name.startswith('.'):
            continue
            
        print(f"\n📁 Uploading model folder: {model_folder.name}")
        
        # Create target directory in Modal volume
        target_dir = Path(f"/models/{model_folder.name}")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Upload all files in the model folder
        for file_path in model_folder.rglob("*"):
            if file_path.is_file():
                total_count += 1
                relative_path = file_path.relative_to(model_folder)
                target_path = target_dir / relative_path
                
                try:
                    # Create parent directories if needed
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file to Modal volume
                    import shutil
                    shutil.copy2(file_path, target_path)
                    
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                    print(f"  ✅ {relative_path} ({file_size:.2f} MB)")
                    success_count += 1
                    
                except Exception as e:
                    print(f"  ❌ {relative_path}: {e}")
    
    print(f"\n📊 Upload Summary:")
    print(f"  ✅ Successfully uploaded: {success_count}/{total_count} files")
    
    if success_count == total_count:
        print("🎉 All models uploaded successfully!")
        return True
    else:
        print("⚠️  Some files failed to upload")
        return False


@management_app.function(
    image=management_image,
    volumes={"/models": model_volume},
    timeout=3600
)
def upload_model_config():
    """Upload model_config.json to Modal storage"""
    print("📤 Uploading model_config.json to Modal storage...")
    
    local_config = Path("/app/model_config.json")
    if not local_config.exists():
        print("❌ model_config.json not found locally")
        return False
    
    try:
        target_path = Path("/models/model_config.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(local_config, target_path)
        
        file_size = local_config.stat().st_size / 1024  # KB
        print(f"✅ model_config.json uploaded ({file_size:.2f} KB)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload model_config.json: {e}")
        return False


@management_app.function(
    image=management_image,
    volumes={"/models": model_volume},
    timeout=3600
)
def validate_model_config():
    """Validate that model_config.json matches actual model folders and files"""
    print("🔍 Validating model configuration...")
    
    # Load model config
    config_path = Path("/models/model_config.json")
    if not config_path.exists():
        print("❌ model_config.json not found")
        return False
    
    with open(config_path) as f:
        config = json.load(f)
    
    configured_models = config.get("models", {})
    print(f"📄 Models in config: {sorted(configured_models.keys())}")
    
    # Find actual model folders
    models_dir = Path("/models")
    actual_folders = set()
    
    for item in models_dir.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            actual_folders.add(item.name)
    
    print(f"📁 Folders found: {sorted(actual_folders)}")
    
    issues_found = False
    
    # Validate each configured model
    for model_id, model_config in configured_models.items():
        print(f"\n🔍 Validating model: {model_id}")
        
        files_info = model_config.get("files", {})
        folder = files_info.get("folder", model_id)
        weights_file = files_info.get("weights")
        classes_file = files_info.get("classes")
        
        # Check if folder exists
        folder_path = models_dir / folder
        if not folder_path.exists():
            print(f"❌ Folder missing: {folder}")
            issues_found = True
            continue
        
        print(f"✅ Folder exists: {folder}")
        
        # Check if weights file exists
        if weights_file:
            weights_path = folder_path / weights_file
            if weights_path.exists():
                print(f"✅ Weights file exists: {weights_file}")
            else:
                print(f"❌ Weights file missing: {weights_file}")
                issues_found = True
        
        # Check if classes file exists  
        if classes_file:
            classes_path = folder_path / classes_file
            if classes_path.exists():
                print(f"✅ Classes file exists: {classes_file}")
            else:
                print(f"❌ Classes file missing: {classes_file}")
                issues_found = True
    
    # Check for folders without config
    configured_folders = {model_config.get("files", {}).get("folder", model_id) 
                         for model_id, model_config in configured_models.items()}
    unconfigured_folders = actual_folders - configured_folders
    
    if unconfigured_folders:
        print(f"\n⚠️  Folders without config: {sorted(unconfigured_folders)}")
        issues_found = True
    
    if not issues_found:
        print("\n✅ All models properly configured!")
        return True
    else:
        print("\n❌ Configuration validation failed")
        return False


@management_app.function(
    image=management_image,
    volumes={"/models": model_volume},
    timeout=3600
)
def setup_modal_storage():
    """Complete setup of Modal storage - upload all local models and config"""
    print("🚀 Setting up Modal storage from local files...")
    print()
    
    # Upload all local model folders
    print("📂 Uploading local model folders...")
    if not upload_local_models.remote():
        print("❌ Failed to upload local models")
        return False
    
    print()
    print("📄 Uploading model configuration...")
    if not upload_model_config.remote():
        print("❌ Failed to upload model configuration")
        return False
    
    print()
    print("🔍 Validating uploaded configuration...")
    if not validate_model_config.remote():
        print("❌ Configuration validation failed")
        return False
    
    print()
    print("✅ Modal storage setup completed successfully!")
    print("📋 Use 'modal run manage_models.py::list_models' to verify")
    return True


if __name__ == "__main__":
    import sys
    
    print(f"🔧 Model Management for {app_name}")
    print()
    print("Available commands:")
    print("  📥 Download YOLO model:")
    print("    modal run manage_models.py::download_yolo_model")
    print()
    print("  📥 Download custom model:")
    print("    modal run manage_models.py::download_custom_model --repo-id 'owner/repo' --filename 'model.onnx' --model-name 'my-model'")
    print()
    print("  📋 List all models:")
    print("    modal run manage_models.py::list_models")
    print()
    print("  � Validate configuration:")
    print("    modal run manage_models.py::validate_model_config")
    print()
    print("  �🗑️ Remove specific model:")
    print("    modal run manage_models.py::remove_model --model-name 'yolo'")