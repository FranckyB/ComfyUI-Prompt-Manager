"""
Utility functions for managing llama.cpp models
"""
import os
import glob

from huggingface_hub import HfApi
from tqdm.auto import tqdm
import requests

# Predefined models - use real filenames as keys
QWEN_MODELS = {
    "Qwen3-1.7B-Q8_0.gguf": {
        "repo": "Qwen/Qwen3-1.7B-GGUF"
    },
    "Qwen3-4B-Q8_0.gguf": {
        "repo": "Qwen/Qwen3-4B-GGUF"
    },
    "Qwen3-8B-Q8_0.gguf": {
        "repo": "Qwen/Qwen3-8B-GGUF"
    }
}

def get_matching_mmproj(model_name):
    """
    Automatically find matching mmproj file for a given model.
    Extracts the base model name and searches for corresponding mmproj file.
    
    Example: "Qwen3-VL-4B-Thinking-Q8_0.gguf" -> "Qwen3-VL-4B-Thinking-mmproj-F16.gguf"
    """
    if not model_name:
        return None
    
    # Remove .gguf extension
    base_name = model_name.replace('.gguf', '')
    
    # Remove quantization suffix (e.g., -Q8_0, -Q4_K_M, etc.)
    # Common patterns: -Q8_0, -Q4_K_M, -Q5_K_S, etc.
    import re
    base_name = re.sub(r'-Q\d+_[KM0-9_]+$', '', base_name)
    base_name = re.sub(r'-[Ff]\d+$', '', base_name)  # Remove -F16, -f32, etc.
    
    # Get all mmproj files
    mmproj_files = get_mmproj_models()
    
    # Search for matching mmproj file
    for mmproj_file in mmproj_files:
        if base_name.lower() in mmproj_file.lower() and 'mmproj' in mmproj_file.lower():
            print(f"[Model Manager] Auto-detected mmproj: {mmproj_file} for model: {model_name}")
            return mmproj_file
    
    return None

def get_mmproj_models():
    """Get list of mmproj files in the models folder"""
    models_dir = get_models_directory()
    mmproj_files = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.gguf') and 'mmproj' in file.lower():
                mmproj_files.append(file)
    
    return sorted(mmproj_files)


def get_mmproj_path(mmproj_name):
    """Get full path to an mmproj file"""
    return os.path.join(get_models_directory(), mmproj_name)

def get_models_directory():
    """Get the path to the models directory (ComfyUI/models/gguf)"""
    # Get the custom_nodes directory (parent of this extension)
    custom_nodes_dir = os.path.dirname(os.path.dirname(__file__))
    # Go up to ComfyUI root
    comfyui_root = os.path.dirname(custom_nodes_dir)
    # Path to models/gguf
    models_dir = os.path.join(comfyui_root, "models", "gguf")

    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    return models_dir

def get_local_models():
    """Get list of local .gguf model files"""
    models_dir = get_models_directory()

    gguf_files = glob.glob(os.path.join(models_dir, "*.gguf"))
    # Return just the filenames, not full paths
    return [os.path.basename(f) for f in sorted(gguf_files)]

def get_huggingface_models():
    """Get list of predefined Qwen models available for download"""
    return list(QWEN_MODELS.keys())

def get_all_models():
    """Get combined list of local and HuggingFace models, excluding already downloaded ones"""
    local_models = get_local_models()
    models_dir = get_models_directory()

    # List all known filenames, local ones first
    all_models = []
    # Add local models first
    if local_models:
        all_models.extend(local_models)
    # Add remote models (not present locally)
    for filename in QWEN_MODELS.keys():
        if filename not in all_models:
            all_models.append(filename)
    return all_models

def is_model_local(model_name):
    """Check if a model exists locally"""
    if model_name == "--- Download from HuggingFace ---":
        return False

    models_dir = get_models_directory()
    model_path = os.path.join(models_dir, model_name)
    return os.path.exists(model_path)

def get_model_path(model_name):
    """Get the full path to a model file"""
    models_dir = get_models_directory()
    return os.path.join(models_dir, model_name)


def download_model(model_name):
    """
    Download a model from HuggingFace with automatic progress display

    Args:
        model_name: Filename of the model (e.g., "Qwen3-4B-Q8_0.gguf")

    Returns:
        Path to downloaded model or None on error
    """

    if model_name not in QWEN_MODELS:
        print(f"[Model Manager] Error: Unknown model: {model_name}")
        return None

    model_info = QWEN_MODELS[model_name]
    repo_id = model_info["repo"]
    filename = model_name
    models_dir = get_models_directory()
    local_path = os.path.join(models_dir, filename)

    # Check if already downloaded
    if os.path.exists(local_path):
        print(f"[Model Manager] Model already exists: {filename}")
        return local_path

    try:
        print(f"[Model Manager] Downloading {filename} from {repo_id}...")
        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
        file_info = next((f for f in repo_info.siblings if f.rfilename == filename), None)
        if not file_info or file_info.size is None:
            print(f"[Model Manager] Could not find file size for {filename} in {repo_id}")
            return None
        total_size = file_info.size
        download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")
        with requests.get(download_url, stream=True) as r, open(local_path, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()
        print(f"[Model Manager] Successfully downloaded {filename}")
        return local_path
    except Exception as e:
        print(f"[Model Manager] Error downloading model: {e}")
        return None
