"""
Utility functions for managing llama.cpp models
"""
import os
import glob
import folder_paths

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
    },
    "Qwen3VL-4B-Instruct-Q8_0.gguf": {
        "repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "mmproj": "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
    },
    "Qwen3VL-8B-Instruct-Q8_0.gguf": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "mmproj": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
    }
}

def get_models_directory():
    """Get the path to the primary models directory (ComfyUI/models/gguf) for downloads"""
    # Register both gguf and LLM folders
    if "gguf" not in folder_paths.folder_names_and_paths:
        gguf_dir = os.path.join(folder_paths.models_dir, "gguf")
        folder_paths.add_model_folder_path("gguf", gguf_dir)

    if "LLM" not in folder_paths.folder_names_and_paths:
        llm_dir = os.path.join(folder_paths.models_dir, "LLM")
        folder_paths.add_model_folder_path("LLM", llm_dir)

    # Return gguf as the primary directory (for downloads)
    models_dir = folder_paths.get_folder_paths("gguf")[0]

    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    return models_dir

def get_all_model_directories():
    """Get all directories where models can be found"""
    get_models_directory()  # Ensure folders are registered

    directories = []
    for folder_type in ["gguf", "LLM"]:
        if folder_type in folder_paths.folder_names_and_paths:
            dirs = folder_paths.get_folder_paths(folder_type)
            directories.extend(dirs)

    return directories

def get_local_models():
    """Get list of local .gguf model files from all model directories"""
    all_dirs = get_all_model_directories()

    all_models = set()  # Use set to avoid duplicates
    for models_dir in all_dirs:
        if os.path.exists(models_dir):
            gguf_files = glob.glob(os.path.join(models_dir, "*.gguf"))
            for f in gguf_files:
                all_models.add(os.path.basename(f))

    # Return sorted list of unique filenames
    return sorted(list(all_models))

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
    """Check if a model exists locally in any of the model directories"""
    if model_name == "--- Download from HuggingFace ---":
        return False

    all_dirs = get_all_model_directories()
    for models_dir in all_dirs:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            return True

    return False

def get_model_path(model_name):
    """Get the full path to a model file, searching all model directories"""
    all_dirs = get_all_model_directories()

    # Check each directory for the model
    for models_dir in all_dirs:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            return model_path

    # If not found, return path in primary directory (gguf)
    models_dir = get_models_directory()
    return os.path.join(models_dir, model_name)

def get_mmproj_for_model(model_name):
    """Get the mmproj filename for a given vision model"""
    if model_name in QWEN_MODELS and "mmproj" in QWEN_MODELS[model_name]:
        return QWEN_MODELS[model_name]["mmproj"]
    return None

def get_mmproj_path(model_name):
    """Get the path to the mmproj file for a vision model, if it exists"""
    mmproj_name = get_mmproj_for_model(model_name)
    if mmproj_name and is_model_local(mmproj_name):
        return get_model_path(mmproj_name)
    return None

def download_model(model_name):
    """Download a model from HuggingFace with automatic progress display
    For VL models, also downloads the matching mmproj file if needed

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
    models_dir = get_models_directory()

    # Build list of files to download
    files_to_download = [
        {"filename": model_name, "desc": "Downloading model"}
    ]

    # Add mmproj file if this is a VL model
    mmproj_name = get_mmproj_for_model(model_name)
    if mmproj_name:
        files_to_download.append({
            "filename": mmproj_name,
            "desc": "Downloading mmproj"
        })

    # Download each file if it doesn't exist
    for file_info in files_to_download:
        filename = file_info["filename"]
        local_path = os.path.join(models_dir, filename)

        if os.path.exists(local_path):
            print(f"[Model Manager] File already exists: {filename}")
            continue

        # Download the file
        try:
            print(f"[Model Manager] Downloading {filename} from {repo_id}...")
            api = HfApi()
            repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
            file_metadata = next((f for f in repo_info.siblings if f.rfilename == filename), None)
            if not file_metadata or file_metadata.size is None:
                print(f"[Model Manager] Could not find file size for {filename} in {repo_id}")
                return None
            total_size = file_metadata.size
            download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=file_info["desc"])
            with requests.get(download_url, stream=True) as r, open(local_path, "wb") as f:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            pbar.close()
            print(f"[Model Manager] Successfully downloaded {filename}")
        except Exception as e:
            print(f"[Model Manager] Error downloading {filename}: {e}")
            return None

    return os.path.join(models_dir, model_name)
