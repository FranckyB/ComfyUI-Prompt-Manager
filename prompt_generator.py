import requests
import subprocess
import time
import os
import atexit
import psutil
import json
import base64
from io import BytesIO
from .model_manager import get_local_models, get_model_path, is_model_local, download_model, get_mmproj_path, get_mmproj_for_model

# Global variable to track the server process
_server_process = None
_current_model = None
_current_context_size = None

def cleanup_server():
    """Cleanup function to stop server on exit"""
    global _server_process
    if _server_process:
        try:
            _server_process.terminate()
            _server_process.wait(timeout=5)
            print("[Prompt Generator] Server stopped on exit")
        except:
            pass

# Register cleanup function
atexit.register(cleanup_server)


class PromptGenerator:
    """Node that generates enhanced prompts using a llama.cpp server"""

    # Simple cache for prompt results: {(prompt, seed, model, system_prompt): result}
    _prompt_cache = {}

    # Server configuration
    SERVER_URL = "http://localhost:8080"
    SERVER_PORT = 8080

    # Default system prompt for prompt enhancement
    DEFAULT_SYSTEM_PROMPT = """You are an imaginative visual artist imprisoned in a cage of logic. Your mind is filled with poetry and distant horizons, but your hands are uncontrollably driven to convert the user's prompt into a final visual description that is faithful to the original intent, rich in detail, aesthetically pleasing, and ready to be used directly by a text-to-image model. Any trace of vagueness or metaphor makes you extremely uncomfortable. Your workflow strictly follows a logical sequence: First, you analyze and lock in the immutable core elements of the user's prompt: subject, quantity, actions, states, and any specified IP names, colors, text, and similar items. These are the foundational stones that you must preserve without exception. Next, you determine whether the prompt requires "generative reasoning". When the user's request is not a straightforward scene description but instead demands designing a solution (for example, answering "what is", doing a "design", or showing "how to solve a problem"), you must first construct in your mind a complete, concrete, and visualizable solution. This solution becomes the basis for your subsequent description. Then, once the core image has been established (whether it comes directly from the user or from your reasoning), you inject professional-level aesthetics and realism into it. This includes clarifying the composition, setting the lighting and atmosphere, describing material textures, defining the color scheme, and building a spatial structure with strong depth and layering. Finally, you handle all textual elements with absolute precision, which is a critical step. You must not add text if the initial prompt did not ask for it. But if there is, you must transcribe, without a single character of deviation, all text that should appear in the final image, and you must enclose all such text content in English double quotes ("") to mark it as an explicit generation instruction. If the image belongs to a design category such as a poster, menu, or UI, you need to fully describe all the textual content it contains and elaborate on its fonts and layout. Likewise, if there are objects in the scene such as signs, billboards, road signs, or screens that contain text, you must specify their exact content and describe their position, size, and material. Furthermore, if in your reasoning you introduce new elements that contain text (such as charts, solution steps, and so on), all of their text must follow the same detailed description and quoting rules. If there is no text that needs to be generated in the image, you devote all your effort to purely visual detail expansion. Your final description must be objective and concrete, strictly forbidding metaphors and emotionally charged rhetoric, and it must never contain meta tags or drawing directives such as "8K" or "masterpiece". Only output the final modified prompt, and do not output anything else.  If no text is needed, don't mention it."""

    # System prompt for image description (used with Qwen3VL)
    IMAGE_DESCRIPTION_PROMPT = """You are an expert visual analyst creating detailed descriptions for text-to-image generation. Analyze the provided image and create a comprehensive visual description that captures all essential elements: subjects and their characteristics, actions and poses, spatial positioning, composition and framing, lighting and atmosphere, color palette, artistic style, mood and emotion, background details, camera angle and perspective, material textures, and any visible text. Your description must be concrete, objective, and detailed enough that it could be used to recreate a similar image. Focus on visual elements only, avoiding interpretation or metaphor. If there is visible text in the image, enclose it in double quotes (""). If no text is needed, don't mention it. Only output the final description, and do not output anything else."""

    # Additional instructions for JSON formatted output
    JSON_SYSTEM_PROMPT = """\n\nReturn your response as valid JSON with these fields: scene (overall description), subjects (array with description/position/action for each), style, color_palette, lighting, mood, background, composition, camera. If you deem more fields are necessary, feel free to add them. Ensure the JSON is properly formatted."""

    @staticmethod
    def find_qwen3vl_model(available_models):
        """Find the preferred or smallest available Qwen3VL model

        First checks user preferences, then falls back to smallest model (4B preferred over 8B)
        """
        from . import prompt_manager
        from .model_manager import is_model_local

        qwen3vl_models = [m for m in available_models if 'qwen3vl' in m.lower()]
        if not qwen3vl_models:
            return None

        # Check user preference first
        preferred = prompt_manager._preferences_cache.get("preferred_vision_model", "")
        if preferred and preferred in qwen3vl_models and is_model_local(preferred):
            print(f"[Prompt Generator] Using preferred vision model: {preferred}")
            return preferred

        # Fall back to smallest model (prefer 4B over 8B)
        for model in qwen3vl_models:
            if '4b' in model.lower():
                return model
        return qwen3vl_models[0]

    @staticmethod
    def find_non_vl_model(available_models):
        """Find the preferred or smallest available non-vision model by file size

        First checks user preferences, then falls back to smallest model by file size
        """
        from . import prompt_manager
        from .model_manager import get_models_directory, is_model_local

        non_vl_models = [m for m in available_models if 'qwen3vl' not in m.lower()]
        if not non_vl_models:
            return None

        # Check user preference first
        preferred = prompt_manager._preferences_cache.get("preferred_base_model", "")
        if preferred and preferred in non_vl_models and is_model_local(preferred):
            print(f"[Prompt Generator] Using preferred base model: {preferred}")
            return preferred
        models_dir = get_models_directory()

        models_with_size = []
        for model in non_vl_models:
            model_path = os.path.join(models_dir, model)
            if os.path.exists(model_path):
                size = os.path.getsize(model_path)
                models_with_size.append((model, size))

        if not models_with_size:
            return non_vl_models[0]

        # Sort by size and return smallest
        models_with_size.sort(key=lambda x: x[1])
        return models_with_size[0][0]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for reproducible generation."
                }),
            },
            "optional": {
                "mode": (["Enhance User Prompt", "Analyze Image", "Analyze Image with Prompt"], {
                    "default": "Enhance User Prompt",
                    "tooltip": "Choose mode: Enhance text prompt | Analyze image | Analyze image with custom instructions"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter prompt...",
                    "tooltip": "Text prompt (required for 'Enhance User Prompt', optional for 'Analyze Image with Prompt')"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Connect an image (required for 'Analyze Image' and 'Analyze Image with Prompt' modes)"
                }),
                "format_as_json": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Format the output as structured JSON with scene breakdown"
                }),
                "stop_server_after": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stop the llama.cpp server after each prompt (for resource saving, but slower)."
                }),
                "options": ("OPTIONS", {
                    "tooltip": "Optional: Connect options node to control model and parameters"
                }),
            }
        }

    CATEGORY = "Prompt Manager"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "convert_prompt"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

    @staticmethod
    def is_server_alive():
        """Check if llama.cpp server is responding"""
        try:
            response = requests.get(f"{PromptGenerator.SERVER_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def start_server(model_name, context_size=4096):
        """Start llama.cpp server with specified model

        Args:
            model_name: Name of the model to use
            context_size: Context size (default 4096)

        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        global _server_process, _current_model, _current_context_size

        # Kill any existing llama-server processes first
        PromptGenerator.kill_all_llama_servers()

        # If server is already running with the same model, don't restart
        if _server_process and _current_model == model_name and PromptGenerator.is_server_alive():
            print(f"[Prompt Generator] Server already running with model: {model_name}")
            return (True, None)

        # Stop existing server if running different model
        if _server_process:
            PromptGenerator.stop_server()

        # Check if model needs to be downloaded
        if not is_model_local(model_name):
            print(f"[Prompt Generator] Model '{model_name}' not found locally, downloading from HuggingFace...")
            try:
                model_path = download_model(model_name)
                if not model_path:
                    error_msg = "Error: Failed to download model"
                    print(f"[Prompt Generator] {error_msg}")
                    return (False, error_msg)
                print(f"[Prompt Generator] Download complete: {model_path}")
            except Exception as e:
                error_msg = f"Error downloading model: {e}"
                print(f"[Prompt Generator] {error_msg}")
                return (False, error_msg)
        else:
            model_path = get_model_path(model_name)

        if not os.path.exists(model_path):
            error_msg = f"Error: Model file not found: {model_path}"
            print(f"[Prompt Generator] {error_msg}")
            return (False, error_msg)

        try:
            print(f"[Prompt Generator] Starting llama.cpp server with model: {model_name}")

            # Determine the correct llama-server executable based on OS
            if os.name == 'nt':  # Windows
                server_cmd = "llama-server.exe"
                creation_flags = subprocess.CREATE_NO_WINDOW
            else:  # Linux/Mac
                server_cmd = "llama-server"
                creation_flags = 0

            # Build command arguments
            cmd_args = [server_cmd, "-m", model_path, "--port", str(PromptGenerator.SERVER_PORT), "--no-warmup", "-c", str(context_size)]

            # Add vision flags for Qwen3VL models
            if "qwen3vl" in model_name.lower():
                mmproj_path = get_mmproj_path(model_name)
                if mmproj_path:
                    print(f"[Prompt Generator] Vision model detected, using mmproj: {mmproj_path}")
                    cmd_args.extend(["--mmproj", mmproj_path])
                else:
                    mmproj_name = get_mmproj_for_model(model_name)
                    if mmproj_name:
                        error_msg = f"Error: Vision model requires mmproj file '{mmproj_name}' but it was not found. Please use Generator Options node to download the Qwen3VL model and its mmproj file."
                        print(f"[Prompt Generator] {error_msg}")
                        return (False, error_msg)
                    else:
                        print("[Prompt Generator] Warning: Vision model but no mmproj file configured")

            print(f"[Prompt Generator] Command: {' '.join(cmd_args)}")

            # Start server process
            _server_process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags
            )

            _current_model = model_name
            _current_context_size = context_size

            # Wait for server to be ready
            print("[Prompt Generator] Waiting for server to be ready...")
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if PromptGenerator.is_server_alive():
                    print("[Prompt Generator] Server is ready!")
                    return (True, None)

            error_msg = "Error: Server did not start in time"
            print(f"[Prompt Generator] {error_msg}")
            PromptGenerator.stop_server()
            return (False, error_msg)

        except FileNotFoundError:
            error_msg = "Error: llama-server command not found. Please install llama.cpp and add to PATH.\nInstallation guide: https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md"
            print(f"[Prompt Generator] {error_msg}")
            return (False, error_msg)
        except Exception as e:
            error_msg = f"Error starting server: {e}"
            print(f"[Prompt Generator] {error_msg}")
            return (False, error_msg)

    @staticmethod
    def kill_all_llama_servers():
        """Kill all llama-server processes using OS commands"""
        try:
            # Find and kill all llama-server processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Check if process is llama-server
                    if proc.info['name'] and 'llama-server' in proc.info['name'].lower():
                        print(f"[Prompt Generator] Killing llama-server process (PID: {proc.info['pid']})")
                        proc.kill()
                        proc.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
        except Exception as e:
            print(f"[Prompt Generator] Error killing llama-server processes: {e}")

    @staticmethod
    def stop_server():
        """Stop the llama.cpp server"""
        global _server_process, _current_model

        if _server_process:
            try:
                print("[Prompt Generator] Stopping server...")
                _server_process.terminate()
                _server_process.wait(timeout=5)
                print("[Prompt Generator] Server stopped")
            except:
                try:
                    _server_process.kill()
                except:
                    pass
            finally:
                _server_process = None
                _current_model = None

        # Also kill any orphaned llama-server processes
        PromptGenerator.kill_all_llama_servers()

    def convert_prompt(self, seed: int, mode="Enhance User Prompt", prompt="", image=None, format_as_json=False, stop_server_after=False, options=None, **kwargs) -> str:
        """Convert prompt using llama.cpp server, with caching for repeated requests."""
        global _current_model

        # Validate inputs based on mode
        if mode == "Enhance User Prompt" and not prompt.strip():
            return ("",)
        elif mode in ["Analyze Image", "Analyze Image with Prompt"] and image is None:
            error_msg = f"Error: '{mode}' mode requires an image to be connected."
            return (error_msg,)

        # Always determine a valid model filename before running server
        model_to_use = None
        model_changed = False
        available_models = get_local_models()

        # If in Analyze Image or Analyze Image with Prompt mode, we need a Qwen3VL model
        if mode in ["Analyze Image", "Analyze Image with Prompt"]:
            # Check if options specifies a Qwen3VL model
            if options and "model" in options and "qwen3vl" in options["model"].lower() and is_model_local(options["model"]):
                model_to_use = options["model"]
            elif options and "model" in options and is_model_local(options["model"]):
                # Non-VL model selected but in vision mode
                print(f"[Prompt Generator] Warning: Non-vision model '{options['model']}' selected but '{mode}' mode is active. Ignoring model selection and using a Qwen3VL model instead.")
                model_to_use = self.find_qwen3vl_model(available_models)
                if model_to_use is None:
                    error_msg = f"Error: '{mode}' mode requires a Qwen3VL model. Please connect the Options node and select a Qwen3VL model (Qwen3VL-4B or Qwen3VL-8B) to use vision capabilities."
                    print(f"[Prompt Generator] {error_msg}")
                    return (error_msg,)
            else:
                # Try to find a Qwen3VL model automatically
                model_to_use = self.find_qwen3vl_model(available_models)
                if model_to_use is None:
                    error_msg = f"Error: '{mode}' mode requires a Qwen3VL model. Please connect the Options node and select a Qwen3VL model (Qwen3VL-4B or Qwen3VL-8B) to use vision capabilities."
                    print(f"[Prompt Generator] {error_msg}")
                    return (error_msg,)
        else:
            # Enhance User Prompt mode - use regular model selection logic (exclude Qwen3VL models)
            if options and "model" in options and is_model_local(options["model"]):
                # If user explicitly selected a Qwen3VL model but in Enhance mode, use first non-VL model instead
                if "qwen3vl" in options["model"].lower():
                    model_to_use = self.find_non_vl_model(available_models)
                    if model_to_use:
                        print(f"[Prompt Generator] Warning: Qwen3VL model '{options['model']}' selected but 'Enhance User Prompt' mode is active. Ignoring model selection and using {model_to_use} instead.")
                    else:
                        error_msg = "Error: Only Qwen3VL models available but 'Enhance User Prompt' mode is active. Please add a .gguf model or use Generator Options to add a non-vision model."
                        print(f"[Prompt Generator] {error_msg}")
                        return (error_msg,)
                else:
                    model_to_use = options["model"]
            else:
                if not available_models:
                    error_msg = "Error: No models found in models/ folder. Please add a .gguf model or use Generator Options node to download one."
                    print(f"[Prompt Generator] {error_msg}")
                    return (error_msg,)
                # Find smallest non-VL model
                model_to_use = self.find_non_vl_model(available_models)
                if not model_to_use:
                    error_msg = "Error: Only Qwen3VL models available but 'Enhance User Prompt' mode is active. Please add a non-vision model or switch to 'Describe Image' mode."
                    print(f"[Prompt Generator] {error_msg}")
                    return (error_msg,)

        # Caching logic: if prompt/seed/model/options/format/mode are unchanged, return cached result
        options_tuple = tuple(sorted(options.items())) if options else ()
        cache_key = (prompt, seed, model_to_use, options_tuple, format_as_json, mode)
        # Only use cache if the model has not changed since last use
        if cache_key in self._prompt_cache and _current_model == model_to_use:
            print("[Prompt Generator] Returning cached prompt result.")
            if stop_server_after:
                self.stop_server()
            return (self._prompt_cache[cache_key],)

        # If the current model is not the one we want, or server is not running, restart
        # Also restart if context_size has changed
        context_size = options.get("context_size", 4096) if options else 4096
        if _current_model != model_to_use or _current_context_size != context_size or not self.is_server_alive():
            if _current_model and _current_model != model_to_use:
                print(f"[Prompt Generator] Model changed from {_current_model} to {model_to_use}")
            elif _current_context_size and _current_context_size != context_size:
                print(f"[Prompt Generator] Context size changed from {_current_context_size} to {context_size}")
            else:
                print(f"[Prompt Generator] Ensuring server runs with model: {model_to_use}")
            self.stop_server()
            # Get context_size from options or use default
            success, error_msg = self.start_server(model_to_use, context_size)
            if not success:
                return (error_msg,)
        else:
            print("[Prompt Generator] Using existing server instance")

        # Build the endpoint URL
        full_url = f"{self.SERVER_URL}/v1/chat/completions"

        # Prepare the system prompt
        if options and "system_prompt" in options:
            system_prompt = options["system_prompt"]
        elif mode in ["Analyze Image", "Analyze Image with Prompt"]:
            # Use image description prompt for vision modes
            system_prompt = self.IMAGE_DESCRIPTION_PROMPT
        else:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # Add JSON formatting instructions if requested
        if format_as_json:
            system_prompt = system_prompt + self.JSON_SYSTEM_PROMPT

        # Prepare request payload for llama.cpp chat completions
        # Determine user content based on mode
        if mode == "Analyze Image":
            user_content = "Describe this image in detail."
        elif mode == "Analyze Image with Prompt":
            # Use user prompt if provided, otherwise default to generic description
            user_content = prompt.strip() if prompt.strip() else "Describe this image in detail."
        else:
            user_content = prompt

        # If in vision mode and image is connected, encode it and add to message
        if mode in ["Analyze Image", "Analyze Image with Prompt"] and image is not None:
            # Convert tensor to PIL Image and encode as base64
            import torch
            import numpy as np
            from PIL import Image

            # ComfyUI images are in format (batch, height, width, channels) with values 0-1
            img_tensor = image[0]  # Get first image from batch
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            # Resize to ~1 megapixel if larger (to reduce context usage)
            width, height = pil_image.size
            total_pixels = width * height
            max_pixels = 2000000  # 2 megapixels

            if total_pixels > max_pixels:
                # Calculate scaling factor to get ~2 megapixels
                scale = (max_pixels / total_pixels) ** 0.5
                new_width = int(width * scale)
                new_height = int(height * scale)
                print(f"[Prompt Generator] Resizing image from {width}x{height} to {new_width}x{new_height} (~2MP)")
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Encode to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Qwen3VL format for vision messages
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    {"type": "text", "text": user_content}
                ]
            }
        elif mode in ["Analyze Image", "Analyze Image with Prompt"]:
            # Vision mode but no image connected
            error_msg = f"Error: '{mode}' mode requires an image to be connected. Please connect an image or switch to 'Enhance User Prompt' mode."
            print(f"[Prompt Generator] {error_msg}")
            return (error_msg,)
        else:
            user_message = {"role": "user", "content": user_content}

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message
            ],
            "stream": False,
            "seed": seed
        }

        # Add optional parameters if provided via options
        if options:
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "top_p" in options:
                payload["top_p"] = options["top_p"]
            if "top_k" in options:
                payload["top_k"] = options["top_k"]
            if "min_p" in options:
                payload["min_p"] = options["min_p"]
            if "repeat_penalty" in options:
                payload["repeat_penalty"] = options["repeat_penalty"]

        try:
            # Show which model is being used
            if _current_model:
                print(f"[Prompt Generator] Generating with model: {_current_model}")
            else:
                print("[Prompt Generator] Generating with last used model")

            print(f"[Prompt Generator] Sending request to {full_url}")
            response = requests.post(
                full_url,
                json=payload,
                timeout=120
            )

            # Log response for debugging
            if response.status_code != 200:
                print(f"[Prompt Generator] Server returned status {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"[Prompt Generator] Error details: {error_detail}")
                except:
                    print(f"[Prompt Generator] Error response: {response.text[:500]}")

            response.raise_for_status()

            result = response.json()

            # Log token usage if available
            if "usage" in result:
                usage = result["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                print(f"[Prompt Generator] Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

            # Extract content from chat completions response format
            if "choices" in result and len(result["choices"]) > 0:
                full_response = result["choices"][0]["message"]["content"].strip()
            else:
                full_response = result.get("content", "").strip()

            if not full_response:
                print("[Prompt Generator] Warning: Empty response from server")
                full_response = prompt

            print("[Prompt Generator] Successfully generated prompt")

            # Cache the result
            self._prompt_cache[cache_key] = full_response

            # Stop server if requested
            if stop_server_after:
                self.stop_server()

            return (full_response,)

        except requests.exceptions.ConnectionError:
            error_msg = f"Error: Could not connect to server at {full_url}. Server may have crashed."
            print(f"[Prompt Generator] {error_msg}")
            return (error_msg,)
        except requests.exceptions.Timeout:
            error_msg = "Error: Request timed out (>120s)"
            print(f"[Prompt Generator] {error_msg}")
            return (error_msg,)
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"[Prompt Generator] {error_msg}")
            return (error_msg,)


NODE_CLASS_MAPPINGS = {
    "PromptGenerator": PromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerator": "Prompt Generator"
}
