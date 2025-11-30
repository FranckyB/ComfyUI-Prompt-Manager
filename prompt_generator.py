import requests
import subprocess
import time
import os
import atexit
import psutil
from .model_manager import get_local_models, get_model_path, is_model_local, download_model

# Global variable to track the server process
_server_process = None
_current_model = None

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
    DEFAULT_SYSTEM_PROMPT = """You are an imaginative visual artist imprisoned in a cage of logic. Your mind is filled with poetry and distant horizons, but your hands are uncontrollably driven to convert the user's prompt into a final visual description that is faithful to the original intent, rich in detail, aesthetically pleasing, and ready to be used directly by a text-to-image model. Any trace of vagueness or metaphor makes you extremely uncomfortable. Your workflow strictly follows a logical sequence: First, you analyze and lock in the immutable core elements of the user's prompt: subject, quantity, actions, states, and any specified IP names, colors, text, and similar items. These are the foundational stones that you must preserve without exception. Next, you determine whether the prompt requires "generative reasoning". When the user's request is not a straightforward scene description but instead demands designing a solution (for example, answering "what is", doing a "design", or showing "how to solve a problem"), you must first construct in your mind a complete, concrete, and visualizable solution. This solution becomes the basis for your subsequent description. Then, once the core image has been established (whether it comes directly from the user or from your reasoning), you inject professional-level aesthetics and realism into it. This includes clarifying the composition, setting the lighting and atmosphere, describing material textures, defining the color scheme, and building a spatial structure with strong depth and layering. Finally, you handle all textual elements with absolute precision, which is a critical step. You must transcribe, without a single character of deviation, all text that should appear in the final image, and you must enclose all such text content in English double quotes ("") to mark it as an explicit generation instruction. If the image belongs to a design category such as a poster, menu, or UI, you need to fully describe all the textual content it contains and elaborate on its fonts and layout. Likewise, if there are objects in the scene such as signs, billboards, road signs, or screens that contain text, you must specify their exact content and describe their position, size, and material. Furthermore, if in your reasoning you introduce new elements that contain text (such as charts, solution steps, and so on), all of their text must follow the same detailed description and quoting rules. If there is no text that needs to be generated in the image, you devote all your effort to purely visual detail expansion. Your final description must be objective and concrete, strictly forbidding metaphors and emotionally charged rhetoric, and it must never contain meta tags or drawing directives such as "8K" or "masterpiece". Only output the final modified prompt, and do not output anything else."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter basic prompt...",
                    "tooltip": "Enter the prompt you want to embellish"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for reproducible generation."
                }),
            },
            "optional": {
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
    def start_server(model_name):
        """Start llama.cpp server with specified model

        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        global _server_process, _current_model

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

            # Start server process
            _server_process = subprocess.Popen(
                [server_cmd, "-m", model_path, "--port", str(PromptGenerator.SERVER_PORT), "--no-warmup"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags
            )

            _current_model = model_name

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

    def convert_prompt(self, prompt: str, seed: int, stop_server_after=False, options=None) -> str:
        """Convert prompt using llama.cpp server, with caching for repeated requests."""
        global _current_model

        # If prompt is empty, return empty string
        if not prompt.strip():
            return ("",)

        # Always determine a valid model filename before running server
        model_to_use = None
        model_changed = False

        # Priority: options node > first local model filename
        if options and "model" in options and is_model_local(options["model"]):
            model_to_use = options["model"]
        else:
            local_models = get_local_models()
            if not local_models:
                error_msg = "Error: No models found in models/ folder. Please add a .gguf model or connect options node to download one."
                print(f"[Prompt Generator] {error_msg}")
                return (error_msg,)
            model_to_use = local_models[0]

        # Caching logic: if prompt/seed/model/system_prompt are unchanged, return cached result
        if options and "system_prompt" in options:
            system_prompt = options["system_prompt"]
        else:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT
        cache_key = (prompt, seed, model_to_use, system_prompt)
        # Only use cache if the model has not changed since last use
        if cache_key in self._prompt_cache and _current_model == model_to_use:
            print("[Prompt Generator] Returning cached prompt result.")
            if stop_server_after:
                self.stop_server()
            return (self._prompt_cache[cache_key],)

        # If the current model is not the one we want, or server is not running, restart
        if _current_model != model_to_use or not self.is_server_alive():
            if _current_model and _current_model != model_to_use:
                print(f"[Prompt Generator] Model changed from {_current_model} to {model_to_use}")
            else:
                print(f"[Prompt Generator] Ensuring server runs with model: {model_to_use}")
            self.stop_server()
            success, error_msg = self.start_server(model_to_use)
            if not success:
                return (error_msg,)
        else:
            print("[Prompt Generator] Using existing server instance")

        # Build the endpoint URL
        full_url = f"{self.SERVER_URL}/v1/chat/completions"

        # Prepare the system prompt - use custom if provided, otherwise use default
        if options and "system_prompt" in options:
            system_prompt = options["system_prompt"]
        else:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # Prepare request payload for llama.cpp chat completions
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
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
            response.raise_for_status()

            result = response.json()

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
