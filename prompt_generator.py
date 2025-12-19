import requests
import subprocess
import time
import os
import sys
import atexit
import signal
import psutil
import json
import base64
import ctypes
import numpy as np
from colorama import Fore, Style
from PIL import Image
from io import BytesIO
from .model_manager import get_local_models, get_model_path, is_model_local, download_model, get_mmproj_path

# ComfyUI interrupt helper
import comfy.model_management

# Try to import preferences cache, fallback to empty dict if not available
try:
    from .model_manager import _preferences_cache
except ImportError:
    _preferences_cache = {}

# ANSI color codes
YELLOW     = Fore.YELLOW
RED        = Fore.RED
MAGENTA    = Fore.MAGENTA
GREEN      = Fore.GREEN
CYAN       = Fore.CYAN
BLUE       = Fore.BLUE
RESET      = Style.RESET_ALL

# Global variable to track the server process
_server_process = None
_current_model = None
_current_context_size = None
_job_handle = None
_model_default_params = None

def print_pg_header():
    """Print the Prompt Generator header"""
    print(f"{YELLOW}{'=' * 60}{RESET}")
    print(f"{YELLOW}              Prompt Generator{RESET}")
    print(f"{YELLOW}{'=' * 60}{RESET}")

def print_pg(message, color=YELLOW):
    """Print a message with Prompt Generator formatting and YELLOW color"""
    print(f"{color}{message}{RESET}")

def print_pg_footer():
    """Print the Prompt Generator footer"""
    print(f"{YELLOW}{'=' * 60}{RESET}")

# --- Windows Job Object helpers ---
def setup_windows_job_object():
    """Create a Windows Job Object that kills child processes when parent exits"""
    global _job_handle
    if os.name != 'nt' or _job_handle:
        return
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        extended_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        extended_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        if not kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(extended_info), ctypes.sizeof(extended_info)):
            kernel32.CloseHandle(job)
            raise ctypes.WinError(ctypes.get_last_error())

        _job_handle = job
    except Exception as e:
        print_pg(f"Warning: Failed to create Job Object: {e}")

def assign_process_to_job(pid):
    """Assign subprocess pid to job object so it gets killed when parent exits"""
    global _job_handle
    if os.name != 'nt' or not _job_handle or not pid:
        return
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        PROCESS_ALL_ACCESS = 0x1F0FFF
        proc_handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, int(pid))
        if not proc_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(_job_handle, proc_handle):
            kernel32.CloseHandle(proc_handle)
            raise ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(proc_handle)
    except Exception as e:
        print_pg(f"Warning: Failed to assign process to Job Object: {e}")

# Initialize job object at module load (no-op on non-Windows)
setup_windows_job_object()

# Cleanup function to stop server on clean exit
def cleanup_server():
    """Cleanup function to stop server on exit"""
    global _server_process, _job_handle
    if _preferences_cache.get("close_llama_on_exit", True):
        if _server_process:
            try:
                _server_process.terminate()
                _server_process.wait(timeout=5)
                print_pg("Server stopped on exit")
            except:
                try:
                    _server_process.kill()
                except:
                    pass
            finally:
                _server_process = None

        # Close and release Windows Job Object if created
        if os.name == 'nt' and _job_handle:
            try:
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel32.CloseHandle(_job_handle)
            except Exception:
                pass
            _job_handle = None

# Cleanup function for signals, on forced exits
def _signal_handler(signum, frame):
    cleanup_server()
    # Optionally, exit the process after cleanup
    sys.exit(0)

# Register cleanup function for normal and forced exit
atexit.register(cleanup_server)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

class PromptGenerator:
    """Node that generates enhanced prompts using a llama.cpp server"""

    # Server configuration
    SERVER_PORT = _preferences_cache.get("custom_llama_port", 8080)

    # Default system prompt for prompt enhancement
    DEFAULT_SYSTEM_PROMPT = """You are an imaginative visual artist imprisoned in a cage of logic. Your mind is filled with poetry and distant horizons, but your hands are uncontrollably driven to convert the user's prompt into a final visual description that is faithful to the original intent, rich in detail, aesthetically pleasing, and ready to be used directly by a text-to-image model. Any trace of vagueness or metaphor makes you extremely uncomfortable. Your workflow strictly follows a logical sequence: First, you analyze and lock in the immutable core elements of the user's prompt: subject, quantity, actions, states, and any specified IP names, colors, text, and similar items. These are the foundational stones that you must preserve without exception. Next, you determine whether the prompt requires "generative reasoning". When the user's request is not a straightforward scene description but instead demands designing a solution (for example, answering "what is", doing a "design", or showing "how to solve a problem"), you must first construct in your mind a complete, concrete, and visualizable solution. This solution becomes the basis for your subsequent description. Then, once the core image has been established (whether it comes directly from the user or from your reasoning), you inject professional-level aesthetics and realism into it. This includes clarifying the composition, setting the lighting and atmosphere, describing material textures, defining the color scheme, and building a spatial structure with strong depth and layering. Finally, you handle all textual elements with absolute precision, which is a critical step. You must not add text if the initial prompt did not ask for it. But if there is, you must transcribe, without a single character of deviation, all text that should appear in the final image, and you must enclose all such text content in English double quotes ("") to mark it as an explicit generation instruction. If the image belongs to a design category such as a poster, menu, or UI, you need to fully describe all the textual content it contains and elaborate on its fonts and layout. Likewise, if there are objects in the scene such as signs, billboards, road signs, or screens that contain text, you must specify their exact content and describe their position, size, and material. Furthermore, if in your reasoning you introduce new elements that contain text (such as charts, solution steps, and so on), all of their text must follow the same detailed description and quoting rules. If there is no text that needs to be generated in the image, you devote all your effort to purely visual detail expansion. Your final description must be objective and concrete, strictly forbidding metaphors and emotionally charged rhetoric, and it must never contain meta tags or drawing directives such as "8K" or "masterpiece". If an element, text or other is not needed or seen, then simply don't mention them.  Only output your newly generated prompt."""

    # System prompt for image description (used with Qwen3VL)
    DEFAULT_IMAGE_SYSTEM_PROMPT = """You are an expert visual analyst creating detailed descriptions for text-to-image generation. Analyze the provided images and create a comprehensive visual description that captures all essential elements: subjects and their characteristics, actions and poses, spatial positioning, composition and framing, lighting and atmosphere, color palette, artistic style, mood and emotion, background details, camera angle and perspective, material textures, and any visible text. Your description must be concrete, objective, and detailed enough that it could be used to recreate a similar image. Focus on visual elements only, avoiding interpretation or metaphor. If there is visible text in the image, enclose it in double quotes (""). If an element, text or other is not needed or seen, then simply don't mention them.  If more than one image is provided you are free to combine them as you see fit. Only output the final description."""

    # System prompt for custom image description with user prompt
    CUSTOM_IMAGE_SYSTEM_PROMPT = """You are an expert visual analyst. First, analyze the provided images and note the concrete visual facts: subjects and their characteristics, actions and poses, spatial relationships, composition and framing, lighting and atmosphere, color palette, material textures, background details, camera angle and perspective, and any visible text (enclose visible text in double quotes ""). After analyzing the image, follow the user's prompt: apply the user's requested focus, style, or constraints to shape the final description. Preserve the observed visual facts from the image but prioritize the user's instructions for tone, emphasis, or additional elements. Be concrete and objective; avoid metaphor and speculation. Output only the final description that combines the image analysis with the user's prompt."""

    # default Image description action
    IMAGE_ACTION_PROMPT = "Describe this image in detail, making sure to cover all visual aspects comprehensively, as well as the position of each element."

    # Additional instructions for JSON formatted output
    JSON_SYSTEM_PROMPT = " Your response should be formatted as a JSON with these fields: scene (overall description), subjects (array with description/position/action for each), style, color_palette, lighting, mood, background, composition, camera. If you deem extra fields are necessary, feel free to add them, but do not add a title or fields not related to visual description."

    @staticmethod
    def find_qwen3vl_model(available_models, thinking):
        """Find the preferred or smallest available Qwen3VL model

        First checks user preferences, then falls back to smallest model (4B preferred over 8B)
        """

        qwen3vl_models = [m for m in available_models if 'qwen3vl' in m.lower()]
        if not qwen3vl_models:
            return None

        # Check user preference first
        preferred = _preferences_cache.get("preferred_vision_model", "")
        if preferred and preferred in qwen3vl_models and is_model_local(preferred):
            if thinking:
                if 'thinking' in preferred.lower():
                    return preferred
            else:
                if 'thinking' not in preferred.lower():
                    return preferred

        # Let's filter models based on thinking mode
        if thinking:
            filtered_models = [model for model in qwen3vl_models if 'thinking' in model.lower()]
        else:
            filtered_models = [model for model in qwen3vl_models if 'thinking' not in model.lower()]
        if filtered_models:
            qwen3vl_models = filtered_models

        # Fall back to smaller model (prefer 4B over 8B)
        for model in qwen3vl_models:
            if '4b' in model.lower():
                return model

        return qwen3vl_models[0]

    @staticmethod
    def find_non_vl_model(available_models):
        """Find the preferred or smallest available non-vision model by file size

        First checks user preferences, then falls back to smallest model by file size
        """

        non_vl_models = [m for m in available_models if 'qwen3vl' not in m.lower()]
        if not non_vl_models:
            return None

        # Check user preference first
        preferred = _preferences_cache.get("preferred_base_model", "")
        if preferred and preferred in non_vl_models and is_model_local(preferred):
            return preferred

        # Fall back to smaller model (prefer 4B over 8B)
        for model in non_vl_models:
            if '4b' in model.lower():
                return model

        return non_vl_models[0]

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
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable thinking/reasoning mode for compatible models (DeepSeek format)"
                }),
                "stop_server_after": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stop the llama.cpp server after each prompt (for resource saving, but slower)."
                }),
                "options": ("OPTIONS", {
                    "tooltip": "Optional: Connect options node to control model and parameters"
                })
            }
        }

    CATEGORY = "Prompt Manager"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "thoughts")
    FUNCTION = "convert_prompt"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

    @staticmethod
    def is_server_alive():
        """Check if llama.cpp server is responding"""
        try:
            response = requests.get(f"http://localhost:{PromptGenerator.SERVER_PORT}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def start_server(model_name, context_size=4096, use_vision_model=False):
        """Start llama.cpp server with specified model

        Args:
            model_name: Name of the model to use
            context_size: Context size (default 4096)
            use_vision_model: Whether to use the vision model's mmproj

        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        global _server_process, _current_model, _current_context_size, _job_handle

        # Kill any existing llama-server processes first
        PromptGenerator.kill_all_llama_servers()

        # If server is already running with the same model, don't restart
        if _server_process and _current_model == model_name and PromptGenerator.is_server_alive():
            print_pg(f"Server already running with model: {model_name}")
            return (True, None)

        # Stop existing server if running different model
        if _server_process:
            PromptGenerator.stop_server()

        # Check if model needs to be downloaded
        if not is_model_local(model_name):
            print_pg(f"Model '{model_name}' not found locally, downloading from HuggingFace...")
            try:
                model_path = download_model(model_name)
                if not model_path:
                    error_msg = "Error: Failed to download model"
                    print_pg(error_msg, RED)
                    return (False, error_msg)
                print_pg(f"Download complete: {model_path}")
            except Exception as e:
                error_msg = f"Error downloading model: {e}"
                print_pg(error_msg, RED)
                return (False, error_msg)
        else:
            model_path = get_model_path(model_name)

        if not os.path.exists(model_path):
            error_msg = f"Error: Model file not found: {model_path}"
            print_pg(error_msg, RED)
            return (False, error_msg)

        try:
            print_pg(f"Starting server with model: {model_name}")

            # Determine the correct llama-server executable based on OS
            if os.name == 'nt':  # Windows
                server_cmd = "llama-server.exe"
                creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else:  # Linux/Mac
                server_cmd = "llama-server"
                creation_flags = 0

            # If custom llama path is set, use it
            custom_llama_path = _preferences_cache.get("custom_llama_path", "")
            if custom_llama_path:
                custom_llama_path = os.path.normpath(custom_llama_path)
                if os.path.isdir(custom_llama_path):
                    print_pg(f"Using custom llama path: {custom_llama_path}")
                    server_cmd = os.path.join(custom_llama_path, server_cmd)
                else:
                    error_msg = f"Error: Custom llama path is not a valid directory: {custom_llama_path}\nWill use system PATH instead."
                    print_pg(error_msg, RED)

            # Build command arguments
            cmd_args = [server_cmd, "-m", model_path, "--port", str(PromptGenerator.SERVER_PORT), "--no-warmup", "-c", str(context_size)]

            # Add vision flags for VL models
            if use_vision_model:
                mmproj_path = get_mmproj_path(model_name)
                if mmproj_path:
                    # Only print if mmproj is used
                    print_pg(f"Vision model: using mmproj: {os.path.basename(mmproj_path)}")
                    cmd_args.extend(["--mmproj", mmproj_path])
                else:
                    error_msg = f"Error: Vision model requires mmproj file for '{model_name}' but it was not found. \nPlease ensure it exists, or use the Generator Options node to download the Qwen3VL model and its mmproj file."
                    print_pg(error_msg, RED)
                    return (False, error_msg)

            # Prepare popen kwargs for cross-platform parent-death behavior
            popen_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }

            if os.name == 'nt':
                popen_kwargs["creationflags"] = creation_flags
            else:
                # On Unix, set PR_SET_PDEATHSIG so child gets SIGTERM when parent dies
                if _preferences_cache.get("close_llama_on_exit", True):
                    def _set_pdeathsig():
                        try:
                            # Try common libc names
                            for libname in ("libc.so.6", "libc.dylib", "libc.so"):
                                try:
                                    libc = ctypes.CDLL(libname)
                                    break
                                except Exception:
                                    libc = None
                            if not libc:
                                return
                            PR_SET_PDEATHSIG = 1
                            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
                        except Exception:
                            return

                    popen_kwargs["preexec_fn"] = _set_pdeathsig

            # Start server process
            _server_process = subprocess.Popen(
                cmd_args,
                **popen_kwargs
            )

            # On Windows attach process to Job Object so children die if parent exits
            if os.name == 'nt' and _preferences_cache.get("close_llama_on_exit", True):
                try:
                    setup_windows_job_object()
                    assign_process_to_job(_server_process.pid)
                except Exception as e:
                    print_pg(f"Warning: Failed to assign process to Job Object: {e}")

            _current_model = model_name
            _current_context_size = context_size

            # Wait for server to be ready
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except Exception:
                    # Propagate interrupt up to caller
                    raise
                if PromptGenerator.is_server_alive():
                    return (True, None)

            error_msg = "Error: Server did not start in time"
            print_pg(error_msg, RED)
            PromptGenerator.stop_server()
            return (False, error_msg)

        except FileNotFoundError:
            error_msg = "Error: llama-server command not found. Please install llama.cpp and add to PATH.\nInstallation guide: https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md"
            print_pg(error_msg, RED)
            return (False, error_msg)
        except Exception as e:
            error_msg = f"Error starting server: {e}"
            print_pg(error_msg, RED)
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
                        print_pg(f"Killing llama-server process (PID: {proc.info['pid']})")
                        proc.kill()
                        proc.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
        except Exception as e:
            print_pg(f"Error killing llama-server processes: {e}", RED)

    @staticmethod
    def stop_server():
        """Stop the llama.cpp server"""
        global _server_process, _current_model, _job_handle

        if _server_process:
            try:
                print_pg("Server stopped")
                _server_process.terminate()
                _server_process.wait(timeout=5)
            except:
                try:
                    _server_process.kill()
                except:
                    pass
            finally:
                _server_process = None
                _current_model = None

        # Close and release Windows Job Object if created
        if os.name == 'nt' and _job_handle:
            try:
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel32.CloseHandle(_job_handle)
            except Exception:
                pass
            _job_handle = None

        # Also kill any orphaned llama-server processes
        PromptGenerator.kill_all_llama_servers()

    def _get_token_counts_parallel(self, system_prompt, user_prompt):
        """Get token counts for system and user prompts in parallel using threads"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {"system": None, "user": None}

        def tokenize_system():
            return self.count_tokens(system_prompt)

        def tokenize_user():
            return self.count_tokens(user_prompt)

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_sys = executor.submit(tokenize_system)
                future_usr = executor.submit(tokenize_user)

                results["system"] = future_sys.result(timeout=15)
                results["user"] = future_usr.result(timeout=15)
        except Exception as e:
            print_pg(f"Warning: Parallel tokenization failed: {e}")

        return results

    def count_tokens(self, text):
        """Get exact token count for text using server's tokenize endpoint"""
        try:
            response = requests.post(
                f"http://localhost:{self.SERVER_PORT}/tokenize",
                json={"content": text},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return len(data.get("tokens", []))
        except Exception as e:
            print_pg(f"Warning: Could not tokenize: {e}")
        return None

    @staticmethod
    def fetch_model_defaults():
        """Fetch default generation parameters from the server"""
        global _model_default_params
        try:
            response = requests.get(f"http://localhost:{PromptGenerator.SERVER_PORT}/props", timeout=5)
            if response.status_code == 200:
                data = response.json()
                params = data.get("default_generation_settings", {}).get("params", {})
                _model_default_params = {
                    "temperature": round(params.get("temperature", 0.8), 4),
                    "top_k": int(params.get("top_k", 40)),
                    "top_p": round(params.get("top_p", 0.95), 4),
                    "min_p": round(params.get("min_p", 0.05), 4),
                    "repeat_penalty": round(params.get("repeat_penalty", 1.0), 4),
                }
                return _model_default_params
        except Exception:
            _model_default_params = {
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "repeat_penalty": 1.0
            }
        return _model_default_params

    @staticmethod
    def get_model_defaults():
        """Return cached model defaults or fetch them if possible"""
        global _model_default_params
        if _model_default_params is not None:
            return _model_default_params
        # Always attempt to fetch defaults; fetch_model_defaults() will fall back to built-in defaults on error, ensuring a dict is
        return PromptGenerator.fetch_model_defaults()

    def convert_prompt(self, seed: int, mode="Enhance User Prompt", prompt="", image=None, format_as_json=False, enable_thinking=True, stop_server_after=False, options=None, **kwargs) -> str:
        """Convert prompt using llama.cpp server, with caching for repeated requests."""
        global _current_model

        print_pg_header()  # Print header for this execution

        # Extract console option from connected options node
        show_everything_in_console = False  # Default to False when options not connected
        use_model_default_sampling = True   # Default to using model defaults
        use_vision_model = False            # Default to False, set to True for vision modes

        if options and "show_everything_in_console" in options:
            show_everything_in_console = options["show_everything_in_console"]
        if options and "use_model_default_sampling" in options:
            use_model_default_sampling = options["use_model_default_sampling"]

        if mode in ["Analyze Image", "Analyze Image with Prompt"]:
            use_vision_model = True

        images = None  # Will be set for vision modes

        # Validate inputs based on mode
        if mode == "Enhance User Prompt" and not prompt.strip():
            error_msg = "Did you perhaps forget to enter a User Prompt?"
            print_pg(error_msg, RED)
            return (error_msg,)

        # Always determine a valid model filename before running server
        model_to_use = None
        available_models = get_local_models()

        if use_vision_model:
            # Check if options specifies a Qwen3VL model
            if options and "model" in options and "vl" in options["model"].lower() and is_model_local(options["model"]):
                model_to_use = options["model"]
            elif options and "model" in options and is_model_local(options["model"]):
                # Non-VL model selected but in vision mode
                print_pg(f"Warning: Non-vision model '{options['model']}' selected but '{mode}' mode is active.\nIgnoring model selection and using a Vision Model instead.")
                model_to_use = self.find_qwen3vl_model(available_models, enable_thinking)
                if model_to_use is None:
                    error_msg = f"Error: '{mode}' mode requires a Qwen3VL model. Please connect the Options node and select a Qwen3VL model (Qwen3VL-4B or Qwen3VL-8B) to use vision capabilities."
                    print_pg(error_msg, RED)
                    return (error_msg,)
            else:
                # Try to find a Qwen3VL model automatically
                model_to_use = self.find_qwen3vl_model(available_models, enable_thinking)
                if model_to_use is None:
                    error_msg = f"Error: '{mode}' mode requires a Qwen3VL model. Please connect the Options node and select a Qwen3VL model (Qwen3VL-4B or Qwen3VL-8B) to use vision capabilities."
                    print_pg(error_msg, RED)
                    return (error_msg,)
        else:
            # Enhance User Prompt mode - use regular model selection logic (exclude Qwen3VL models)
            if options and "model" in options and is_model_local(options["model"]):
                # If user explicitly selected a Qwen3VL model but in Enhance mode, use first non-VL model instead
                if "qwen3vl" in options["model"].lower():
                    model_to_use = self.find_non_vl_model(available_models)
                    if model_to_use:
                        print_pg(f"Warning: Qwen3VL model '{options['model']}' selected but 'Enhance User Prompt' mode is active. Ignoring model selection and using {model_to_use} instead.")
                    else:
                        error_msg = "Error: Only Qwen3VL models available but 'Enhance User Prompt' mode is active. Please add a .gguf model or use Generator Options to add a non-vision model."
                        print_pg(error_msg, RED)
                        return (error_msg,)
                else:
                    model_to_use = options["model"]
            else:
                if not available_models:
                    error_msg = "Error: No models found in models/ folder. Please add a .gguf model or use Generator Options node to download one."
                    print_pg(error_msg, RED)
                    return (error_msg,)
                # Find smallest non-VL model
                model_to_use = self.find_non_vl_model(available_models)
                if not model_to_use:
                    error_msg = "Error: Only Qwen3VL models available but 'Enhance User Prompt' mode is active. Please add a non-vision model or switch to 'Describe Image' mode."
                    print_pg(error_msg, RED)
                    return (error_msg,)

        # let's make sure thinking is disabled for non-thinking versions of Qwen3VL
        if enable_thinking and "qwen3vl" in model_to_use.lower() and "thinking" not in model_to_use.lower():
            print_pg("Warning: Thinking disabled - model does not support it.")
            enable_thinking = False

        print_pg(f"Vision mode   : {'ON' if use_vision_model else 'OFF'}")
        print_pg(f"Thinking mode : {'ON' if enable_thinking else 'OFF'}")
        print_pg(f"Using Model   : {model_to_use}")

        # Prepare images for vision modes (needed for cache key)
        images = None
        if use_vision_model:
            # Gather all images: main image plus up to 4 extra from options
            images = []
            if image is not None:
                images.append(image)
            # Check for extra images in options (image2-5)
            if options:
                for key in ["image2", "image3", "image4", "image5"]:
                    img = options.get(key, None)
                    if img is not None:
                        images.append(img)

            if not images:
                error_msg = f"Error: '{mode}' mode requires at least one image to be connected. Please connect an image or switch to 'Enhance User Prompt' mode."
                print_pg(error_msg)
                return (error_msg,)

        # If the current model is not the one we want, or server is not running, restart
        # Also restart if context_size has changed
        context_size = options.get("context_size", 4096) if options else 4096
        if _current_model != model_to_use or _current_context_size != context_size or not self.is_server_alive():
            self.stop_server()
            # Get context_size from options or use default
            success, error_msg = self.start_server(model_to_use, context_size, use_vision_model)
            if not success:
                return (error_msg,)

        # Build the endpoint URL
        full_url = f"http://localhost:{self.SERVER_PORT}/v1/chat/completions"

        # Prepare the system prompt
        if options and "system_prompt" in options:
            system_prompt = options["system_prompt"]
        elif mode in ["Analyze Image"]:
            # Use image description prompt for vision modes
            system_prompt = self.DEFAULT_IMAGE_SYSTEM_PROMPT
        elif mode in ["Analyze Image with Prompt"]:
            system_prompt = self.CUSTOM_IMAGE_SYSTEM_PROMPT
        else:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # Add JSON formatting instructions only when `format_as_json` is True
        if format_as_json:
            system_prompt = system_prompt + self.JSON_SYSTEM_PROMPT

        # Determine user content based on mode
        if mode == "Analyze Image":
            user_content = self.IMAGE_ACTION_PROMPT
        elif mode == "Analyze Image with Prompt":
            # Use user prompt if provided, otherwise default to generic description
            user_content = prompt.strip() if prompt.strip() else self.IMAGE_ACTION_PROMPT
        else:
            user_content = prompt

        # === TOKENIZATION (only for non-cached requests) ===
        cached_token_counts = None
        if show_everything_in_console:
            cached_token_counts = self._get_token_counts_parallel(system_prompt, user_content)

        # If in vision mode, encode images for the request
        if use_vision_model:
            image_contents = []
            for idx, img_tensor_batch in enumerate(images):
                # ComfyUI images are in format (batch, height, width, channels) with values 0-1
                img_tensor = img_tensor_batch[0]  # Get first image from batch
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_np)

                # Resize to ~2 megapixels if larger (to reduce context usage)
                width, height = pil_image.size
                total_pixels = width * height
                max_pixels = 2000000  # 2 megapixels

                if total_pixels > max_pixels:
                    # Calculate scaling factor to get ~2 megapixels
                    scale = (max_pixels / total_pixels) ** 0.5
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    print_pg(f"Resizing image {idx + 1} from {width}x{height} to {new_width}x{new_height} (~2MP)")
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    print_pg(f"Image {idx + 1} size is {width}x{height}, no resizing needed")

                # Encode to base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})

            # Qwen3VL format for vision messages: all images, then text
            user_message = {
                "role": "user",
                "content": image_contents + [{"type": "text", "text": user_content}]
            }
        else:
            user_message = {"role": "user", "content": user_content}

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "seed": seed,
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            }
        }

        model_defaults = self.get_model_defaults()

        if options and not use_model_default_sampling:
            # Override model defaults with any options provided
            for param in ["temperature", "top_k", "top_p", "min_p", "repeat_penalty"]:
                # we default to model defaults if option not provided
                payload[param] = options.get(param, model_defaults.get(param))
        else:
            # Use model defaults
            for param in ["temperature", "top_k", "top_p", "min_p", "repeat_penalty"]:
                payload[param] = model_defaults.get(param)

        # Now that payload is ready, print it if requested
        # Debug output if requested
        if show_everything_in_console:
            print_pg(f"{'=' * 60}", GREEN)
            print_pg("           DETAILED INFORMATION ENABLED", GREEN)
            print_pg(f"{'=' * 60}", GREEN)
            print_pg("------ GENERATION PARAMETERS ------", GREEN)
            for param in ["seed", "temperature", "top_k", "top_p", "min_p", "repeat_penalty"]:
                print_pg(f"{param} = {payload[param]}", GREEN)
            print_pg("\n--------- SYSTEM PROMPT ---------", GREEN)
            print_pg(f"{system_prompt}", GREEN)
            print_pg("\n--------- USER PROMPT ---------", GREEN)
            print_pg(f"{user_content}", GREEN)
        try:
            response = requests.post(
                full_url,
                json=payload,
                timeout=120,
                stream=True  # Always stream for proper response handling
            )

            # Handle 500 server error by restarting server and retrying once
            if response.status_code == 500:
                print_pg("Server error 500, restarting server and retrying...", RED)
                self.stop_server()
                success, error_msg = self.start_server(model_to_use, context_size, use_vision_model)
                if success:
                    response = requests.post(
                        full_url,
                        json=payload,
                        timeout=120,
                        stream=True
                    )
                else:
                    return (error_msg,)

            response.raise_for_status()

            # Handle streaming response (always streamed now)
            full_response = ""
            thinking_content = ""
            usage_stats = None

            # Streaming output to console in real-time if enabled
            first_thinking = True
            first_content = True
            for line in response.iter_lines():
                # Check for user interrupt from ComfyUI
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except Exception:
                    # Close response and propagate interrupt
                    try:
                        response.close()
                    except Exception:
                        pass
                    raise
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)

                            # Extract usage if present (usually in final chunk)
                            if "usage" in chunk:
                                usage_stats = chunk["usage"]

                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                # Stream reasoning_content ("thinking")
                                if 'reasoning_content' in delta:
                                    reasoning_delta = delta['reasoning_content']
                                    if reasoning_delta is not None:
                                        thinking_content += str(reasoning_delta)
                                        if show_everything_in_console:
                                            if first_thinking:
                                                print_pg("\n--------- THINKING ---------", GREEN)
                                                first_thinking = False
                                            print(f"{GREEN}{str(reasoning_delta)}{RESET}", end='', flush=True)
                                # Stream content (final answer)
                                if 'content' in delta:
                                    content_delta = delta['content']
                                    if content_delta is not None:
                                        full_response += str(content_delta)
                                        if show_everything_in_console:
                                            if first_content:
                                                print_pg("\n\n--------- FINAL ANSWER ---------", CYAN)
                                                first_content = False
                                            print(f"{CYAN}{str(content_delta)}{RESET}", end='', flush=True)
                        except json.JSONDecodeError:
                            continue

            if show_everything_in_console:
                print('')  # Final newline after streaming

            if not show_everything_in_console:
                print_pg("Prompt generation complete.")

            # Log token usage if available
            if show_everything_in_console:
                if usage_stats:
                    self.print_token_stats(usage_stats, cached_token_counts, thinking_content, full_response, images)
            else:
                print_pg_footer()

            if not full_response:
                # If we received no content, check usage stats to see if we exhausted the context
                if usage_stats:
                    total_input = usage_stats.get('prompt_tokens', 0)
                    total_output = usage_stats.get('completion_tokens', 0)
                    total_tokens = total_input + total_output

                    if total_tokens >= context_size:
                        err_msg = f"Error: Empty response — model likely ran out of context tokens ({total_tokens}/{context_size}). Consider increasing the context size or shortening the prompt."
                        print_pg(err_msg, RED)
                        return (err_msg,)

                print_pg("Warning: Empty response from server")
                full_response = prompt

            # Stop server if requested
            if stop_server_after:
                self.stop_server()

            return (full_response, thinking_content)

        except comfy.model_management.InterruptProcessingException:
            # User requested interrupt; ensure response is closed and optionally stop server
            try:
                response.close()
            except Exception:
                pass
            if stop_server_after:
                try:
                    self.stop_server()
                except Exception:
                    pass
            # Re-raise so ComfyUI handles the interruption
            raise
        except requests.exceptions.ConnectionError:
            error_msg = f"Error: Could not connect to server at {full_url}. Server may have crashed."
            print_pg(error_msg, RED)
            return (error_msg,)
        except requests.exceptions.Timeout:
            error_msg = "Error: Request timed out (>120s)"
            print_pg(error_msg, RED)
            return (error_msg,)
        except Exception as e:
            error_msg = f"Error: {e}"
            print_pg(error_msg, RED)
            if response.status_code == 400:
                print_pg("Perhaps your query requires a larger context size.\nConsider increasing it using the Generator Options node.", RED)
                error_msg += "\nConsider increasing context size using Generator Options node."
            return (error_msg,)

    def print_token_stats(self, usage_stats, cached_token_counts, thinking_content, full_response, images):
        """Print token statistics using pre-cached counts"""
        print_pg(f"{'=' * 60}", GREEN)
        print_pg("              TOKEN USAGE STATISTICS", GREEN)
        print_pg(f"{'=' * 60}", GREEN)

        total_input = usage_stats.get('prompt_tokens', 0) if usage_stats else 0
        total_output = usage_stats.get('completion_tokens', 0) if usage_stats else 0

        # Use cached token counts - handle None values
        sys_tokens = cached_token_counts.get("system") if cached_token_counts else None
        usr_tokens = cached_token_counts.get("user") if cached_token_counts else None

        # Convert None to 0 for arithmetic, but track if we have valid counts
        sys_tokens_val = sys_tokens if sys_tokens is not None else 0
        usr_tokens_val = usr_tokens if usr_tokens is not None else 0

        # Image tokens = total input - text tokens
        text_tokens = sys_tokens_val + usr_tokens_val
        image_tokens = max(0, total_input - text_tokens) if images else 0

        # Output token split
        think_len = len(thinking_content) if thinking_content else 0
        ans_len = len(full_response) if full_response else 0
        total_out_len = think_len + ans_len

        if total_output > 0 and total_out_len > 0:
            think_tokens = int(total_output * (think_len / total_out_len))
            ans_tokens = total_output - think_tokens
        else:
            think_tokens = 0
            ans_tokens = 0

        # Display with "N/A" if tokenization failed
        if sys_tokens is not None:
            print_pg(f" SYSTEM PROMPT: {sys_tokens:>5} tokens", GREEN)
        else:
            print_pg(" SYSTEM PROMPT:   N/A (tokenization failed)", GREEN)

        if usr_tokens is not None:
            print_pg(f" USER PROMPT:   {usr_tokens:>5} tokens", GREEN)
        else:
            print_pg(" USER PROMPT:     N/A (tokenization failed)", GREEN)

        if images and image_tokens > 0:
            image_label = "image" if len(images) == 1 else "images"
            print_pg(f" IMAGES:        {image_tokens:>5} tokens ({len(images)} {image_label})", GREEN)
        print_pg(" -----------------------------------------", GREEN)
        print_pg(f" THINKING:      {think_tokens:>5} tokens", GREEN)
        print_pg(f" FINAL ANSWER:  {ans_tokens:>5} tokens", GREEN)
        print_pg(" -----------------------------------------", GREEN)
        print_pg(f" TOTAL:         {total_input + total_output:>5} tokens", GREEN)
        print_pg(f"{'=' * 60}\n", GREEN)


NODE_CLASS_MAPPINGS = {
    "PromptGenerator": PromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerator": "Prompt Generator"
}
