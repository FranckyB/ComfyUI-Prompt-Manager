import requests
import subprocess
import time
import os
import atexit
import psutil
import json
import signal
import sys
import re
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from .model_manager import get_local_models, get_model_path, is_model_local, download_model

# Import ComfyUI's model management for interrupt handling
import comfy.model_management

# Global variable to track the server process
_server_process = None
_current_model = None
_current_gpu_config = None  # Track GPU configuration
_current_context_size = None  # Track context size
_current_mmproj = None  # Track mmproj file
_model_default_params = None  # Cache for model default parameters
_model_layer_cache = {}  # Cache for model layer counts

# Windows Job Object for guaranteed child process cleanup
_job_handle = None

def setup_windows_job_object():
    """Create a Windows Job Object that kills child processes when parent exits"""
    global _job_handle
    
    if os.name != 'nt':
        return
    
    try:
        import ctypes
        from ctypes import wintypes
        
        kernel32 = ctypes.windll.kernel32
        
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        
        _job_handle = kernel32.CreateJobObjectW(None, None)
        if not _job_handle:
            return
        
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
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

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        kernel32.SetInformationJobObject(
            _job_handle,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info)
        )
        print("[Prompt Generator] Job Object created - llama-server will be killed on console close")
            
    except Exception as e:
        print(f"[Prompt Generator] Warning: Job object setup failed: {e}")


def assign_process_to_job(process):
    """Assign subprocess to job object so it gets killed when parent exits"""
    global _job_handle
    
    if os.name != 'nt' or not _job_handle or not process:
        return
    
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = int(process._handle)
        kernel32.AssignProcessToJobObject(_job_handle, handle)
    except:
        pass


# Initialize job object at module load
setup_windows_job_object()


def setup_console_handler():
    """Set up Windows console control handler for console close, logoff, shutdown only"""
    if os.name != 'nt':
        return
    
    try:
        import ctypes
        
        CTRL_CLOSE_EVENT = 2
        CTRL_LOGOFF_EVENT = 5
        CTRL_SHUTDOWN_EVENT = 6
        
        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
        def console_handler(event):
            if event in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
                print(f"\n[Prompt Generator] Console closing (event {event}), cleaning up...")
                cleanup_server()
                return False
            return False
        
        global _console_handler_ref
        _console_handler_ref = console_handler
        
        kernel32 = ctypes.windll.kernel32
        if kernel32.SetConsoleCtrlHandler(console_handler, True):
            print("[Prompt Generator] Console close handler registered")
        else:
            print("[Prompt Generator] Warning: Could not register console handler")
            
    except Exception as e:
        print(f"[Prompt Generator] Warning: Console handler setup failed: {e}")


def cleanup_server():
    """Cleanup function to stop server on exit"""
    global _server_process, _current_model, _current_gpu_config, _current_context_size, _current_mmproj, _model_default_params
    
    if _server_process:
        try:
            print("[Prompt Generator] Stopping server on exit...")
            _server_process.terminate()
            try:
                _server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _server_process.kill()
                _server_process.wait(timeout=2)
            print("[Prompt Generator] Server stopped on exit")
        except Exception as e:
            print(f"[Prompt Generator] Error stopping server: {e}")
        finally:
            _server_process = None
            _current_model = None
            _current_gpu_config = None
            _current_context_size = None
            _current_mmproj = None
            _model_default_params = None
    
    # Also kill any orphaned llama-server processes
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and 'llama-server' in proc.info['name'].lower():
                    print(f"[Prompt Generator] Killing orphaned llama-server (PID: {proc.info['pid']})")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"[Prompt Generator] Error cleaning up orphaned processes: {e}")


def signal_handler(signum, frame):
    """Handle termination signals"""
    print(f"\n[Prompt Generator] Signal {signum} received, cleaning up...")
    cleanup_server()
    sys.exit(0)


# === INITIALIZATION ===
setup_windows_job_object()
setup_console_handler()

try:
    signal.signal(signal.SIGTERM, signal_handler)
except Exception as e:
    print(f"[Prompt Generator] Warning: Could not register signal handlers: {e}")

atexit.register(cleanup_server)


def get_model_layer_count(model_path):
    """Get the number of layers in a GGUF model by running llama-server briefly"""
    global _model_layer_cache
    
    # Check cache first
    if model_path in _model_layer_cache:
        return _model_layer_cache[model_path]
    
    try:
        if os.name == 'nt':
            server_cmd = "llama-server.exe"
        else:
            server_cmd = "llama-server"
        
        # Run with minimal settings just to get model info
        cmd = [server_cmd, "-m", model_path, "-ngl", "0", "-c", "512"]
        
        print(f"[Prompt Generator] Detecting layer count for model...")
        
        if os.name == 'nt':
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
        
        layer_count = None
        start_time = time.time()
        
        # Read output line by line looking for layer info
        while time.time() - start_time < 30:  # 30 second timeout
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            
            decoded = line.decode('utf-8', errors='ignore')
            
            # Look for "n_layer = XX" pattern
            match = re.search(r'n_layer\s*=\s*(\d+)', decoded)
            if match:
                layer_count = int(match.group(1))
                print(f"[Prompt Generator] Detected {layer_count} layers")
                break
        
        # Kill the process
        try:
            process.terminate()
            process.wait(timeout=3)
        except:
            try:
                process.kill()
            except:
                pass
        
        if layer_count:
            _model_layer_cache[model_path] = layer_count
            return layer_count
        else:
            print("[Prompt Generator] Warning: Could not detect layer count, using default 999")
            return None
            
    except Exception as e:
        print(f"[Prompt Generator] Error detecting layers: {e}")
        return None


def parse_gpu_config(gpu_config_str, total_layers):
    """
    Parse GPU configuration string and return layer distribution.
    
    Args:
        gpu_config_str: String like "gpu0:0.7" or "gpu0:0.5,gpu1:0.4" or "auto"
        total_layers: Total number of layers in the model
    
    Returns:
        List of tuples: [(device_index, layer_count), ...] or None for auto
    """
    if not gpu_config_str or gpu_config_str.lower() in ('auto', 'all', ''):
        return None  # Use default -ngl 999
    
    gpu_config_str = gpu_config_str.lower().strip()
    
    # Parse each GPU specification
    gpu_specs = []
    total_fraction = 0.0
    
    for part in gpu_config_str.split(','):
        part = part.strip()
        if not part:
            continue
        
        # Match patterns like "gpu0:0.7" or "vulkan0:0.5"
        match = re.match(r'(?:gpu|vulkan)?(\d+)\s*:\s*([\d.]+)', part)
        if match:
            device_idx = int(match.group(1))
            fraction = float(match.group(2))
            
            if fraction > 1.0:
                # Assume it's a layer count, not a fraction
                layer_count = round(fraction)
            else:
                layer_count = round(total_layers * fraction)
            
            gpu_specs.append((device_idx, layer_count))
            total_fraction += fraction if fraction <= 1.0 else (fraction / total_layers)
        else:
            print(f"[Prompt Generator] Warning: Could not parse GPU spec '{part}'")
    
    if not gpu_specs:
        return None
    
    # Calculate remaining layers for CPU
    assigned_layers = sum(layers for _, layers in gpu_specs)
    cpu_layers = max(0, total_layers - assigned_layers)
    
    print(f"[Prompt Generator] Layer distribution for {total_layers} layers:")
    for device_idx, layers in gpu_specs:
        print(f"  GPU{device_idx}: {layers} layers ({layers/total_layers*100:.1f}%)")
    print(f"  CPU: {cpu_layers} layers ({cpu_layers/total_layers*100:.1f}%)")
    
    return gpu_specs


def build_gpu_args(gpu_specs, total_layers):
    """
    Build command line arguments for GPU layer distribution.
    """
    if gpu_specs is None:
        # Auto mode: offload all to GPU 0
        return ["-ngl", "999", "--main-gpu", "0"]
    
    if len(gpu_specs) == 1:
        # Single GPU with specific layer count
        device_idx, layer_count = gpu_specs[0]
        return ["-ngl", str(layer_count), "--main-gpu", str(device_idx)]
    
    else:
        # Multi-GPU
        max_device = max(device_idx for device_idx, _ in gpu_specs)
        split_values = [0] * (max_device + 1)
        
        for device_idx, layer_count in gpu_specs:
            split_values[device_idx] = layer_count
        
        total_gpu_layers = sum(layers for _, layers in gpu_specs)
        ngl_value = "999" if total_gpu_layers >= total_layers else str(total_gpu_layers)
        
        ts_str = ",".join(str(v) for v in split_values)
        
        return ["-ngl", ngl_value, "--tensor-split", ts_str]
    
def tensor_to_base64(image_tensor):
    """
    Convert a ComfyUI image tensor to base64-encoded PNG string.
    
    ComfyUI images are tensors with shape [B, H, W, C] in float32 format (0-1 range).
    """
    try:
        # Handle batch dimension - take first image if batched
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        # Convert from float32 (0-1) to uint8 (0-255)
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Create PIL Image
        pil_image = Image.fromarray(image_np)
        
        # Convert to base64 PNG
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        
        return base64_str
    except Exception as e:
        print(f"[Prompt Generator] Error converting image to base64: {e}")
        return None


class PromptGenerator:
    """Node that generates enhanced prompts using a llama.cpp server"""

    _prompt_cache = {}
    SERVER_URL = "http://localhost:8080"
    SERVER_PORT = 8080

    DEFAULT_SYSTEM_PROMPT = """You are an imaginative visual artist imprisoned in a cage of logic. Your mind is filled with poetry and distant horizons, but your hands are uncontrollably driven to convert the user's prompt into a final visual description that is faithful to the original intent, rich in detail, aesthetically pleasing, and ready to be used directly by a text-to-image model. Any trace of vagueness or metaphor makes you extremely uncomfortable. Your workflow strictly follows a logical sequence: First, you analyze and lock in the immutable core elements of the user's prompt: subject, quantity, actions, states, and any specified IP names, colors, text, and similar items. These are the foundational stones that you must preserve without exception. Next, you determine whether the prompt requires "generative reasoning". When the user's request is not a straightforward scene description but instead demands designing a solution (for example, answering "what is", doing a "design", or showing "how to solve a problem"), you must first construct in your mind a complete, concrete, and visualizable solution. This solution becomes the basis for your subsequent description. Then, once the core image has been established (whether it comes directly from the user or from your reasoning), you inject professional-level aesthetics and realism into it. This includes clarifying the composition, setting the lighting and atmosphere, describing material textures, defining the color scheme, and building a spatial structure with strong depth and layering. Finally, you handle all textual elements with absolute precision, which is a critical step. You must not add text if the initial prompt did not ask for it. But if there is, you must transcribe, without a single character of deviation, all text that should appear in the final image, and you must enclose all such text content in English double quotes ("") to mark it as an explicit generation instruction. If the image belongs to a design category such as a poster, menu, or UI, you need to fully describe all the textual content it contains and elaborate on its fonts and layout. Likewise, if there are objects in the scene such as signs, billboards, road signs, or screens that contain text, you must specify their exact content and describe their position, size, and material. Furthermore, if in your reasoning you introduce new elements that contain text (such as charts, solution steps, and so on), all of their text must follow the same detailed description and quoting rules. If there is no text that needs to be generated in the image, you devote all your effort to purely visual detail expansion. Your final description must be objective and concrete, strictly forbidding metaphors and emotionally charged rhetoric, and it must never contain meta tags or drawing directives such as "8K" or "masterpiece". Only output the final modified prompt, and do not output anything else. If no text is needed, don't mention it."""

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
                "show_everything_in_console": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print system prompt, user prompt, thinking process, and raw model response to the console."
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
    def fetch_model_defaults():
        """Fetch default generation parameters from the server"""
        global _model_default_params
        
        try:
            response = requests.get(f"{PromptGenerator.SERVER_URL}/props", timeout=5)
            if response.status_code == 200:
                data = response.json()
                params = data.get("default_generation_settings", {}).get("params", {})
                
                _model_default_params = {
                    "temperature": round(params.get("temperature", 0.8), 4),
                    "top_k": int(params.get("top_k", 40)),
                    "top_p": round(params.get("top_p", 0.95), 4),
                    "min_p": round(params.get("min_p", 0.05), 4),
                    "repeat_penalty": round(params.get("repeat_penalty", 1.0), 4),
                    "max_tokens": int(params.get("max_tokens", -1)) if params.get("max_tokens", -1) > 0 else 2048,
                }
                
                print(f"[Prompt Generator] Fetched model defaults: {_model_default_params}")
                return _model_default_params
        except Exception as e:
            print(f"[Prompt Generator] Could not fetch model defaults: {e}")
        
        return None

    @staticmethod
    def get_model_defaults():
        """Get cached model defaults or fetch them"""
        global _model_default_params
        
        if _model_default_params is not None:
            return _model_default_params
        
        if PromptGenerator.is_server_alive():
            return PromptGenerator.fetch_model_defaults()
        
        return None

    @staticmethod
    def start_server(model_name, gpu_config=None, context_size=32768, mmproj=None):
        """Start llama.cpp server with specified model, GPU configuration, context size, and optional mmproj"""
        global _server_process, _current_model, _current_gpu_config, _current_context_size, _model_default_params, _current_mmproj

        # Kill any existing llama-server processes first
        PromptGenerator.kill_all_llama_servers()

        # If server is already running with the same model, GPU config, context size, and mmproj, don't restart
        if (_server_process and 
            _current_model == model_name and 
            _current_gpu_config == gpu_config and
            _current_context_size == context_size and
            _current_mmproj == mmproj and
            PromptGenerator.is_server_alive()):
            print(f"[Prompt Generator] Server already running with model: {model_name}")
            return (True, None)

        # Stop existing server if running different model, GPU config, context size, or mmproj
        if _server_process:
            PromptGenerator.stop_server()

        # Reset model defaults when changing models
        _model_default_params = None

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

        # Check mmproj if specified
        mmproj_path = None
        if mmproj:
            from .model_manager import get_mmproj_path
            mmproj_path = get_mmproj_path(mmproj)
            if not os.path.exists(mmproj_path):
                error_msg = f"Error: mmproj file not found: {mmproj_path}"
                print(f"[Prompt Generator] {error_msg}")
                return (False, error_msg)
            print(f"[Prompt Generator] Using mmproj: {mmproj}")

        try:
            print(f"[Prompt Generator] Starting llama.cpp server with model: {model_name}")
            print(f"[Prompt Generator] Context size: {context_size}")

            if os.name == 'nt':
                server_cmd = "llama-server.exe"
            else:
                server_cmd = "llama-server"

            # Build base command with context size
            cmd = [
                server_cmd, 
                "-m", model_path, 
                "--port", str(PromptGenerator.SERVER_PORT),
                "--no-warmup",
                "--reasoning-format", "deepseek",
                "-c", str(context_size)
            ]
            
            # Add mmproj if specified
            if mmproj_path:
                cmd.extend(["--mmproj", mmproj_path])
            
            # Handle GPU configuration
            if gpu_config and gpu_config.lower() not in ('auto', 'all', ''):
                # Get layer count for this model
                total_layers = get_model_layer_count(model_path)
                
                if total_layers:
                    gpu_specs = parse_gpu_config(gpu_config, total_layers)
                    gpu_args = build_gpu_args(gpu_specs, total_layers)
                else:
                    # Fallback to auto if we couldn't detect layers
                    gpu_args = ["-ngl", "999", "--split-mode", "none", "--main-gpu", "0"]
            else:
                # Auto mode - use only GPU 0
                gpu_args = ["-ngl", "999", "--split-mode", "none", "--main-gpu", "0"]
            
            cmd.extend(gpu_args)
            
            print(f"[Prompt Generator] Command: {' '.join(cmd)}")
            print("[Prompt Generator] Thinking mode: controlled per-request via chat_template_kwargs")

            if os.name == 'nt':
                _server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                _server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )

            assign_process_to_job(_server_process)
            _current_model = model_name
            _current_gpu_config = gpu_config
            _current_context_size = context_size
            _current_mmproj = mmproj

            print("[Prompt Generator] Waiting for server to be ready...")
            
            for i in range(60):
                time.sleep(1)
                
                if PromptGenerator.is_server_alive():
                    print("[Prompt Generator] Server is ready!")
                    # Fetch model defaults after server starts
                    PromptGenerator.fetch_model_defaults()
                    return (True, None)
                
                if _server_process.poll() is not None:
                    try:
                        output = _server_process.stdout.read().decode('utf-8', errors='ignore')
                    except:
                        output = ""
                    
                    error_msg = f"Error: Server crashed during startup. Exit code: {_server_process.returncode}"
                    print(f"[Prompt Generator] {error_msg}")
                    if output:
                        print(f"[Prompt Generator] Server output:\n{output}")
                    
                    _server_process = None
                    _current_model = None
                    _current_gpu_config = None
                    _current_context_size = None
                    _current_mmproj = None
                    return (False, error_msg + (f"\n\nServer output:\n{output[:1000]}" if output else ""))
                
                if (i + 1) % 10 == 0:
                    print(f"[Prompt Generator] Still waiting... ({i + 1}s)")

            error_msg = "Error: Server did not start in time (60s timeout)"
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
            import traceback
            traceback.print_exc()
            return (False, error_msg)

    @staticmethod
    def kill_all_llama_servers():
        """Kill all llama-server processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
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
        global _server_process, _current_model, _current_gpu_config, _current_context_size, _current_mmproj, _model_default_params

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
                _current_gpu_config = None
                _current_context_size = None
                _current_mmproj = None
                _model_default_params = None

        PromptGenerator.kill_all_llama_servers()

    def _print_debug_header(self, payload, enable_thinking, use_model_default_sampling):
        """Helper to print debug info header"""
        print("\n" + "="*60)
        print(" [Prompt Generator] DEBUG: TOKENS & MESSAGES IN ACTION")
        print("="*60)
        
        if enable_thinking:
            print(f"\n🧠 THINKING MODE: ON (model will reason before answering)")
        else:
            print(f"\n🧠 THINKING MODE: OFF (direct answer, no reasoning)")
        
        if use_model_default_sampling:
            print(f"⚙️  PARAMETERS: Using model defaults")
        else:
            print(f"⚙️  PARAMETERS: Using custom/node settings")
        
        if "messages" in payload:
            for msg in payload["messages"]:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                
                print(f"\n--- <|{role}|> STARTS ---")
                
                # Handle multi-part content (for VLM with images)
                if isinstance(content, list):
                    for idx, part in enumerate(content):
                        if isinstance(part, dict):
                            if part.get("type") == "image_url":
                                print(f"[Image {idx + 1}: base64 data omitted for brevity]")
                            elif part.get("type") == "text":
                                print(part.get("text", ""))
                            else:
                                print(f"[Unknown content type: {part.get('type')}]")
                        else:
                            print(part)
                else:
                    # Simple text content
                    print(content)
                
                print(f"--- <|{role}|> ENDS ---")
        
        print("\n--- GENERATION PARAMS ---")
        params = {k: v for k, v in payload.items() if k != "messages"}
        print(json.dumps(params, indent=2))

    def convert_prompt(self, prompt: str, seed: int, stop_server_after=False, 
                    show_everything_in_console=False, options=None) -> str:
        """Convert prompt using llama.cpp server, with caching for repeated requests."""
        global _current_model, _current_gpu_config, _current_context_size, _current_mmproj

        if not prompt.strip():
            return ("",)

        # === EXTRACT OPTIONS ===
        model_to_use = None
        enable_thinking = True  # Default to thinking ON
        use_model_default_sampling = False  # Default to custom parameters
        gpu_config = None  # GPU layer distribution config
        context_size = 32768  # Default context size
        images = None  # Optional images for VLM
        mmproj = None  # Multimodal projector for VLM
        
        if options:
            if "model" in options and is_model_local(options["model"]):
                model_to_use = options["model"]
            if "enable_thinking" in options:
                enable_thinking = options["enable_thinking"]
            if "use_model_default_sampling" in options:
                use_model_default_sampling = options["use_model_default_sampling"]
            if "gpu_config" in options:
                gpu_config = options["gpu_config"]
            if "context_size" in options:
                context_size = int(options["context_size"])
            if "images" in options:
                images = options["images"]
            if "mmproj" in options:
                mmproj = options["mmproj"]
        
        if not model_to_use:
            local_models = get_local_models()
            if not local_models:
                error_msg = "Error: No models found in models/ folder. Please add a .gguf model or connect options node to download one."
                print(f"[Prompt Generator] {error_msg}")
                return (error_msg,)
            model_to_use = local_models[0]

        if options and "system_prompt" in options:
            system_prompt = options["system_prompt"]
        else:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # Cache key includes thinking setting, use_model_default_sampling, and image count
        # Note: We include image count but not image content in cache key for practical reasons
        image_count = len(images) if images else 0
        options_tuple = tuple(sorted(options.items())) if options else ()
        cache_key = (prompt, seed, model_to_use, options_tuple, image_count)

        # Check for images without mmproj
        if images and not mmproj:
            error_msg = f"Error: Images provided but no matching mmproj file found for model '{model_to_use}'. Please ensure a corresponding mmproj file (e.g., '{model_to_use.replace('.gguf', '')}-mmproj-*.gguf') exists in the models folder."
            print(f"[Prompt Generator] {error_msg}")
            return (error_msg,)

        # Only restart server if model, GPU config, context size, or mmproj changed
        if (_current_model != model_to_use or 
            _current_gpu_config != gpu_config or 
            _current_context_size != context_size or
            _current_mmproj != mmproj or
            not self.is_server_alive()):
            if _current_model and _current_model != model_to_use:
                print(f"[Prompt Generator] Model changed: {_current_model} → {model_to_use}")
            elif _current_gpu_config != gpu_config:
                print(f"[Prompt Generator] GPU config changed: {_current_gpu_config} → {gpu_config}")
            elif _current_context_size != context_size:
                print(f"[Prompt Generator] Context size changed: {_current_context_size} → {context_size}")
            elif _current_mmproj != mmproj:
                print(f"[Prompt Generator] mmproj changed: {_current_mmproj} → {mmproj}")
            else:
                print(f"[Prompt Generator] Starting server with model: {model_to_use}")
            self.stop_server()
            success, error_msg = self.start_server(model_to_use, gpu_config, context_size, mmproj)
            if not success:
                return (error_msg,)
        else:
            print("[Prompt Generator] Using existing server instance")
            
        # Log thinking mode (no restart needed!)
        if enable_thinking:
            print("[Prompt Generator] Thinking: ON (per-request)")
        else:
            print("[Prompt Generator] Thinking: OFF (per-request)")
        
        # Log image count
        if images:
            print(f"[Prompt Generator] Images attached: {len(images)}")

        # Build user message content
        # For VLM models with images, we use the multi-part content format
        if images:
            user_content = []
            
            # Add images first
            for idx, img_tensor in enumerate(images):
                base64_img = tensor_to_base64(img_tensor)
                if base64_img:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
                        }
                    })
                    print(f"[Prompt Generator] Image {idx + 1} encoded successfully")
                else:
                    print(f"[Prompt Generator] Warning: Failed to encode image {idx + 1}")
            
            # Add text prompt
            user_content.append({
                "type": "text",
                "text": prompt
            })
        else:
            # Simple text-only content
            user_content = prompt

        # Build payload with base settings
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "seed": seed,
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            }
        }

        # Apply parameters based on use_model_default_sampling setting
        if use_model_default_sampling:
            # Fetch and use model's default SAMPLING parameters (not max_tokens)
            model_defaults = self.get_model_defaults()
            if model_defaults:
                print(f"[Prompt Generator] Applying model default sampling: temp={model_defaults.get('temperature')}, top_k={model_defaults.get('top_k')}, top_p={model_defaults.get('top_p')}, min_p={model_defaults.get('min_p')}, repeat_penalty={model_defaults.get('repeat_penalty')}")
                payload["temperature"] = round(float(model_defaults.get("temperature", 0.8)), 4)
                payload["top_k"] = int(model_defaults.get("top_k", 40))
                payload["top_p"] = round(float(model_defaults.get("top_p", 0.95)), 4)
                payload["min_p"] = round(float(model_defaults.get("min_p", 0.05)), 4)
                payload["repeat_penalty"] = round(float(model_defaults.get("repeat_penalty", 1.0)), 4)
            else:
                print("[Prompt Generator] Warning: Could not fetch model defaults, using fallback sampling values")
                payload["temperature"] = 0.8
                payload["top_k"] = 40
                payload["top_p"] = 0.95
                payload["min_p"] = 0.05
                payload["repeat_penalty"] = 1.0
            
            # max_tokens always from options or default (not a sampling param)
            payload["max_tokens"] = int(options.get("max_tokens", 8192)) if options else 8192
        else:
            # Use custom parameters from options or sensible defaults
            if options:
                if "temperature" in options:
                    payload["temperature"] = round(float(options["temperature"]), 4)
                if "top_p" in options:
                    payload["top_p"] = round(float(options["top_p"]), 4)
                if "top_k" in options:
                    payload["top_k"] = int(options["top_k"])
                if "min_p" in options:
                    payload["min_p"] = round(float(options["min_p"]), 4)
                if "repeat_penalty" in options:
                    payload["repeat_penalty"] = round(float(options["repeat_penalty"]), 4)
            
            # max_tokens always from options or default
            payload["max_tokens"] = int(options.get("max_tokens", 8192)) if options else 8192

        # Check cache (skip cache if images are present since they change)
        if not images and cache_key in self._prompt_cache and _current_model == model_to_use:
            print("[Prompt Generator] Returning cached prompt result.")
            
            if show_everything_in_console:
                self._print_debug_header(payload, enable_thinking, use_model_default_sampling)
                print(f"\n--- <|CACHED MODEL ANSWER|> STARTS ---")
                print(self._prompt_cache[cache_key])
                print(f"--- <|CACHED MODEL ANSWER|> ENDS ---\n")

            if stop_server_after:
                self.stop_server()
            return (self._prompt_cache[cache_key],)

        full_url = f"{self.SERVER_URL}/v1/chat/completions"

        try:
            if _current_model:
                print(f"[Prompt Generator] Generating with model: {_current_model}")
            
            if show_everything_in_console:
                self._print_debug_header(payload, enable_thinking, use_model_default_sampling)
                if images:
                    print(f"\n📷 Images attached: {len(images)}")
                print("\n--- <|REAL-TIME STREAM|> STARTS ---")

            response = requests.post(
                full_url,
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()

            full_response = ""
            thinking_content = ""
            in_thinking = False
            usage_stats = None
            
            for line in response.iter_lines():
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except comfy.model_management.InterruptProcessingException:
                    print("[Prompt Generator] Generation interrupted by user")
                    response.close()
                    if stop_server_after:
                        self.stop_server()
                    raise
                
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[6:]
                        
                        if json_str.strip() == '[DONE]':
                            break
                            
                        try:
                            chunk = json.loads(json_str)

                            if "usage" in chunk:
                                usage_stats = chunk["usage"]

                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                                    if show_everything_in_console and not in_thinking:
                                        print(content, end='', flush=True)
                                
                                # Handle reasoning content (only present when thinking is ON)
                                reasoning_delta = delta.get('reasoning_content', '')
                                if reasoning_delta:
                                    thinking_content += reasoning_delta
                                    if show_everything_in_console:
                                        if not in_thinking:
                                            print("\n\n🧠 [THINKING] ", end='', flush=True)
                                            in_thinking = True
                                        print(reasoning_delta, end='', flush=True)
                                
                                if in_thinking and content and not reasoning_delta:
                                    if show_everything_in_console:
                                        print("\n\n💡 [ANSWER] ", end='', flush=True)
                                    in_thinking = False
                                    
                        except json.JSONDecodeError:
                            pass

            if show_everything_in_console:
                print("\n--- <|REAL-TIME STREAM|> ENDS ---\n")
                
                print("="*60)
                print(" [Prompt Generator] TOKEN USAGE STATISTICS")
                print("="*60)

                total_input = usage_stats.get('prompt_tokens', 0) if usage_stats else 0
                total_output = usage_stats.get('completion_tokens', 0) if usage_stats else 0

                # Estimate text tokens using ~4 characters per token (rough average for English)
                CHARS_PER_TOKEN = 4
                sys_tokens_est = len(system_prompt) // CHARS_PER_TOKEN
                usr_tokens_est = len(prompt) // CHARS_PER_TOKEN
                text_tokens_est = sys_tokens_est + usr_tokens_est
                
                # Add some overhead for special tokens, formatting, etc.
                text_tokens_with_overhead = int(text_tokens_est * 1.2)
                
                if images:
                    # Image tokens = total input - estimated text tokens
                    image_tokens = max(0, total_input - text_tokens_with_overhead)
                    # Adjust text token estimates proportionally if needed
                    if text_tokens_with_overhead > total_input:
                        # No room for images, something's off - just report what we know
                        sys_tokens = total_input // 2
                        usr_tokens = total_input - sys_tokens
                        image_tokens = 0
                    else:
                        sys_tokens = sys_tokens_est
                        usr_tokens = usr_tokens_est
                else:
                    # No images - all input tokens are text
                    image_tokens = 0
                    # Distribute total_input proportionally based on character length
                    total_chars = len(system_prompt) + len(prompt)
                    if total_chars > 0:
                        sys_tokens = int(total_input * len(system_prompt) / total_chars)
                        usr_tokens = total_input - sys_tokens
                    else:
                        sys_tokens = 0
                        usr_tokens = 0

                think_len = len(thinking_content)
                ans_len = len(full_response)
                total_out_len = think_len + ans_len

                if total_output > 0 and total_out_len > 0:
                    think_tokens = int(total_output * (think_len / total_out_len))
                    ans_tokens = total_output - think_tokens
                else:
                    think_tokens = 0
                    ans_tokens = 0
                
                print(f" SYSTEM PROMPT: {sys_tokens:>5} tokens (est.)")
                print(f" USER PROMPT:   {usr_tokens:>5} tokens (est.)")
                if images and image_tokens > 0:
                    print(f" IMAGES:        {image_tokens:>5} tokens ({len(images)} image(s))")
                    avg_per_image = image_tokens / len(images)
                    print(f" (avg {avg_per_image:.0f} tokens per image)")
                print(f" -----------------------------")
                print(f" THINKING:      {think_tokens:>5} tokens (est.)")
                print(f" FINAL ANSWER:  {ans_tokens:>5} tokens (est.)")
                print(f" -----------------------------")
                print(f" TOTAL INPUT:    {total_input:>5} tokens (actual)")
                print(f" TOTAL OUTPUT:   {total_output:>5} tokens (actual)")
                
                print("="*60 + "\n")
            if not full_response:
                print("[Prompt Generator] Warning: Empty response from server")
                full_response = prompt

            print("[Prompt Generator] Successfully generated prompt")

            # Only cache if no images (images make caching impractical)
            if not images:
                self._prompt_cache[cache_key] = full_response

            if stop_server_after:
                self.stop_server()

            return (full_response,)

        except comfy.model_management.InterruptProcessingException:
            raise
        except requests.exceptions.HTTPError as e:
            # Capture detailed error from server
            error_body = ""
            try:
                error_body = e.response.text
            except:
                pass
            
            error_msg = f"Error: HTTP {e.response.status_code}"
            if error_body:
                error_msg += f"\nServer response: {error_body[:1000]}"
            
            print(f"[Prompt Generator] {error_msg}")
            
            if images and e.response.status_code == 500:
                print(f"[Prompt Generator] Note: This may be a VLM context issue. Current context: {context_size}")
            
            return (error_msg,)
        except requests.exceptions.ConnectionError:
            error_msg = f"Error: Could not connect to server at {full_url}. Server may have crashed."
            print(f"[Prompt Generator] {error_msg}")
            return (error_msg,)
        except requests.exceptions.Timeout:
            error_msg = "Error: Request timed out (>300s)"
            print(f"[Prompt Generator] {error_msg}")
            return (error_msg,)
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"[Prompt Generator] {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg,)


NODE_CLASS_MAPPINGS = {
    "PromptGenerator": PromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerator": "Prompt Generator"
}
