import requests
import subprocess
import time
import os
import atexit
import psutil
import json
import signal
import sys
from .model_manager import get_local_models, get_model_path, is_model_local, download_model

# Import ComfyUI's model management for interrupt handling
import comfy.model_management

# Global variable to track the server process
_server_process = None
_current_model = None

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
        
        # Console control handler types
        CTRL_C_EVENT = 0
        CTRL_BREAK_EVENT = 1
        CTRL_CLOSE_EVENT = 2
        CTRL_LOGOFF_EVENT = 5
        CTRL_SHUTDOWN_EVENT = 6
        
        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
        def console_handler(event):
            # ONLY handle close/logoff/shutdown - NOT Ctrl+C or Ctrl+Break
            if event in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
                print(f"\n[Prompt Generator] Console closing (event {event}), cleaning up...")
                cleanup_server()
                return False  # Let other handlers run
            # For Ctrl+C and Ctrl+Break, return False to let ComfyUI handle them
            return False
        
        # Keep a reference to prevent garbage collection
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
    global _server_process, _current_model
    
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

# Set up Windows Job Object (most reliable method)
setup_windows_job_object()

# Set up console control handler for Windows (close button only)
setup_console_handler()

# Register signal handlers - ONLY SIGTERM, not SIGINT (Ctrl+C)
try:
    signal.signal(signal.SIGTERM, signal_handler)
except Exception as e:
    print(f"[Prompt Generator] Warning: Could not register signal handlers: {e}")

# Register atexit cleanup as fallback
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

            # Start server process WITH --reasoning-format deepseek for thinking models
            _server_process = subprocess.Popen(
                [server_cmd, "-m", model_path, "--port", str(PromptGenerator.SERVER_PORT), 
                "--no-warmup", "--reasoning-format", "deepseek"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags
            )

            assign_process_to_job(_server_process)

            _current_model = model_name

            # Assign to job object for guaranteed cleanup on Windows
            assign_process_to_job(_server_process)

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

    def _print_debug_header(self, payload):
        """Helper to print colorful debug info header"""
        print("\n" + "="*60)
        print(" [Prompt Generator] DEBUG: TOKENS & MESSAGES IN ACTION")
        print("="*60)
        
        # Print System and User Prompt from payload
        if "messages" in payload:
            for msg in payload["messages"]:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                print(f"\n--- <|{role}|> STARTS ---")
                print(content)
                print(f"--- <|{role}|> ENDS ---")
        
        # Print parameters
        print("\n--- GENERATION PARAMS ---")
        params = {k: v for k, v in payload.items() if k != "messages"}
        print(json.dumps(params, indent=2))

    def convert_prompt(self, prompt: str, seed: int, stop_server_after=False, show_everything_in_console=False, options=None) -> str:
        """Convert prompt using llama.cpp server, with caching for repeated requests."""
        global _current_model

        # If prompt is empty, return empty string
        if not prompt.strip():
            return ("",)

        # Always determine a valid model filename before running server
        model_to_use = None
        
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

        # Prepare system prompt (needed for cache key)
        if options and "system_prompt" in options:
            system_prompt = options["system_prompt"]
        else:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # Caching logic
        options_tuple = tuple(sorted(options.items())) if options else ()
        cache_key = (prompt, seed, model_to_use, options_tuple)
        
        # Prepare the payload structure
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "seed": seed,
            "max_tokens": 8192  # Default max tokens
        }
                
        # Add optional parameters if provided via options
        if options:
            if "max_tokens" in options:
                payload["max_tokens"] = options["max_tokens"]
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

        # Only use cache if the model has not changed since last use
        if cache_key in self._prompt_cache and _current_model == model_to_use:
            print("[Prompt Generator] Returning cached prompt result.")
            
            if show_everything_in_console:
                self._print_debug_header(payload)
                print(f"\n--- <|CACHED MODEL ANSWER|> STARTS ---")
                print(self._prompt_cache[cache_key])
                print(f"--- <|CACHED MODEL ANSWER|> ENDS ---\n")

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

        try:
            if _current_model:
                print(f"[Prompt Generator] Generating with model: {_current_model}")
            
            if show_everything_in_console:
                self._print_debug_header(payload)
                print("\n--- <|REAL-TIME STREAM|> STARTS ---")

            # Send request with stream=True
            response = requests.post(
                full_url,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            full_response = ""
            thinking_content = ""
            in_thinking = False
            usage_stats = None
            
            # Process the stream
            for line in response.iter_lines():
                # CHECK FOR COMFYUI INTERRUPT
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except comfy.model_management.InterruptProcessingException:
                    print("[Prompt Generator] Generation interrupted by user")
                    response.close()
                    if stop_server_after:
                        self.stop_server()
                    # Re-raise the exception so ComfyUI handles it properly
                    raise
                
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[6:]  # Remove 'data: ' prefix
                        
                        # Check for stream end signal
                        if json_str.strip() == '[DONE]':
                            break
                            
                        try:
                            chunk = json.loads(json_str)

                            # Capture usage stats if present (usually in the last chunk)
                            if "usage" in chunk:
                                usage_stats = chunk["usage"]

                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                
                                # Handle main content
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                                    if show_everything_in_console and not in_thinking:
                                        print(content, end='', flush=True)
                                
                                # Handle reasoning/thinking content
                                reasoning_delta = delta.get('reasoning_content', '')
                                if reasoning_delta and show_everything_in_console:
                                    thinking_content += reasoning_delta
                                    if not in_thinking:
                                        print("\n\n🧠 [THINKING] ", end='', flush=True)
                                        in_thinking = True
                                    print(reasoning_delta, end='', flush=True)
                                
                                # Check if we exited thinking mode
                                if in_thinking and content and not reasoning_delta:
                                    print("\n\n💡 [ANSWER] ", end='', flush=True)
                                    in_thinking = False
                                    
                        except json.JSONDecodeError:
                            pass

            # Finalize debug output
            if show_everything_in_console:
                print("\n--- <|REAL-TIME STREAM|> ENDS ---\n")
                
                # --- TOKEN STATS LOGIC ---
                print("="*60)
                print(" [Prompt Generator] TOKEN USAGE STATISTICS")
                print("="*60)

                # Use server stats if available, otherwise 0
                total_input = usage_stats.get('prompt_tokens', 0) if usage_stats else 0
                total_output = usage_stats.get('completion_tokens', 0) if usage_stats else 0

                # Calculate Proportions (Approximation based on char length)
                # Since we don't have a tokenizer locally, we split the total input tokens
                # based on the character length ratio of system vs user prompt.
                sys_len = len(system_prompt)
                usr_len = len(prompt)
                total_in_len = sys_len + usr_len
                
                if total_input > 0 and total_in_len > 0:
                    sys_tokens = int(total_input * (sys_len / total_in_len))
                    usr_tokens = total_input - sys_tokens
                else:
                    sys_tokens = 0
                    usr_tokens = 0

                # Same logic for Output (Thinking vs Answer)
                think_len = len(thinking_content)
                ans_len = len(full_response)
                total_out_len = think_len + ans_len

                if total_output > 0 and total_out_len > 0:
                    think_tokens = int(total_output * (think_len / total_out_len))
                    ans_tokens = total_output - think_tokens
                else:
                    think_tokens = 0
                    ans_tokens = 0
                
                print(f" SYSTEM PROMPT: {sys_tokens:>6} tokens")
                print(f" USER PROMPT:   {usr_tokens:>6} tokens")
                print(f" -----------------------------")
                print(f" THINKING:      {think_tokens:>6} tokens")
                print(f" FINAL ANSWER:  {ans_tokens:>6} tokens")
                print(f" -----------------------------")
                print(f" TOTAL INPUT:   {total_input:>6} tokens")
                print(f" TOTAL OUTPUT:  {total_output:>6} tokens")
                print("="*60 + "\n")

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

        except comfy.model_management.InterruptProcessingException:
            # Re-raise interrupt exceptions so ComfyUI handles them properly
            raise
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
