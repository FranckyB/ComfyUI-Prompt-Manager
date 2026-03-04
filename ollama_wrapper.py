"""
Ollama integration for ComfyUI Prompt Manager.

Provides model discovery, generation, and memory management
via Ollama's native API and OpenAI-compatible endpoint.
"""
import requests
import json

# ============================================================================
# Default endpoints
# ============================================================================

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _get_ollama_base(user_config):
    """Resolve Ollama base URL (without /v1 or /api) from config."""
    url = str(user_config.get("ollama_url", DEFAULT_OLLAMA_URL)).strip().rstrip("/")
    if not url:
        url = DEFAULT_OLLAMA_URL
    # Strip trailing /v1 or /api paths so we always have the root
    for suffix in ("/v1", "/api", "/api/tags", "/api/chat"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


# ============================================================================
# Model discovery
# ============================================================================

def discover_ollama_models(user_config):
    """Query Ollama for available models.

    Returns:
        (list_of_model_names, status_message)
    """
    base = _get_ollama_base(user_config)
    tags_url = f"{base}/api/tags"
    timeout = 12

    try:
        response = requests.get(tags_url, timeout=timeout)
        if response.status_code == 200:
            payload = response.json()
            names = []
            if isinstance(payload, dict):
                for item in payload.get("models", []):
                    if isinstance(item, dict):
                        name = item.get("model") or item.get("name")
                        if isinstance(name, str) and name.strip():
                            names.append(name.strip())
            if names:
                unique = sorted(set(names), key=lambda x: x.lower())
                return unique, f"Found {len(unique)} Ollama model(s)."
            return [], "Ollama responded but no models were found. Pull one first, e.g. `ollama pull llama3.1`."
        return [], f"Ollama returned HTTP {response.status_code}."
    except requests.exceptions.ConnectionError:
        return [], f"Could not connect to Ollama at {base}. Start `ollama serve` and retry."
    except requests.exceptions.Timeout:
        return [], f"Timed out querying Ollama at {base}."
    except Exception as e:
        return [], f"Failed to query Ollama models: {e}"


def is_ollama_available(user_config):
    """Check if Ollama is reachable."""
    base = _get_ollama_base(user_config)
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ============================================================================
# Generation (OpenAI-compatible chat/completions via Ollama)
# ============================================================================

def generate_chat(user_config, model_name, messages, temperature=0.8,
                  top_k=40, top_p=0.95, min_p=0.05, repeat_penalty=1.0,
                  seed=None, stream=True, timeout=120,
                  use_model_defaults=True, enable_thinking=False):
    """Send a chat completion request to Ollama's OpenAI-compatible endpoint.

    Args:
        user_config: preferences dict with ollama_url, ollama_keep_alive, etc.
        model_name: Ollama model tag (e.g. "llama3.1:latest")
        messages: list of {role, content} dicts (OpenAI format)
        stream: if True, returns a streaming response iterator
        timeout: request timeout in seconds

    Returns:
        If stream=False: (response_text, thinking_text, usage_stats, error_msg)
        If stream=True:  (response_object, error_msg) — caller handles streaming
    """
    base = _get_ollama_base(user_config)
    url = f"{base}/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
    }

    # Set keep_alive to control how long the model stays loaded after this request
    keep_alive = user_config.get("ollama_keep_alive", "5m")
    if keep_alive:
        payload["keep_alive"] = keep_alive

    if stream:
        payload["stream_options"] = {"include_usage": True}

    if seed is not None:
        payload["seed"] = seed

    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    if not use_model_defaults:
        payload["temperature"] = temperature
        payload["top_k"] = top_k
        payload["top_p"] = top_p
        payload["min_p"] = min_p
        payload["repeat_penalty"] = repeat_penalty

    try:
        response = requests.post(url, json=payload, timeout=timeout, stream=stream)
        if response.status_code != 200:
            detail = _extract_error(response)
            return (None, f"Ollama request failed ({response.status_code}): {detail}")
        if stream:
            return (response, None)
        else:
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            usage = result.get("usage")
            return (text.strip(), "", usage, None)
    except requests.exceptions.ConnectionError:
        return (None, "Could not connect to Ollama. Start 'ollama serve' and retry.")
    except requests.exceptions.Timeout:
        return (None, f"Ollama request timed out ({timeout}s). Try a shorter prompt or smaller model.")
    except (KeyError, IndexError):
        return (None, "Unexpected response format from Ollama.")
    except Exception as e:
        return (None, f"Ollama generation failed: {e}")


# ============================================================================
# Memory management — unload model from Ollama VRAM
# ============================================================================

def unload_model(user_config, model_name):
    """Tell Ollama to unload a model from memory (free VRAM).

    Uses the /api/generate endpoint with keep_alive=0 to immediately
    evict the model from GPU memory.
    """
    base = _get_ollama_base(user_config)
    url = f"{base}/api/generate"

    try:
        response = requests.post(url, json={
            "model": model_name,
            "keep_alive": 0,
        }, timeout=15)
        if response.status_code == 200:
            return True, f"Model '{model_name}' unloaded from Ollama memory."
        return False, f"Ollama returned HTTP {response.status_code} while unloading model."
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to Ollama to unload model."
    except Exception as e:
        return False, f"Failed to unload model from Ollama: {e}"


# ============================================================================
# Helpers
# ============================================================================

def _extract_error(response):
    """Extract error message from an Ollama error response."""
    try:
        payload = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:300] if text else f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str) and err.strip():
            return err.strip()
        msg = payload.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()

    return f"HTTP {response.status_code}"
