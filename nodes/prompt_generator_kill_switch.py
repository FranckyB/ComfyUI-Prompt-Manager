from .prompt_generator import PromptGenerator, _preferences_cache, ollama_unload_model


class PromptGeneratorKillSwitch:
    """Pass-through node that also stops the local llm server."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {"forceInput": True, "tooltip": "Any input value to pass through unchanged."}),
                "ollama_model": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional Ollama model name to unload. Empty uses preferred model from settings."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "pass_and_kill"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Pass-through any value, stop Prompt Generator's llama.cpp server, and unload an Ollama model from memory."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _unload_ollama(self, ollama_model: str):
        model_name = str(ollama_model or "").strip()
        if not model_name:
            model_name = str(_preferences_cache.get("preferred_model", "") or "").strip()

        if not model_name:
            print("Prompt Generator Kill Switch: No Ollama model provided/preferred; skipping Ollama unload.")
            return

        ok, msg = ollama_unload_model(_preferences_cache, model_name)
        print(f"Prompt Generator Kill Switch: {msg}")
        if not ok:
            print("Prompt Generator Kill Switch: Ollama unload did not succeed.")

    def pass_and_kill(self, value, ollama_model=""):
        PromptGenerator.stop_server()
        self._unload_ollama(ollama_model)
        return (value,)
