from .prompt_generator import PromptGenerator, _preferences_cache, ollama_unload_model


class PromptGeneratorKillSwitch:
    """Pass-through node that also stops the local llm server."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {"forceInput": True, "tooltip": "Any input value to pass through unchanged."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "pass_and_kill"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Pass-through any value, stop Prompt Generator's llama.cpp server, and unload all Ollama models from memory."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def pass_and_kill(self, value):
        PromptGenerator.stop_server()
        ok, msg = ollama_unload_model(_preferences_cache)
        if msg:
            print(f"Prompt Generator Kill Switch: {msg}")
        return (value,)
