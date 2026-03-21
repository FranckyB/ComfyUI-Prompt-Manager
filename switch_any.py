"""
ComfyUI Switch Any - A universal switch node with named inputs.
Selects one of up to 10 inputs to pass through. Non-selected inputs are
stripped from the execution graph so they are never evaluated by ComfyUI.
"""

import re


def parse_names(names_str, count):
    """Split a names string by comma or semicolon into a list of *count* names."""
    parts = re.split(r"[;,]", names_str)
    parts = [p.strip() for p in parts if p.strip()]
    # Pad with defaults if the user provided fewer names than count
    result = []
    for i in range(count):
        if i < len(parts):
            result.append(parts[i])
        else:
            result.append(f"Input {i + 1}")
    return result


class SwitchAny:
    """Universal switch node that passes through one of up to 10 named inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        default_names = [f"Input {i + 1}" for i in range(10)]
        return {
            "required": {
                "select": (default_names, {"default": default_names[0]}),
                "num_inputs": ("INT", {
                    "default": 2, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of active inputs (1-10)"
                }),
                "names": ("STRING", {
                    "default": "",
                    "placeholder": "WAN; FLUX; SDXL  (comma or semicolon separated)",
                    "tooltip": "Custom names for each input, separated by comma or semicolon"
                }),
            },
            "optional": {
                **{
                    f"input_{i + 1}": ("*", {"lazy": True})
                    for i in range(10)
                },
            },
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "A Universal switch with up to 10 named inputs. With True Lazy Evaluation, only the selected input is evaluated. Other inputs are completely ignored by ComfyUI, allowing you to switch between different branches of your workflow without any performance cost from the inactive branches."
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def check_lazy_status(self, select, num_inputs=2, names="", **kwargs):
        """Only request the single selected input — everything else stays dormant."""
        name_list = parse_names(names, num_inputs)
        for i, name in enumerate(name_list):
            if name == select:
                return [f"input_{i + 1}"]
        # Fallback: first input
        return ["input_1"]

    def switch(self, select, num_inputs=2, names="", **kwargs):
        name_list = parse_names(names, num_inputs)
        for i, name in enumerate(name_list):
            if name == select:
                value = kwargs.get(f"input_{i + 1}")
                if value is None:
                    raise ValueError(f"Selected input '{name}' is not connected.")
                return (value,)
        raise ValueError(f"No input matches selection '{select}'.")


class SwitchAnyBool:
    """Boolean switch node — passes through the on_true or on_false input based on a toggle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "on_true": ("*", {"lazy": True}),
                "on_false": ("*", {"lazy": True}),
            },
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Boolean switch with true/false inputs. Only the active branch is evaluated."
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def check_lazy_status(self, condition, **kwargs):
        return ["on_true"] if condition else ["on_false"]

    def switch(self, condition, on_true=None, on_false=None):
        value = on_true if condition else on_false
        label = "on_true" if condition else "on_false"
        if value is None:
            raise ValueError(f"Selected input '{label}' is not connected.")
        return (value,)
