"""
Expression Selector - substitutes [expression] in a prompt with a saved expression.

Expressions are stored as normal Prompt Manager prompts under the "Expressions"
category, so the existing prompt browser, thumbnail generation, and save flow can
be reused.
"""
import json
import math
import os
import re
import server

from .prompt_manager_basic import PromptManager, _get_workflow_node


_EXPRESSION_CATEGORY = "Expressions"


def _find_expression_case_insensitive(prompts_data, name):
    """Return the expression entry dict and the canonical category name."""
    if not isinstance(prompts_data, dict):
        return None, _EXPRESSION_CATEGORY

    # Find the canonical category name (preserve user casing).
    category = _EXPRESSION_CATEGORY
    for cat in prompts_data.keys():
        if cat.lower() == _EXPRESSION_CATEGORY.lower():
            category = cat
            break

    entries = prompts_data.get(category)
    if not isinstance(entries, dict):
        return None, category

    if name in entries:
        entry = entries[name]
        return (entry if isinstance(entry, dict) else None), category

    name_lower = str(name or "").lower()
    for entry_name, entry in entries.items():
        if entry_name == "__meta__":
            continue
        if entry_name.lower() == name_lower:
            return (entry if isinstance(entry, dict) else None), category

    return None, category


class ExpressionSelector:
    """Select an expression and substitute it into a prompt."""

    @classmethod
    def INPUT_TYPES(s):
        prompts_data = PromptManager.load_prompts()
        expression_names = []
        first_text = ""

        if isinstance(prompts_data, dict):
            for cat, entries in prompts_data.items():
                if cat.lower() != _EXPRESSION_CATEGORY.lower():
                    continue
                if not isinstance(entries, dict):
                    continue
                for entry_name, entry in entries.items():
                    if entry_name == "__meta__":
                        continue
                    expression_names.append(entry_name)
                break

        expression_names = sorted(expression_names, key=str.lower)
        if not expression_names:
            expression_names = [""]

        first_name = expression_names[0]
        if first_name:
            entry, _ = _find_expression_case_insensitive(prompts_data, first_name)
            if isinstance(entry, dict):
                first_text = entry.get("prompt", "") or ""

        return {
            "required": {
                "name": (expression_names, {"default": first_name}),
                "subject_gender": (["female", "male"], {
                    "default": "female",
                    "tooltip": "Adapt gendered pronouns in the expression so it matches the described subject.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "lazy": True,
                    "tooltip": "Connect a base prompt. The selected expression will be appended after it, prefixed with 'expression:'.",
                }),
                "expression_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Apply Comfy-style prompt strength to the expression. 1 keeps the default behavior.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "api_prompt": "PROMPT",
            }
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Append a saved expression from the Expression category after the input prompt."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select_expression"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        """Allow any expression name, including temporary unsaved ones."""
        return True

    @classmethod
    def IS_CHANGED(cls, name, subject_gender="female", prompt="", expression_strength=1.0, **kwargs):
        return (name, subject_gender, expression_strength, prompt)

    _GENDER_SUBSTITUTIONS = {
        # Source expressions are written with male pronouns; adapt to female.
        "female": {
            r"\bhe\b": "she",
            r"\bhe's\b": "she's",
            r"\bhe'll\b": "she'll",
            r"\bhe'd\b": "she'd",
            r"\bhim\b": "her",
            r"\bhis\b": "her",
            r"\bhimself\b": "herself",
        },
        # Source expressions are written with female pronouns; adapt to male.
        # Object-position "her" is mapped first to "him" before the generic
        # possessive fallback turns remaining "her" into "his".
        "male": {
            r"\bat her\b": "at him",
            r"\bto her\b": "to him",
            r"\bwith her\b": "with him",
            r"\bnear her\b": "near him",
            r"\bbehind her\b": "behind him",
            r"\bfor her\b": "for him",
            r"\babout her\b": "about him",
            r"\bon her\b": "on him",
            r"\bshe\b": "he",
            r"\bshe's\b": "he's",
            r"\bshe'll\b": "he'll",
            r"\bshe'd\b": "he'd",
            r"\bhers\b": "his",
            r"\bherself\b": "himself",
            r"\bher\b": "his",
        },
    }

    @classmethod
    def _apply_gender_substitutions(cls, text, subject_gender):
        """Swap pronouns in the expression to match the selected subject gender."""
        if subject_gender not in cls._GENDER_SUBSTITUTIONS:
            return text
        substitutions = cls._GENDER_SUBSTITUTIONS[subject_gender]
        result = str(text or "").lower()
        # Apply longest patterns first to avoid partial replacements interfering.
        for pattern in sorted(substitutions.keys(), key=len, reverse=True):
            result = re.sub(pattern, substitutions[pattern], result)
        return result

    @staticmethod
    def _normalize_expression_strength(expression_strength):
        try:
            strength = float(expression_strength)
        except (TypeError, ValueError):
            strength = 1.0
        if not math.isfinite(strength):
            strength = 1.0
        return max(0.0, min(5.0, strength))

    @staticmethod
    def _substitute_expression(prompt, expression_text, subject_gender="female", expression_strength=1.0):
        expr = expression_text if isinstance(expression_text, str) else str(expression_text)
        expr = ExpressionSelector._apply_gender_substitutions(expr.strip(), subject_gender)

        # If the selected expression is the special "(none)" sentinel, return the
        # original prompt unchanged so no expression block is appended.
        if expr.lower() == "(none)":
            return prompt if isinstance(prompt, str) else ""

        strength_value = ExpressionSelector._normalize_expression_strength(expression_strength)
        strength_text = f"{strength_value:.15g}"

        if strength_value == 1.0:
            if expr and not expr.lower().startswith("expression:"):
                expr = f"expression: {expr}"
            if not isinstance(prompt, str) or not prompt.strip():
                return expr
            return f"{prompt.strip()}\n{expr}"

        stripped_expr = expr[10:].lstrip() if expr.lower().startswith("expression:") else expr
        formatted_expr = f"({stripped_expr}:{strength_text})"

        if not isinstance(prompt, str) or not prompt.strip():
            return formatted_expr

        return f"{prompt.strip()}\n{formatted_expr}"

    @staticmethod
    def _patch_runtime_prompt_metadata(unique_id, output_text, extra_pnginfo=None, api_prompt=None):
        """Persist the resolved prompt into workflow/api metadata for downstream save nodes."""
        node_id = str(unique_id) if unique_id is not None else ""
        if not node_id:
            return

        # Copy-on-write: these are live objects used by ComfyUI for cache bookkeeping.
        workflow_node = _get_workflow_node(extra_pnginfo, node_id)
        if isinstance(workflow_node, dict):
            widgets = workflow_node.get("widgets_values")
            if isinstance(widgets, list) and len(widgets) > 1:
                # ExpressionSelector widgets order: [name, prompt]
                workflow_node["widgets_values"] = widgets[:1] + [output_text] + widgets[2:]

        if isinstance(api_prompt, dict):
            prompt_node = api_prompt.get(node_id)
            if isinstance(prompt_node, dict):
                inputs = prompt_node.get("inputs")
                if isinstance(inputs, dict):
                    prompt_node["inputs"] = {**inputs, "prompt": output_text}

    def select_expression(self, name, subject_gender="female", prompt="", expression_strength=1.0, unique_id=None, extra_pnginfo=None, api_prompt=None):
        prompts_data = PromptManager.load_prompts()
        entry, _ = _find_expression_case_insensitive(prompts_data, name)
        expression_text = ""
        thumbnail = None
        if isinstance(entry, dict):
            expression_text = entry.get("prompt", "") or ""
            thumbnail = entry.get("thumbnail")

        output_text = self._substitute_expression(prompt, expression_text, subject_gender, expression_strength)

        if unique_id is not None:
            server.PromptServer.instance.send_sync("expression-selector-update", {
                "node_id": unique_id,
                "name": name,
                "subject_gender": subject_gender,
                "expression_text": expression_text,
                "output_text": output_text,
                "thumbnail": thumbnail,
            })

        self._patch_runtime_prompt_metadata(unique_id, output_text, extra_pnginfo, api_prompt)
        return (output_text,)

    def check_lazy_status(self, name, subject_gender="female", prompt=None, **kwargs):
        return ["prompt"] if prompt is None else []


@server.PromptServer.instance.routes.post("/prompt-manager/merge-default-expressions")
async def merge_default_expressions(request):
    """Merge bundled expressions.json into the user's prompt data if no Expressions category exists."""
    try:
        default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "expressions.json")
        default_expressions = {}
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                default_expressions = json.load(f)

        prompts = PromptManager.load_prompts()

        # Case-insensitive check for an existing Expressions category.
        existing_category = None
        for cat in prompts.keys():
            if cat.lower() == _EXPRESSION_CATEGORY.lower():
                existing_category = cat
                break

        if existing_category is not None:
            return server.web.json_response({"success": True, "merged": False, "prompts": prompts})

        # Merge default expressions into user data, preserving the bundled structure.
        for category, entries in default_expressions.items():
            if category not in prompts:
                prompts[category] = {}
            if not isinstance(entries, dict):
                continue
            for prompt_name, prompt_data in entries.items():
                if prompt_name == "__meta__":
                    continue
                prompts[category][prompt_name] = prompt_data if isinstance(prompt_data, dict) else {"prompt": prompt_data}

        PromptManager.save_prompts(prompts)
        return server.web.json_response({"success": True, "merged": True, "prompts": prompts})
    except Exception as e:
        print(f"[ExpressionSelector] Error merging default expressions: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)
