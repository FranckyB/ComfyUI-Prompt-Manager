#!/usr/bin/env python3
"""
Contract tests for Recipe Builder execute behavior.

This test file is intentionally self-contained and stubs ComfyUI runtime modules
so it can run outside a full ComfyUI session:
    python tests/test_recipe_builder_contract.py

Expected flow-of-data contract (maintainer note):
1. Builder can run with or without incoming recipe_data.
2. When recipe_data is connected, Builder pulls all model slots (A/B/C/D).
3. User-authored UI state lives in per-slot profiles and lock flags.
4. Connected multi inputs (prompts/loras) may update slot state,
    but locked sections must be protected from those inputs.
5. LoRAs are additive: recipe_data LoRAs plus connected input LoRAs,
    while preserving user state (active/order/strength) and missing flags.
6. If a multi input is disconnected, treat it as an empty list/payload so
    stale input-derived state does not persist.
7. Final rule: after update/merge, execute must emit recipe_data authored from
    the now-populated Builder state. Prompts/samplers/resolution/LoRAs should
    reflect what the user has set (subject to locks and connected inputs).
"""

import importlib.util
import json
import os
import random
import sys
import types
import unittest
from copy import deepcopy


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPE_BUILDER_PATH = os.path.join(ROOT, "nodes", "recipe_builder.py")


def _install_stub_modules():
    """Install lightweight runtime stubs required to import recipe_builder.py."""
    if "server" not in sys.modules:
        server_mod = types.ModuleType("server")

        class _Routes:
            def get(self, *_args, **_kwargs):
                return lambda fn: fn

            def post(self, *_args, **_kwargs):
                return lambda fn: fn

        class _PromptServerInstance:
            def __init__(self):
                self.routes = _Routes()

            def send_sync(self, *_args, **_kwargs):
                return None

        server_mod.PromptServer = types.SimpleNamespace(instance=_PromptServerInstance())
        server_mod.web = types.SimpleNamespace(
            json_response=lambda data, **_kwargs: data,
            Response=lambda **kwargs: kwargs,
        )
        sys.modules["server"] = server_mod

    if "folder_paths" not in sys.modules:
        folder_paths_mod = types.ModuleType("folder_paths")
        folder_paths_mod.models_dir = ROOT
        folder_paths_mod.get_filename_list = lambda _kind=None: []
        folder_paths_mod.get_full_path = lambda *_args, **_kwargs: ""
        folder_paths_mod.get_folder_paths = lambda *_args, **_kwargs: []
        sys.modules["folder_paths"] = folder_paths_mod

    if "comfy" not in sys.modules:
        comfy_mod = types.ModuleType("comfy")
        comfy_mod.__path__ = []
        comfy_samplers = types.ModuleType("comfy.samplers")
        comfy_samplers.KSampler = types.SimpleNamespace(
            SAMPLERS=["euler"],
            SCHEDULERS=["simple"],
        )
        comfy_sd = types.ModuleType("comfy.sd")
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_sample = types.ModuleType("comfy.sample")
        comfy_model_mgmt = types.ModuleType("comfy.model_management")
        comfy_mod.samplers = comfy_samplers
        comfy_mod.sd = comfy_sd
        comfy_mod.utils = comfy_utils
        comfy_mod.sample = comfy_sample
        comfy_mod.model_management = comfy_model_mgmt
        sys.modules["comfy"] = comfy_mod
        sys.modules["comfy.samplers"] = comfy_samplers
        sys.modules["comfy.sd"] = comfy_sd
        sys.modules["comfy.utils"] = comfy_utils
        sys.modules["comfy.sample"] = comfy_sample
        sys.modules["comfy.model_management"] = comfy_model_mgmt

    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch_mod

    if "numpy" not in sys.modules:
        numpy_mod = types.ModuleType("numpy")
        sys.modules["numpy"] = numpy_mod

    if "PIL" not in sys.modules:
        pil_mod = types.ModuleType("PIL")

        class _DummyImage:
            @staticmethod
            def open(*_args, **_kwargs):
                return None

        pil_mod.Image = _DummyImage
        pil_mod.ImageOps = types.SimpleNamespace(exif_transpose=lambda img: img)
        sys.modules["PIL"] = pil_mod


def _install_recipe_builder_dependency_stubs(pkg_root_name):
    """Install package-scoped stubs used by recipe_builder imports."""
    pkg_mod = types.ModuleType(pkg_root_name)
    pkg_mod.__path__ = [ROOT]
    sys.modules[pkg_root_name] = pkg_mod

    nodes_pkg_name = f"{pkg_root_name}.nodes"
    nodes_pkg_mod = types.ModuleType(nodes_pkg_name)
    nodes_pkg_mod.__path__ = [os.path.join(ROOT, "nodes")]
    sys.modules[nodes_pkg_name] = nodes_pkg_mod

    py_pkg_name = f"{pkg_root_name}.py"
    py_pkg_mod = types.ModuleType(py_pkg_name)
    py_pkg_mod.__path__ = [os.path.join(ROOT, "py")]
    sys.modules[py_pkg_name] = py_pkg_mod

    prompt_extractor_name = f"{pkg_root_name}.nodes.prompt_extractor"
    prompt_extractor_mod = types.ModuleType(prompt_extractor_name)
    prompt_extractor_mod.parse_workflow_for_prompts = lambda *_args, **_kwargs: {}
    prompt_extractor_mod.extract_metadata_from_png = lambda *_args, **_kwargs: {}
    prompt_extractor_mod.extract_metadata_from_jpeg = lambda *_args, **_kwargs: {}
    prompt_extractor_mod.extract_metadata_from_json = lambda *_args, **_kwargs: {}
    prompt_extractor_mod.extract_metadata_from_video = lambda *_args, **_kwargs: {}
    prompt_extractor_mod.build_node_map = lambda *_args, **_kwargs: {}
    prompt_extractor_mod.build_link_map = lambda *_args, **_kwargs: {}
    prompt_extractor_mod.extract_video_frame_av_to_tensor = lambda *_args, **_kwargs: None
    prompt_extractor_mod.get_cached_video_frame = lambda *_args, **_kwargs: None
    prompt_extractor_mod.get_placeholder_image_tensor = lambda *_args, **_kwargs: None
    sys.modules[prompt_extractor_name] = prompt_extractor_mod

    wf_families_name = f"{pkg_root_name}.py.workflow_families"
    wf_families_mod = types.ModuleType(wf_families_name)
    wf_families_mod.MODEL_FAMILIES = {
        "sdxl": {"label": "SDXL", "checkpoint": True},
        "wan": {"label": "Wan", "checkpoint": True},
    }
    wf_families_mod.get_model_family = lambda *_args, **_kwargs: "sdxl"
    wf_families_mod.get_family_label = lambda family: str(family or "")
    wf_families_mod.get_family_sampler_strategy = lambda *_args, **_kwargs: "single"
    wf_families_mod.get_compatible_families = lambda family: [family] if family else ["sdxl"]
    wf_families_mod.get_all_family_labels = lambda: ["SDXL", "Wan"]

    def _compat_models(*_args, **kwargs):
        rows = ["ckpt_a", "ckpt_b", "ckpt_c", "ckpt_d", "standalone_ckpt_c"]
        if kwargs.get("return_recommended"):
            return rows, (rows[0] if rows else "")
        return rows

    def _compat_vaes(*_args, **kwargs):
        rows = ["vae_default"]
        if kwargs.get("return_recommended"):
            return rows, (rows[0] if rows else "")
        return rows

    def _compat_clips(*_args, **kwargs):
        rows = ["clip_a"]
        if kwargs.get("return_recommended"):
            return rows, list(rows)
        return rows

    wf_families_mod.list_compatible_models = _compat_models
    wf_families_mod.list_compatible_vaes = _compat_vaes
    wf_families_mod.list_compatible_clips = _compat_clips
    sys.modules[wf_families_name] = wf_families_mod

    wf_extract_name = f"{pkg_root_name}.py.workflow_extraction_utils"
    wf_extract_mod = types.ModuleType(wf_extract_name)
    wf_extract_mod.extract_sampler_params = lambda *_args, **_kwargs: {}
    wf_extract_mod.extract_vae_info = lambda *_args, **_kwargs: {"name": "", "source": ""}
    wf_extract_mod.extract_clip_info = lambda *_args, **_kwargs: {"names": [], "type": "", "source": ""}
    wf_extract_mod.extract_resolution = lambda *_args, **_kwargs: {"width": 768, "height": 1280, "batch_size": 1, "length": None}
    wf_extract_mod.resolve_model_name = lambda name, _family=None: (name or "", bool(name))
    wf_extract_mod.resolve_vae_name = lambda name, _family=None: (name or "", bool(name))
    wf_extract_mod.resolve_clip_names = lambda names, _family=None: (list(names or []), bool(names))

    def _build_simplified(extracted, wf_overrides, sampler_params):
        return {
            "_source": "RecipeBuilder",
            "positive_prompt": wf_overrides.get("positive_prompt", extracted.get("positive_prompt", "")),
            "negative_prompt": wf_overrides.get("negative_prompt", extracted.get("negative_prompt", "")),
            "model_a": wf_overrides.get("model_a", extracted.get("model_a", "")),
            "model_b": wf_overrides.get("model_b", extracted.get("model_b", "")),
            "family": extracted.get("model_family", ""),
            "vae": wf_overrides.get("vae", extracted.get("vae", {}).get("name", "")),
            "clip": wf_overrides.get("clip_names", extracted.get("clip", {}).get("names", [])),
            "loras_a": list(extracted.get("loras_a", [])),
            "loras_b": list(extracted.get("loras_b", [])),
            "sampler": dict(sampler_params or {}),
            "resolution": dict(extracted.get("resolution", {})),
            "clip_type": extracted.get("clip", {}).get("type", ""),
            "loader_type": extracted.get("loader_type", ""),
        }

    wf_extract_mod.build_simplified_workflow_data = _build_simplified
    sys.modules[wf_extract_name] = wf_extract_mod

    wf_data_name = f"{pkg_root_name}.py.workflow_data_utils"
    wf_data_mod = types.ModuleType(wf_data_name)
    wf_data_mod.ensure_v2_recipe_data = lambda data: data
    wf_data_mod.strip_runtime_objects = lambda data: data
    wf_data_mod.to_json_safe_workflow_data = lambda data: data
    sys.modules[wf_data_name] = wf_data_mod

    lora_utils_name = f"{pkg_root_name}.py.lora_utils"
    lora_utils_mod = types.ModuleType(lora_utils_name)

    def _resolve_lora_path(name):
        if not name:
            return "", False
        return (f"/loras/{name}", name != "missing_lora")

    lora_utils_mod.resolve_lora_path = _resolve_lora_path
    lora_utils_mod.strip_lora_extension = lambda n: str(n or "").replace(".safetensors", "")
    sys.modules[lora_utils_name] = lora_utils_mod


def _load_recipe_builder_module():
    _install_stub_modules()
    pkg_root_name = "cpm_testpkg"
    _install_recipe_builder_dependency_stubs(pkg_root_name)

    module_name = f"{pkg_root_name}.nodes.recipe_builder"
    spec = importlib.util.spec_from_file_location(module_name, RECIPE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for recipe_builder.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_model_block(name="", positive="", negative="", loras=None):
    return {
        "positive_prompt": positive,
        "negative_prompt": negative,
        "family": "sdxl",
        "model": name,
        "loras": list(loras or []),
        "clip_type": "",
        "loader_type": "",
        "vae": "",
        "clip": [],
        "sampler": {
            "steps": 20,
            "cfg": 5.0,
            "denoise": 1.0,
            "seed": 0,
            "sampler_name": "euler",
            "scheduler": "simple",
        },
        "resolution": {"width": 768, "height": 1280, "batch_size": 1, "length": None},
    }


def _make_base_recipe():
    return {
        "version": 2,
        "models": {
            "model_a": _make_model_block("ckpt_a", "pos_a", "neg_a", [{"name": "recipe_lora", "model_strength": 0.8, "clip_strength": 0.7, "active": True}]),
            "model_b": _make_model_block("ckpt_b", "pos_b", "neg_b"),
            "model_c": _make_model_block("ckpt_c", "pos_c", "neg_c"),
            "model_d": _make_model_block("ckpt_d", "pos_d", "neg_d"),
        },
    }


def _default_slot_profile(model_name="", pos="", neg=""):
    return {
        "ov": {
            "positive_prompt": pos,
            "negative_prompt": neg,
            "model_a": model_name,
            "_family": "sdxl",
            "clip_type": "",
            "loader_type": "",
            "vae": "",
            "clip_names": [],
            "loras_a": [],
            "steps_a": 20,
            "cfg": 5.0,
            "denoise": 1.0,
            "seed_a": 0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "_seed_auto": False,
            "width": 768,
            "height": 1280,
            "batch_size": 1,
            "length": None,
            "_section_locks": {
                "prompt": False,
                "model": False,
                "loras": False,
                "sampler": False,
                "resolution": False,
                "seed": False,
                "positive": False,
                "negative": False,
            },
        },
        "ls": {},
    }


class RecipeBuilderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rb_module = _load_recipe_builder_module()
        cls.builder = cls.rb_module.WorkflowBuilder()

    def _exec(self, **kwargs):
        defaults = {
            "recipe_data": None,
            "multi_pos_prompts": "",
            "multi_neg_prompts": "",
            "multi_lora_stack": {},
            "override_data": "{}",
            "lora_state": "{}",
            "unique_id": "contract-test",
        }
        defaults.update(kwargs)
        result = self.builder.execute(**defaults)
        self.assertIn("result", result)
        self.assertIn("ui", result)
        return result

    def test_recipe_data_pull_preserves_all_four_models(self):
        recipe = _make_base_recipe()
        slot_profiles = {
            "model_a": _default_slot_profile(),
            "model_b": _default_slot_profile(),
            "model_c": _default_slot_profile(),
            "model_d": _default_slot_profile(),
        }
        override_data = json.dumps({"_model_slot": "model_a", "_slot_profiles": slot_profiles})

        out = self._exec(recipe_data=deepcopy(recipe), override_data=override_data)
        output_wf = out["result"][0]
        models = output_wf.get("models", {})

        self.assertEqual(set(models.keys()), {"model_a", "model_b", "model_c", "model_d"})
        self.assertEqual(models["model_a"]["model"], "ckpt_a")
        self.assertEqual(models["model_b"]["model"], "ckpt_b")
        self.assertEqual(models["model_c"]["model"], "ckpt_c")
        self.assertEqual(models["model_d"]["model"], "ckpt_d")

        # UI payload should also carry all slots so switching model view is consistent.
        extracted = out["ui"]["workflow_info"][0]["extracted"]
        slot_profiles = extracted.get("_slot_profiles", {})
        self.assertEqual(set(slot_profiles.keys()), {"model_a", "model_b", "model_c", "model_d"})

    def test_standalone_mode_creates_all_slots_and_applies_slot_profile(self):
        slot_profiles = {
            "model_a": _default_slot_profile(),
            "model_b": _default_slot_profile(),
            "model_c": _default_slot_profile(model_name="standalone_ckpt_c", pos="standalone_pos_c", neg="standalone_neg_c"),
            "model_d": _default_slot_profile(),
        }
        override_data = json.dumps({"_model_slot": "model_c", "_slot_profiles": slot_profiles})

        out = self._exec(recipe_data=None, override_data=override_data)
        models = out["result"][0].get("models", {})

        self.assertEqual(set(models.keys()), {"model_a", "model_b", "model_c", "model_d"})
        self.assertEqual(models["model_c"]["model"], "standalone_ckpt_c")
        self.assertEqual(models["model_c"]["positive_prompt"], "standalone_pos_c")
        self.assertEqual(models["model_c"]["negative_prompt"], "standalone_neg_c")

    def test_multi_prompt_overrides_recipe_when_unlocked(self):
        recipe = _make_base_recipe()
        slot_profiles = {
            "model_a": _default_slot_profile(model_name="ckpt_a", pos="slot_pos", neg="slot_neg"),
            "model_b": _default_slot_profile(),
            "model_c": _default_slot_profile(),
            "model_d": _default_slot_profile(),
        }
        override_data = json.dumps({"_model_slot": "model_a", "_slot_profiles": slot_profiles})

        out = self._exec(
            recipe_data=deepcopy(recipe),
            override_data=override_data,
            multi_pos_prompts={"model_a": "incoming_multi_pos"},
            multi_neg_prompts={"model_a": "incoming_multi_neg"},
        )
        model_a = out["result"][0]["models"]["model_a"]

        self.assertEqual(model_a["positive_prompt"], "incoming_multi_pos")
        self.assertEqual(model_a["negative_prompt"], "incoming_multi_neg")

    def test_locked_prompt_sections_ignore_connected_prompt_inputs(self):
        recipe = _make_base_recipe()
        profile_a = _default_slot_profile(model_name="ckpt_a", pos="locked_user_pos", neg="locked_user_neg")
        profile_a["ov"]["_section_locks"]["positive"] = True
        profile_a["ov"]["_section_locks"]["negative"] = True

        slot_profiles = {
            "model_a": profile_a,
            "model_b": _default_slot_profile(),
            "model_c": _default_slot_profile(),
            "model_d": _default_slot_profile(),
        }
        override_data = json.dumps({"_model_slot": "model_a", "_slot_profiles": slot_profiles})

        out = self._exec(
            recipe_data=deepcopy(recipe),
            override_data=override_data,
            multi_pos_prompts={"model_a": "incoming_multi_pos"},
            multi_neg_prompts={"model_a": "incoming_multi_neg"},
        )
        model_a = out["result"][0]["models"]["model_a"]

        self.assertEqual(model_a["positive_prompt"], "locked_user_pos")
        self.assertEqual(model_a["negative_prompt"], "locked_user_neg")

    def test_loras_are_additive_and_preserve_state_and_missing_flag(self):
        recipe = _make_base_recipe()
        recipe["models"]["model_a"]["loras"] = [
            {"name": "recipe_lora", "model_strength": 0.8, "clip_strength": 0.7, "active": True},
            {"name": "missing_lora", "model_strength": 1.0, "clip_strength": 1.0, "active": True},
        ]

        profile_a = _default_slot_profile(model_name="ckpt_a")
        profile_a["ov"]["loras_a"] = [
            {"name": "input_lora", "model_strength": 0.25, "clip_strength": 0.2, "active": False, "source_input": True},
            {"name": "recipe_lora", "model_strength": 0.9, "clip_strength": 0.85, "active": True},
        ]

        slot_profiles = {
            "model_a": profile_a,
            "model_b": _default_slot_profile(),
            "model_c": _default_slot_profile(),
            "model_d": _default_slot_profile(),
        }
        override_data = json.dumps({"_model_slot": "model_a", "_slot_profiles": slot_profiles})

        out = self._exec(
            recipe_data=deepcopy(recipe),
            override_data=override_data,
            multi_lora_stack={"model_a": [["input_lora", 0.75, 0.65]]},
        )

        model_a = out["result"][0]["models"]["model_a"]
        loras = model_a["loras"]
        by_name = {row.get("name"): row for row in loras}

        self.assertIn("recipe_lora", by_name)
        self.assertIn("input_lora", by_name)
        self.assertIn("missing_lora", by_name)

        self.assertFalse(by_name["input_lora"].get("active", True))
        self.assertEqual(float(by_name["input_lora"].get("model_strength", 0)), 0.25)
        self.assertEqual(float(by_name["input_lora"].get("clip_strength", 0)), 0.2)
        lora_names = [row.get("name") for row in loras]
        self.assertLess(lora_names.index("input_lora"), lora_names.index("recipe_lora"))

        extracted = out["ui"]["workflow_info"][0]["extracted"]
        lora_availability = extracted.get("lora_availability", {})
        self.assertIn("missing_lora", lora_availability)
        self.assertFalse(lora_availability["missing_lora"])

    def test_disconnected_multi_loras_behaves_as_empty_list(self):
        recipe = _make_base_recipe()
        profile_a = _default_slot_profile(model_name="ckpt_a")
        profile_a["ov"]["loras_a"] = [
            {"name": "stale_input_lora", "model_strength": 0.6, "clip_strength": 0.6, "active": True, "source_input": True},
            {"name": "recipe_lora", "model_strength": 0.8, "clip_strength": 0.7, "active": True},
        ]

        slot_profiles = {
            "model_a": profile_a,
            "model_b": _default_slot_profile(),
            "model_c": _default_slot_profile(),
            "model_d": _default_slot_profile(),
        }
        override_data = json.dumps({"_model_slot": "model_a", "_slot_profiles": slot_profiles})

        out = self._exec(recipe_data=deepcopy(recipe), override_data=override_data, multi_lora_stack={})
        loras = out["result"][0]["models"]["model_a"].get("loras", [])
        lora_names = [row.get("name") for row in loras]

        self.assertNotIn("stale_input_lora", lora_names)
        self.assertIn("recipe_lora", lora_names)

    def test_randomized_prompt_lock_contract(self):
        rng = random.Random(1337)
        slots = ("model_a", "model_b", "model_c", "model_d")

        def _tok(prefix):
            return f"{prefix}_{rng.randint(1000, 9999)}"

        for _ in range(40):
            recipe = {"version": 2, "models": {}}
            slot_profiles = {}
            multi_pos = {}
            multi_neg = {}
            expected_pos = {}
            expected_neg = {}

            for slot in slots:
                recipe_pos = _tok(f"rpos_{slot}")
                recipe_neg = _tok(f"rneg_{slot}")
                recipe_model = _tok(f"rmodel_{slot}")
                recipe["models"][slot] = _make_model_block(recipe_model, recipe_pos, recipe_neg)

                profile = _default_slot_profile(
                    model_name=recipe_model,
                    pos=_tok(f"upos_{slot}"),
                    neg=_tok(f"uneg_{slot}"),
                )
                lock_pos = bool(rng.getrandbits(1))
                lock_neg = bool(rng.getrandbits(1))
                profile["ov"]["_section_locks"]["positive"] = lock_pos
                profile["ov"]["_section_locks"]["negative"] = lock_neg
                slot_profiles[slot] = profile

                if bool(rng.getrandbits(1)):
                    multi_pos[slot] = _tok(f"inpos_{slot}")
                if bool(rng.getrandbits(1)):
                    multi_neg[slot] = _tok(f"inneg_{slot}")

                expected_pos[slot] = (
                    profile["ov"]["positive_prompt"]
                    if lock_pos
                    else multi_pos.get(slot, recipe_pos)
                )
                expected_neg[slot] = (
                    profile["ov"]["negative_prompt"]
                    if lock_neg
                    else multi_neg.get(slot, recipe_neg)
                )

            override_data = json.dumps({"_model_slot": "model_a", "_slot_profiles": slot_profiles})
            out = self._exec(
                recipe_data=deepcopy(recipe),
                override_data=override_data,
                multi_pos_prompts=multi_pos,
                multi_neg_prompts=multi_neg,
            )

            models = out["result"][0].get("models", {})
            self.assertEqual(set(models.keys()), set(slots))

            for slot in slots:
                self.assertEqual(models[slot].get("positive_prompt"), expected_pos[slot])
                self.assertEqual(models[slot].get("negative_prompt"), expected_neg[slot])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecipeBuilderContractTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
