import ast
import inspect
from io import BytesIO
import logging
import os
from pathlib import Path
import sys
import types
import traceback
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "nodes" / "prompt_generator.py"


class ModelSelected(Exception):
    pass


class PromptGeneratorModelTests(unittest.TestCase):
    def setUp(self):
        module = ast.parse(SOURCE.read_text(encoding="utf-8"))
        node_class = next(
            node for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "PromptGenerator"
        )
        methods = {
            "VALIDATE_INPUTS", "resolve_preferred_model", "find_text_model",
            "find_vision_model", "convert_prompt", "start_server",
        }
        node_class.body = [
            node for node in node_class.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        self.models = ["default.gguf", "selected.gguf", "vision.gguf"]
        self.preferences = {"preferred_model": "default.gguf"}
        self.logs = []
        namespace = {
            "__package__": "model_tests.nodes",
            "os": os,
            "sys": sys,
            "logging": logging,
            "_preferences_cache": self.preferences,
            "_server_process": None,
            "_current_model": None,
            "_current_context_size": None,
            "_current_gpu_device": None,
            "get_local_models": lambda: self.models,
            "is_model_local": lambda name: name in self.models,
            "has_vision_support": lambda name: name == "vision.gguf",
            "get_default_context_size": lambda: 2048,
            "print_pg_header": lambda: None,
            "print_pg": lambda *args: self.logs.append(" ".join(map(str, args))),
            "RED": "",
            "YELLOW": "",
        }
        helpers = [
            node for node in module.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {
                "PromptGeneratorStartupError", "_PromptGeneratorStartupLogFilter",
                "_install_prompt_generator_startup_log_filter",
            }
        ]
        exec(compile(ast.Module(body=[*helpers, node_class], type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.namespace = namespace
        self.generator = namespace["PromptGenerator"]
        self.generator.get_system_prompt_entry = staticmethod(
            lambda value: {"category": "Image", "name": "Test", "prompt": "instruction"}
        )
        self.generator.stop_server = staticmethod(lambda: None)
        self.generator.get_server_port = staticmethod(lambda: 1234)
        self.real_start_server = self.generator.start_server
        self.generator.start_server = staticmethod(self.capture_model)
        self.generator._generate_via_ollama = staticmethod(
            lambda **kwargs: self.capture_model(kwargs["model_to_use"])
        )
        self.ollama_models = ["default:latest", "selected:latest"]
        ollama = types.ModuleType("model_tests.py.ollama_wrapper")
        ollama.discover_ollama_models = lambda preferences: (self.ollama_models, "test discovery")
        self.module_patch = patch.dict(sys.modules, {ollama.__name__: ollama})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    @staticmethod
    def capture_model(model, *args, **kwargs):
        raise ModelSelected(model)

    def selected_model(self, **kwargs):
        with self.assertRaises(ModelSelected) as result:
            self.generator().convert_prompt(seed=0, prompt="test", clear_vram_on_run=False, **kwargs)
        return result.exception.args[0]

    def assert_fallback_logged(self, name):
        self.assertTrue(any(name in line and "falling back" in line for line in self.logs), self.logs)

    def test_validation_exempts_only_model_and_accepts_stale_names(self):
        self.assertEqual(list(inspect.signature(self.generator.VALIDATE_INPUTS).parameters), ["model"])
        self.assertIs(self.generator.VALIDATE_INPUTS("deleted.gguf"), True)
        self.assertIs(self.generator.VALIDATE_INPUTS(), True)
        self.assertIsNot(self.generator.VALIDATE_INPUTS(123), True)

    def test_missing_local_model_uses_preferred_and_logs(self):
        self.assertEqual(self.selected_model(model="deleted.gguf"), "default.gguf")
        self.assert_fallback_logged("deleted.gguf")

    def test_valid_selection_overrides_default_without_warning(self):
        self.assertEqual(self.selected_model(model="selected.gguf"), "selected.gguf")
        self.assertFalse(any("falling back" in line for line in self.logs))

    def test_missing_legacy_options_model_uses_default_without_mutating_options(self):
        options = {"model": "deleted.gguf", "temperature": 0.7}
        self.assertEqual(self.selected_model(options=options), "default.gguf")
        self.assertEqual(options, {"model": "deleted.gguf", "temperature": 0.7})
        self.assert_fallback_logged("deleted.gguf")

    def test_missing_preferred_model_uses_available_model(self):
        self.preferences["preferred_model"] = "also-deleted.gguf"
        self.assertEqual(self.selected_model(model="deleted.gguf"), "default.gguf")

    def test_image_fallback_requires_vision_model(self):
        self.assertEqual(self.selected_model(model="deleted.gguf", image=object()), "vision.gguf")
        self.assert_fallback_logged("deleted.gguf")

    def test_no_models_still_reports_actionable_error(self):
        self.models.clear()
        with self.assertRaisesRegex(RuntimeError, "No models found"):
            self.selected_model(model="deleted.gguf")

    def test_missing_ollama_model_uses_preferred_and_logs(self):
        self.preferences.update(llm_backend="ollama", preferred_model="default:latest")
        self.assertEqual(self.selected_model(model="deleted:latest"), "default:latest")
        self.assert_fallback_logged("deleted:latest")

    def test_ollama_untagged_alias_is_preserved(self):
        self.preferences.update(llm_backend="ollama", preferred_model="default:latest")
        self.assertEqual(self.selected_model(model="selected"), "selected")
        self.assertFalse(any("falling back" in line for line in self.logs))

    def test_ollama_missing_preferred_uses_discovered_model(self):
        self.preferences.update(llm_backend="ollama", preferred_model="also-deleted:latest")
        self.assertEqual(self.selected_model(model="deleted:latest"), "default:latest")

    def test_no_ollama_models_still_reports_actionable_error(self):
        self.preferences["llm_backend"] = "ollama"
        self.ollama_models.clear()
        with self.assertRaisesRegex(RuntimeError, "No Ollama model available"):
            self.selected_model(model="deleted:latest")

    def test_clip_generation_does_not_validate_external_models(self):
        self.generator._generate_via_clip = staticmethod(lambda **kwargs: ("clip result", ""))
        result = self.generator().convert_prompt(
            seed=0, prompt="test", model="deleted.gguf", clip=object(), clear_vram_on_run=False,
        )
        self.assertEqual(result, ("clip result", ""))
        self.assertFalse(any("falling back" in line for line in self.logs))

    def simulate_startup(self, errors, clear_vram=False, use_gpu=True, show_console=False):
        self.preferences["close_llama_on_exit"] = False
        processes = [
            types.SimpleNamespace(
                returncode=1, poll=lambda: 1, stderr=BytesIO(error.encode("utf-8")),
            )
            for error in errors
        ]
        self.debug_logger = Mock()
        self.popen = Mock(side_effect=processes)
        self.namespace.update({
            "get_model_path": lambda name: str(SOURCE),
            "resolve_llama_server_command": lambda path: "/usr/local/bin/llama-server",
            "detect_llama_gpu_support": lambda command: (use_gpu, ["CUDA"] if use_gpu else [], None),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: use_gpu)),
            "comfy": types.SimpleNamespace(model_management=types.SimpleNamespace(
                throw_exception_if_processing_interrupted=lambda: None,
            )),
            "subprocess": types.SimpleNamespace(Popen=self.popen, PIPE=-1),
            "time": types.SimpleNamespace(sleep=lambda seconds: None),
            "logging": self.debug_logger,
        })
        self.generator.kill_all_llama_servers = staticmethod(lambda: None)
        return self.real_start_server("selected.gguf", clear_vram_on_run=clear_vram, show_everything_in_console=show_console)

    def test_startup_failure_keeps_vram_hint_and_stderr_only_in_debug(self):
        stderr = (
            "common_init_: failed to create context with model '/models/selected.gguf'\n"
            "load_model: failed to create_context with model '/models/selected.gguf'\n"
            "llama_server: exiting due to model loading error"
        )
        success, message = self.simulate_startup([stderr])
        self.assertFalse(success)
        self.assertIn("Try enabling 'Clear VRAM on run'", message)
        self.assertNotIn("\n", message)
        self.assertNotIn("common_init_", message)
        self.assertIn("Show everything in Console", message)
        self.debug_logger.debug.assert_any_call("[PromptGenerator] llama-server stderr:\n%s", stderr)
        self.assertFalse(any("common_init_" in line or "could not load" in line for line in self.logs))
        self.generator.start_server = staticmethod(lambda *args, **kwargs: (success, message))
        with self.assertRaisesRegex(RuntimeError, "Clear VRAM on run") as error:
            self.generator().convert_prompt(seed=0, prompt="test", clear_vram_on_run=False)
        self.assertEqual(str(error.exception), message)

    def test_startup_hint_does_not_suggest_enabling_an_enabled_setting_or_cpu_vram(self):
        for clear_vram, use_gpu in [(True, True), (False, False)]:
            with self.subTest(clear_vram=clear_vram, use_gpu=use_gpu):
                success, message = self.simulate_startup(["failed to allocate memory"], clear_vram, use_gpu)
                self.assertFalse(success)
                self.assertNotIn("Clear VRAM", message)
                self.assertIn("smaller model", message)

    def test_compatibility_retry_failure_retains_actionable_hint(self):
        success, message = self.simulate_startup(["unknown argument: --flash-attn", "CUDA error: out of memory"])
        self.assertFalse(success)
        self.assertEqual(self.popen.call_count, 2)
        self.assertIn("Clear VRAM on run", message)
        self.assertNotIn("CUDA error", message)

    def test_unrecognized_startup_failure_reports_exit_code_not_raw_stderr(self):
        success, message = self.simulate_startup(["unexpected internal server error\nverbose details"])
        self.assertFalse(success)
        self.assertIn("exited with code 1", message)
        self.assertIn("Show everything in Console", message)
        self.assertNotIn("verbose details", message)
        self.assertNotIn("\n", message)

    def log_failure(self, error, level=logging.INFO, filename="execution.py", function="execute", handler=None):
        log_namespace = {
            "logging": logging, "traceback": traceback, "sys": sys,
            "full_type_name": lambda exception_type: exception_type.__name__,
            "input_data_all": None, "class_type": "PromptGenerator", "real_node_id": "generator-test",
            "comfy": types.SimpleNamespace(model_management=types.SimpleNamespace(is_oom=lambda error: False)),
            "ExecutionResult": types.SimpleNamespace(FAILURE="failure"),
        }
        log_function = ast.parse(
            f"def {function}(error):\n"
            "    try:\n"
            "        raise error\n"
            "    except Exception as ex:\n"
            "        logging.error(f'!!! Exception during processing !!! {ex}')\n"
            "        logging.error(traceback.format_exc())\n"
            "        return ex\n",
        )
        if handler is not None:
            log_function.body[0].body[0].handlers = [handler]
        exec(compile(ast.fix_missing_locations(log_function), filename, "exec"), log_namespace)
        records = []
        log_handler = logging.Handler()
        log_handler.emit = records.append
        root = logging.getLogger()
        with patch.object(root, "handlers", [log_handler]), patch.object(root, "filters", []):
            previous_level = root.level
            root.setLevel(level)
            try:
                install_filter = self.namespace["_install_prompt_generator_startup_log_filter"]
                install_filter()
                install_filter()
                self.assertEqual(len(root.filters), 1)
                returned = log_namespace[function](error)
                if handler is None:
                    self.assertIs(returned, error)
                else:
                    self.assertEqual(returned[0], "failure")
                    self.assertIs(returned[2], error)
                    self.assertIn(str(error), returned[1]["exception_message"])
                    self.assertTrue(returned[1]["traceback"])
            finally:
                root.setLevel(previous_level)
        return [record.getMessage() for record in records]

    def test_expected_startup_failure_logs_only_one_summary(self):
        error = self.namespace["PromptGeneratorStartupError"]("Try enabling 'Clear VRAM on run'.")
        messages = self.log_failure(error)
        self.assertEqual(messages, [f"!!! Exception during processing !!! {error}"])

    def test_debug_keeps_expected_startup_traceback(self):
        error = self.namespace["PromptGeneratorStartupError"]("Model load failed")
        messages = self.log_failure(error, level=logging.DEBUG)
        self.assertEqual(len(messages), 2)
        self.assertIn("Traceback (most recent call last):", messages[1])

    def test_console_toggle_controls_full_startup_output_and_traceback_per_run(self):
        stderr = "CUDA error: out of memory\nfull server diagnostic details"
        self.generator.start_server = staticmethod(
            lambda *args, **kwargs: self.simulate_startup(
                [stderr], show_console=kwargs["show_everything_in_console"],
            )
        )
        for enabled in [True, False]:
            with self.subTest(enabled=enabled):
                self.logs.clear()
                with self.assertRaises(self.namespace["PromptGeneratorStartupError"]) as result:
                    self.generator().convert_prompt(
                        seed=0, prompt="test", clear_vram_on_run=False,
                        options={"show_everything_in_console": enabled},
                    )
                self.assertEqual(result.exception.show_everything_in_console, enabled)
                self.assertEqual(any(stderr in line for line in self.logs), enabled)
                self.assertEqual(any("--ctx-size" in line for line in self.logs), enabled)
                self.namespace["logging"] = logging
                messages = self.log_failure(result.exception)
                self.assertEqual(len(messages), 2 if enabled else 1)
                self.assertEqual(any("Traceback (most recent call last):" in message for message in messages), enabled)

    def test_console_toggle_keeps_full_output_for_compatibility_retry(self):
        errors = ["unknown argument: --flash-attn", "CUDA error: out of memory"]
        success, message = self.simulate_startup(errors, show_console=True)
        self.assertFalse(success)
        self.assertEqual(self.popen.call_count, 2)
        for stderr in errors:
            self.assertTrue(any(stderr in line for line in self.logs))
        self.assertEqual(sum(line.startswith("Launching:") for line in self.logs), 2)
        self.assertNotIn("Enable 'Show everything in Console'", message)

    def test_other_errors_keep_tracebacks_even_with_identical_messages(self):
        for error in [RuntimeError("Prompt Generator: Model load failed"), ValueError("unexpected bug")]:
            with self.subTest(error=error):
                messages = self.log_failure(error)
                self.assertEqual(len(messages), 2)
                self.assertIn("Traceback (most recent call last):", messages[1])

    def test_startup_tracebacks_from_other_log_call_sites_are_untouched(self):
        error = self.namespace["PromptGeneratorStartupError"]("Model load failed")
        for filename, function in [("another_node.py", "execute"), ("execution.py", "other_function")]:
            with self.subTest(filename=filename, function=function):
                messages = self.log_failure(error, filename=filename, function=function)
                self.assertEqual(len(messages), 2)

    def test_plain_traceback_text_outside_an_exception_is_untouched(self):
        log_filter = self.namespace["_PromptGeneratorStartupLogFilter"]()
        record = logging.LogRecord(
            "root", logging.ERROR, "execution.py", 1,
            "Traceback (most recent call last):", (), None, func="execute",
        )
        self.assertTrue(log_filter.filter(record))

    def test_installed_comfyui_handler_respects_console_toggle_and_preserves_failure(self):
        execution_file = SOURCE.parents[3] / "execution.py"
        if not execution_file.is_file():
            self.skipTest("ComfyUI execution.py is not installed alongside this repository")
        module = ast.parse(execution_file.read_text(encoding="utf-8"))
        execute = next(node for node in module.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute")
        handler = next(
            node for node in ast.walk(execute)
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name)
            and node.type.id == "Exception" and node.name == "ex"
        )
        for enabled in [False, True]:
            with self.subTest(enabled=enabled):
                error = self.namespace["PromptGeneratorStartupError"](
                    "Try enabling 'Clear VRAM on run'.", show_everything_in_console=enabled,
                )
                messages = self.log_failure(error, filename=str(execution_file), handler=handler)
                self.assertEqual(messages[0], f"!!! Exception during processing !!! {error}")
                self.assertEqual(len(messages), 2 if enabled else 1)
                if enabled:
                    self.assertIn("Traceback (most recent call last):", messages[1])


if __name__ == "__main__":
    unittest.main()