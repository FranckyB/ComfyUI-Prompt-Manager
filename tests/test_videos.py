import sys, json, types, os, glob, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r'D:\ComfyUI-Dev')
mock_server = types.ModuleType('server')
mock_server.web = types.SimpleNamespace(json_response=lambda *a, **k: None)
class FakePromptServer:
    class instance:
        class routes:
            @staticmethod
            def post(path): return lambda f: f
            @staticmethod
            def get(path): return lambda f: f
mock_server.PromptServer = FakePromptServer
sys.modules['server'] = mock_server

sys.path.insert(0, r'D:\ComfyUI\custom_nodes\ComfyUI-Prompt-Manager')
from nodes.prompt_extractor import parse_workflow_for_prompts, extract_metadata_from_video, MODEL_LOADER_TYPES, resolve_model_path

vids_dir = r'D:\ComfyUI-Dev\input\Workflow\Vids'
ok_count = 0
no_model_count = 0
no_metadata_count = 0
error_count = 0

for f in sorted(glob.glob(os.path.join(vids_dir, '*.mp4'))):
    basename = os.path.basename(f)
    try:
        result = extract_metadata_from_video(f)
        if not result or (not result[0] and not result[1]):
            no_metadata_count += 1
            continue
        
        prompt_str, workflow_str = result
        prompt_data = json.loads(prompt_str) if isinstance(prompt_str, str) else (prompt_str or {})
        workflow_data = json.loads(workflow_str) if isinstance(workflow_str, str) else (workflow_str or {})
        
        if not prompt_data and not workflow_data:
            no_metadata_count += 1
            continue

        parsed = parse_workflow_for_prompts(prompt_data, workflow_data)
        models_a = parsed.get('models_a', [])
        models_b = parsed.get('models_b', [])

        if models_a or models_b:
            ok_count += 1
            a_resolved = resolve_model_path(models_a[0])[0] if models_a else ""
            b_resolved = resolve_model_path(models_b[0])[0] if models_b else ""
            print(f"OK: {basename} -> A={a_resolved}, B={b_resolved}")
        else:
            no_model_count += 1
            # Check what loader types exist
            all_types = set()
            loader_types = set()
            if prompt_data:
                for ndata in prompt_data.values():
                    if isinstance(ndata, dict):
                        ct = ndata.get('class_type', '')
                        all_types.add(ct)
                        ct_lower = ct.lower()
                        if any(k in ct_lower for k in ['checkpoint', 'unet', 'diffusion', 'model', 'loader']):
                            loader_types.add(ct)
            if workflow_data and 'nodes' in workflow_data:
                for node in workflow_data['nodes']:
                    if isinstance(node, dict):
                        nt = node.get('type', '')
                        all_types.add(nt)
                        nt_lower = nt.lower()
                        if any(k in nt_lower for k in ['checkpoint', 'unet', 'diffusion', 'model', 'loader']):
                            loader_types.add(nt)
            print(f"NO MODELS: {basename}")
            if loader_types:
                print(f"  Loader-like types: {sorted(loader_types)}")
            else:
                print(f"  ALL types: {sorted(all_types)}")
    except Exception as e:
        error_count += 1
        print(f"ERROR: {basename}: {type(e).__name__}: {e}")

print(f"\n=== SUMMARY: {ok_count} OK, {no_model_count} no models, {no_metadata_count} no metadata, {error_count} errors ===")
