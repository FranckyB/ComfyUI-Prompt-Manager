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
from nodes.prompt_extractor import parse_workflow_for_prompts, MODEL_LOADER_TYPES, parse_a1111_parameters
from PIL import Image

pics_dir = r'D:\ComfyUI-Dev\input\Workflow\Pics'
no_model_count = 0
ok_count = 0

for f in sorted(glob.glob(os.path.join(pics_dir, '*.png'))):
    try:
        img = Image.open(f)
        prompt_str = img.info.get('prompt', '')
        workflow_str = img.info.get('workflow', '')
        parameters_str = img.info.get('parameters', '')
        if not prompt_str and not workflow_str and not parameters_str:
            continue
        
        # Handle A1111 parameters format
        if not prompt_str and not workflow_str and parameters_str:
            parsed_a1111 = parse_a1111_parameters(parameters_str)
            prompt_data = parsed_a1111 if parsed_a1111 else {}
            workflow_data = {}
        else:
            prompt_data = json.loads(prompt_str) if prompt_str else {}
            workflow_data = json.loads(workflow_str) if workflow_str else {}
        
        # Check what model loaders exist
        api_loaders = []
        if prompt_data:
            for nid, ndata in prompt_data.items():
                if isinstance(ndata, dict):
                    ct = ndata.get('class_type', '')
                    if ct in MODEL_LOADER_TYPES:
                        api_loaders.append((nid, ct))
        
        wf_loaders = []
        wf_loader_types = set()
        if workflow_data and 'nodes' in workflow_data:
            for node in workflow_data['nodes']:
                if isinstance(node, dict):
                    ntype = node.get('type', '')
                    if ntype in MODEL_LOADER_TYPES:
                        wid = node.get('widgets_values', [])
                        wf_loaders.append((node.get('id'), ntype, wid[:2] if wid else []))
                    # Also check for checkpoint-like types not in our list
                    nt_lower = ntype.lower()
                    if any(k in nt_lower for k in ['checkpoint', 'unet', 'diffusion', 'model_loader']):
                        wf_loader_types.add(ntype)
        
        parsed = parse_workflow_for_prompts(prompt_data, workflow_data)
        models_a = parsed.get('models_a', [])
        models_b = parsed.get('models_b', [])
        
        basename = os.path.basename(f)
        if not models_a and not models_b:
            no_model_count += 1
            print(f"\nNO MODELS: {basename}")
            if api_loaders:
                print(f"  API loaders found: {api_loaders}")
            if wf_loaders:
                print(f"  WF loaders found: {wf_loaders}")
            if wf_loader_types:
                print(f"  Checkpoint-like types in WF: {wf_loader_types}")
            if not api_loaders and not wf_loaders and not wf_loader_types:
                all_types = set()
                if prompt_data:
                    for ndata in prompt_data.values():
                        if isinstance(ndata, dict):
                            all_types.add(ndata.get('class_type', ''))
                if workflow_data and 'nodes' in workflow_data:
                    for node in workflow_data['nodes']:
                        if isinstance(node, dict):
                            all_types.add(node.get('type', ''))
                print(f"  ALL types: {sorted(all_types)}")
        else:
            ok_count += 1
    except Exception as e:
        print(f"ERROR: {os.path.basename(f)}: {e}")

print(f"\n=== SUMMARY: {ok_count} OK, {no_model_count} no models ===")
