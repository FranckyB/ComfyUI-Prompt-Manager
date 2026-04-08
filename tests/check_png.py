import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from PIL import Image

img = Image.open(r'D:\ComfyUI-Dev\output\ComfyUI_00006_.png')
print('=== Keys in info ===')
for k in img.info:
    print(f'  {k}: {len(str(img.info[k]))} chars')

# Check workflow for extracted_data
wf = json.loads(img.info.get('workflow', '{}'))
for node in wf.get('nodes', []):
    ntype = node.get('type', '')
    nid = node.get('id', '?')
    if ntype == 'PromptExtractor':
        print(f'\n=== PromptExtractor node {nid} ===')
        ext = node.get('extracted_data')
        if ext:
            print('extracted_data FOUND:')
            print(json.dumps(ext, indent=2, ensure_ascii=False)[:2000])
        else:
            print('extracted_data: MISSING')
            print('Node keys:', list(node.keys()))
            # Show widgets_values
            wv = node.get('widgets_values', [])
            print(f'widgets_values ({len(wv)} items): {str(wv)[:500]}')

# Also check prompt (API format) for PromptExtractor
prompt = json.loads(img.info.get('prompt', '{}'))
for nid, ndata in prompt.items():
    if isinstance(ndata, dict) and ndata.get('class_type') == 'PromptExtractor':
        print(f'\n=== API PromptExtractor node {nid} ===')
        inputs = ndata.get('inputs', {})
        print('Input keys:', list(inputs.keys()))
        for k in ['loras_a_toggle', 'loras_b_toggle', 'lora_stack_a', 'lora_stack_b']:
            if k in inputs:
                val = inputs[k]
                print(f'  {k}: {str(val)[:300]}')
