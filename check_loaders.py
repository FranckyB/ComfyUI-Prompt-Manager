import sys, json, subprocess, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Check videos with missing loader types
test_files = {
    'DiffusionModelLoaderKJ': r'D:\ComfyUI-Dev\input\Workflow\Vids\2026_01_08_60FPS_00002.mp4',
    'WanVideoModelLoader': r'D:\ComfyUI-Dev\input\Workflow\Vids\vid_00061.mp4',
    'SeaArtUnetLoader': r'D:\ComfyUI-Dev\input\Workflow\Vids\25439d30-d0d7-4dd4-b64c-b46345c76351.mp4',
    'CyberdyneModelHub': r'D:\ComfyUI-Dev\input\Workflow\Vids\d39dbdc3-f08c-44e1-bd4c-410ecb08d3c9.mp4', 
    'CheckpointInNum': r'D:\ComfyUI-Dev\input\Workflow\Vids\num__00168.mp4',
}

for label, f in test_files.items():
    print(f"\n=== {label} ===")
    r = subprocess.run(['ffprobe','-v','quiet','-print_format','json','-show_format',f], capture_output=True, text=True, timeout=10)
    data = json.loads(r.stdout)
    tags = data['format']['tags']
    
    # Try direct tags first, then comment
    prompt = tags.get('prompt')
    if not prompt:
        comment = tags.get('comment', '{}')
        comment_data = json.loads(comment)
        prompt = comment_data.get('prompt', '{}')
    
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    
    targets = ['Diffusion', 'Wan', 'SeaArt', 'Cyberdyne', 'Checkpoint', 'UNET', 'Unet', 'Model']
    for nid, ndata in prompt.items():
        ct = ndata.get('class_type', '')
        if 'Cyberdyne' in ct or ct == 'Wan_Cyberdyne_Genisys':
            inputs = ndata.get('inputs', {})
            inp_str = json.dumps(inputs)
            if len(inp_str) > 800:
                inp_str = inp_str[:800] + '...'
            print(f"  Node {nid}: {ct}")
            print(f"    inputs: {inp_str}")
