# ComfyUI Prompt Manager

A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows both saving and generating new prompts using llama.cpp.

(**Llama.cpp needs to be installed and accessible from command prompt.**)

<div align="center">
  <img src="docs/prompt_manager.png" alt="Prompt Manager">
</div>

### Key features:
### Prompt Manager:
- **Category Organization**: Create and manage multiple categories to organize your prompts
- **Save & Load Prompts**: Quickly save and recall your favorite prompts with custom names
- **LLM Input Toggle**: Connect text outputs from other nodes and toggle between using them or your internal prompts
- **LLM Input Toggle**: When in use, display of categories and prompt is disabled, allowing user to switch category and save.
- **Persistent Storage**: All prompts saved in your ComfyUI user folder

### Prompt Generator
- **Automatic llama.cpp Server Management**: Automatically starts llama.cpp server as needed
- **Server Control**: Toggle to stop server after generation (frees VRAM)
- **AI-Powered Prompt Enhancement**: Transform basic prompts into detailed, vivid descriptions optimized for image generation
- **Local Model Support**: Automatically detects and uses first .gguf model found in the models/guff folder.

### Prompt Generator Options
- **Model Selection**: Choose from local models or download from HuggingFace
- **Model Auto-Download**: Easy download of Qwen3 1.7B, 4B, or 8B from HuggingFace.
- **LLM Parameters**: Fine-tune temperature, top_k, top_p, min_p, repeat_penalty
- **Custom Instructions**: Override default system prompt for different enhancement styles

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/FranckyB/ComfyUI-Prompt-Manager.git
   ```

3. Install dependencies:
   ```bash
   cd ComfyUI-Prompt-Manager
   pip install -r requirements.txt
   ```

4. Install llama.cpp. It can be found here: [llama.cpp install](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)

5. If you have them, place your .gguf models in the models/gguf folder. (Create folder if not present)


6. Restart ComfyUI

## Usage

### Prompt Manager

1. **Add the Node**: Add Node → Prompt Manager
2. **Select a Category**: Use the dropdown to choose from your categories
3. **Choose a Prompt**: Select a saved prompt from the name dropdown
4. **Connect prompt output**: Connect Prompt Manager output to your clip text encode node.


### Prompt Generator

**Basic Usage** (assuming a model is present in models\gguf):
1. **Add the Node**: Add Node → Prompt Generator
2. **Connect llm_output to prompt manager**: If you want to save your generated prompt connect to prompt manager.
3. **type your prompt**: Type in your basic prompt that will be embelished
4. **Save memory**: Toggle "stop_server_after" ON to free VRAM after generation, if seed is set to fix, it won't restart unless prompt changes.
4. **Run Workflow**: Once calculated, the prompt will display in the Prompt Manager window and allow you to save it.

**Advanced Usage**:
1. **Add the Options Node**: Add Node → Prompt Generator Options
2. **Connect Options**: Connect the options node to the Prompt Generator Options input
3. **Select from available models**: Select from models found in your modeld\gguf folder. Qwen models will be available to download.
4. **Adjust settings**: Adjust LLM parameters (temperature, top_k, etc.)
5. **Customize LLM**: Customize the default LLM instructions to modify the responses llama returns.

**Size of Qwen models found in options**
- Qwen3-1.7B-Q8_0.gguf: Fastest, lowest VRAM (~2GB)
- Qwen3-4B-Q8_0.gguf:   Balanced performance (~4GB VRAM)
- Qwen3-8B-Q8_0.gguf:   Best quality, highest VRAM (~8GB)

**Model Management**:
- Place gguf files in models/gguf folder
- Downloaded models are also placed in this folder.

## Requirements

- ComfyUI
- Python 3.8+
- requests >= 2.31.0
- huggingface_hub >= 0.20.0
- llama-server (from llama.cpp)

## Troubleshooting

**Problem**: Prompts don't appear in the dropdown
- **Solution**: Make sure the category has saved prompts. Try creating a new prompt first.

**Problem**: Changes aren't saved
- **Solution**: Click the "Save Prompt" button after making changes. Direct edits in the text field are temporary.

**Problem**: Can't see LLM output in the node
- **Solution**: Make sure the LLM output is connected to the "llm_input" and run the workflow.

**Problem**: "llama-server command not found"
- **Solution**: Install llama.cpp and make sure `llama-server` is available in command line. See [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)

**Problem**: "No models found"
- **Solution**: Either place a .gguf file in the `models/` folder, or connect the Prompt Generator Option node and select a model size (1.7B, 4B, or 8B) to download from HuggingFace

**Problem**: Server won't start
- **Solution**: Check that port 8080 is not in use. Close any existing llama-server processes.

**Problem**: Model download fails
- **Solution**: Check your internet connection and HuggingFace availability. Large models may take time to download.

**Problem**: Generation is slow
- **Solution**: Either, Use a smaller quantized model (Q4 instead of Q8) or toggle 'stop_server_after' to quit llama.cpp after generating prompt.

## Changelog

### Version 1.5.1
- LLM output remains avaiable when use llm is off. So it can be edited
- Improved Caching detection, any change to options will be detected and force a new ouput
- Improved some UI quirks

### Version 1.5.0
- Added Prompt Generator node with automatic llama.cpp server management
- Added Prompt Generator Options node for model selection and parameters
- Automatic model detection and auto-download from HuggingFace for Qwen3 models.
- VRAM management with optional server shutdown

### Version 1.1.0
- Added LLM input toggle for switching between internal and external text
- Made text fields scrollable even when disabled
- Fixed reload bugs with toggle state

### Version 1.0.0
- Initial release
- Category and prompt management
- LLM output integration
