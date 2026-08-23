**This documentation is ongoing and will continue to improve**

### Prompt Manager
Allows for Saving and Loading prompts quickly using the Prompt Browser UI.
- Clicking the Blue Prompt Name button, under the text area, will open the Prompt Browser.
- Pressing the < > on each side of the button will quickly switch to the next prompt.
- **In Prompt Browser mode:**
  - Click the + to create new Categories.
  - Generate Thumbnails for Prompts or Categories (Right Click)
    - If not set, a Pop Up will ask you to select the model to use for generation.
  - Enable Edit mode to enable writing Prompts directly in the Browser.
    - Be sure to save the prompt before generating its thumbnail
  - Right click provides other options such as:
    - Tagging as NSFW (filtering by default can be enabled in prefs)
    - Moving or Renaming
    - Pasting an image in as a Prompt
    - Generating a thumbnail for a Prompt
  - It is also possible to simply drag an image over a prompt to replace its Thumbnail.
- **On the node's Main UI:**
- Use Prompt input toggle to display incoming prompt (from Prompt Generator for example).
    - Toggling off allows editing that new prompt.
- Use LoRA input allows choosing which LoRAs are used:
    - Prompt LoRAs Only: This turns off any LoRAs connected to our node.
    - Combine LoRAs: This uses both the prompt and incoming LoRAs, but will remove duplicates.
    - Input LoRAs Only: For those times when you want just the Prompt but use your own new recipe.
    - **Do note: If a prompt is saved again, this can change the LoRAs assigned to the Prompt**
- LoRA Support:
    - Supports the Add-On LoRA Manager to display LoRA Thumbnails on Hover.
    - LoRA input supports either Traditional LoRA Stackers or the new included Multi LoRA Stack.
    - Using normal LoRA Stackers, two stacks of LoRAs can be used and will be displayed in our UI.
    - Connecting Multi LoRA Stack instead allows up to four stacks to be displayed.
        - Multi LoRA Stack uses the LoRA Manager add-on at its core and is needed for it to work.
    - When LoRAs are used, the UI will display them as tags, allowing strength editing or toggling On or Off.
    - A list of Trigger words can be connected and will be displayed as Tags that can be toggled On or Off.
    - LoRAs are saved with prompt entries; this includes their state (On or Off) as well as their Strength.
    - Missing LoRAs will be displayed as Red.
- The Save Button opens up the Prompt Browser, allowing saving under any category.
- The New Prompt Button is practical when you want to clear everything, including LoRAs, and start fresh.
- The More button allows access to Exporting your Saved Prompt, but also merging another JSON. Be careful.

### Prompt Manager (Basic)
A simpler no frills version that supports only Prompts. (This is the OG node)
- Select category and prompt using a purely comfy UI.
- If saving over an existing prompt, LoRAs and Thumbnails will be preserved.
- If NSFW filtering is enabled, NSFW Prompts will be hidden.
- LLM input toggle works similar to full node.


### Prompt Generator
Generates prompts using a local LLM. Supports either Llama.cpp or Ollama, but can also be used directly with Comfy's Text encoders.
Allows downloading of 3 different sizes of Unsloth's Qwen 3.5, but users are free to add more models in models/gguf. If found, the add-on will list them. (Refresh when adding new ones)

- Comes with a bundle of pre-made System Prompts. 
- Create your own system prompts, using the **Prompt Browser** node
- I highly recommend using the **rgthree add-on** and enable *"Auto Nest Subdirectories in Menus"*. This will display categories as submenus.
- Options to output in JSON format and enable Thinking mode.
- The stop_server_after toggle:
  - Kills Llama.cpp after generating the prompt
  - Tries to force Ollama to unload the current model
- The clear_vram_on_run toggle:
  - Unloads Comfy's model from memory before generating the prompt
  - Clears Vram again once the prompt is generated.
  - It is recommended to leave on, unless you are using small models
- Both Stop_server and clear_vram are ignored when using Text encoders, as we instead let Comfy manage the VRAM.
- Lastly, the **Model Selection**:
  - (Use Default): Uses the default model set in preferences, or the first found, if not set.
  - 3 Different Quant size of Qwen 3.5 can be automatically downloaded.
  - Any models you add in models/gguf, or any custom folder if set in preferences.
- For Multi-image support, connect the Generator Option node.
- Use the Prompt Generator Options node for controlling model parameters.

### Prompt Generator Options
This node provides extra control to **Prompt Generator** and is NOT mandatory.  
- Modify the selected system prompt, by either appending extra instructions or replacing it entirely.
- Provides extra images input for a Total of 5.
- Change the System settings for the model
- Change the GPU used by the LLM

### Prompt Browser
Prompt Browser lets you create and edit all 3 prompt libraries from a single interface:
- System prompts: The System Prompts used by Prompt Generator.
- Compose prompts: Snippets used by Prompt Composer
- Prompt Manager prompts: Our saved workflow prompts

Prompt Generator now uses a *prompt_generator_data.json*, so you can author your own prompts instead of being limited to the previous simple presets. The add-on also comes with new default prompt, these can be imported back into your saved user system prompts by using the **More** button in the Prompt Browser Node.  

New Prompts may be added from time to time, so this is an easy way to import them back in.


### Prompt Extractor
Extract prompt, model, and LoRA info out of images, videos, and JSON workflow files.
- Supports ComfyUI, A1111/Forge, and WebP metadata formats.
- Grabs the frame of your choice from videos using a time slider.
- Dual LoRA stack extraction for compatible workflows.
- Can browse from either the input or output folders directly.
- Can also be used as an advanced media loader with metadata awareness

## Recipe Toolset (Experimental)

The Recipe Toolset are a set of experimental nodes to lets you build, manage, and rerun reusable pipelines. You can write recipes from scratch in Builder, combine them with Prompt Manager / Prompt Manager Advanced to reuse saved prompts, or import existing metadata through Extractor when you want to bootstrap from something you already made. Then saved them as complete Recipes.

As Stated these were an experiment into improving Manager, but as been put aside a bit.  The technology behind it is what allowed for Thumbnail Generation in Prompt Browser.  The Recipe Relay node is great when creating slick workflows with a single link between subgraph, as it can stack 4 workflows of data in a single connection.

### Recipe Builder
The main place to create or tweak recipes.
- Build or modify recipes from scratch in one central spot.
- Inspect and adjust recipe fields without wiring every parameter by hand.
- Use it standalone, or connect it with other recipe/prompt nodes.
- Works well with Prompt Manager so saved prompts can be slotted into recipe-driven pipelines.
- Great for iterative tuning: extract once, keep tweaking values and rerendering until the pipeline feels right.
- Helps standardize presets so repeated jobs start from the same known-good setup.

### Recipe Extractor
Reads and normalizes workflow metadata into reusable recipe data.
- Converts raw generation metadata into a cleaner recipe format that's easier to edit and pass between nodes.
- Keeps important settings intact when moving from one-off runs to reusable setups.
- Useful when you want to recreate a look/style from an existing output and iterate from there.

### Recipe Renderer
Runs recipe-driven generation so you can rerun pipelines quickly.
- Interprets recipe data during execution to reproduce prior results or batch variants with minimal manual work.
- Cuts down setup time for repeated jobs by moving configuration into reusable recipe data.
- Especially handy when testing different prompts or inputs while keeping the same core recipe.

### Recipe Relay
A clean handoff point for recipe data between nodes.
- Passes recipe data and compatible context across the graph.
- Keeps larger graphs modular and easier to follow.
- Preserves context through the pipeline, so you don't lose recipe continuity when splitting graphs into stages.

### Recipe Model Loader
Loads models/checkpoints based on recipe data.
- Resolves model selections encoded in recipes so execution follows the intended model path automatically.
- Supports model slot selection and loader-specific behavior.
- Makes recipe playback more reliable across sessions by tying loader behavior to the saved recipe, not memory.
- Useful for multi-model pipelines where model choice is part of the recipe itself.

### Recipe Manager
Save, browse, and reuse recipe entries.
- Persistent recipe storage so successful setups can be cataloged and recalled quickly.
- Works with Builder for quick review and updates.
- Encourages a library-style workflow: capture working recipes, name/tag them, and reapply them across projects.
- Good for keeping repeatable production presets while still allowing safe edits over time.

## Model Family Support

Built-in support includes:
- Flux 1
- Flux 2
- Ernie
- SDXL
- Wan
- Qwen
- Z-Image
- Wan Image
- Wan Video (I2V and T2V)
