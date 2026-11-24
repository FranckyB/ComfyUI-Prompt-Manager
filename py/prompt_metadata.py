"""The prompt a Prompt Manager node output, written into the workflow and API prompt that save nodes embed, so a
saved file carries it as typed text instead of a link to the node that fed it."""


def get_workflow_node(extra_pnginfo, node_id):
    """Find workflow node by id, including nested subgraphs (id chains like a:b:c)."""
    if not isinstance(extra_pnginfo, dict):
        return None

    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return None

    nodes_list = workflow.get("nodes", []) if isinstance(workflow.get("nodes"), list) else []
    subgraphs = []
    definitions = workflow.get("definitions")
    if isinstance(definitions, dict):
        maybe_subgraphs = definitions.get("subgraphs")
        if isinstance(maybe_subgraphs, list):
            subgraphs = maybe_subgraphs

    found = None
    for part in str(node_id).split(":"):
        found = next((n for n in nodes_list if isinstance(n, dict) and str(n.get("id")) == part), None)
        if not isinstance(found, dict):
            return None

        subgraph = next((sg for sg in subgraphs if isinstance(sg, dict) and str(sg.get("id")) == str(found.get("type"))), None)
        if isinstance(subgraph, dict) and isinstance(subgraph.get("nodes"), list):
            nodes_list = subgraph.get("nodes", [])

    return found


def record_prompt(unique_id, text, extra_pnginfo, api_prompt, toggle, link_input, text_index):
    """Store `text` as the node's typed prompt with its input `toggle` off (widget 2) and drop the `link_input` link:
    `text_index` is the text widget's position in widgets_values."""
    node_id = str(unique_id) if unique_id is not None else ""
    if not node_id:
        return

    workflow_node = get_workflow_node(extra_pnginfo, node_id)
    widgets = workflow_node.get("widgets_values") if isinstance(workflow_node, dict) else None
    if isinstance(widgets, list):
        if len(widgets) > 2:
            widgets[2] = False
        if len(widgets) > text_index:
            widgets[text_index] = text

    inputs = api_prompt.get(node_id, {}).get("inputs") if isinstance(api_prompt, dict) else None
    if isinstance(inputs, dict):
        inputs[toggle] = False
        inputs["text"] = text
        inputs.pop(link_input, None)
