let lockStyleInstalled = false;

function installLockStyle() {
    if (lockStyleInstalled) return;
    lockStyleInstalled = true;
    const style = document.createElement("style");
    style.textContent = '.lg-node-widgets textarea[data-pm-locked="true"] { opacity: 0.5; }';
    document.head.appendChild(style);
}

// Read-only + ghosted prompt textarea: classic edits inputEl; Nodes 2.0 renders its own textarea from widget options.
export function setPromptTextLocked(widget, locked) {
    if (!widget) return;
    locked = !!locked;
    if (widget.inputEl) {
        widget.inputEl.readOnly = locked;
        if (locked) widget.inputEl.style.pointerEvents = "auto";
        widget.inputEl.style.opacity = locked ? "0.5" : "";
    }
    if (!!widget.options?.read_only === locked) return;
    installLockStyle();
    // Reassigning options (not mutating) is what makes the Vue widget store react.
    widget.options = { ...widget.options, read_only: locked, "data-pm-locked": locked ? "true" : undefined };
}
