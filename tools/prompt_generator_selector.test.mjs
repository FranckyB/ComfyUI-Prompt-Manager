import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../js/prompt_generator.js", import.meta.url), "utf8");
const adaptersSource = await readFile(new URL("../js/prompt_store_adapters.js", import.meta.url), "utf8");
const adapters = await import(`data:text/javascript,${encodeURIComponent(adaptersSource)}`);

class Element extends EventTarget {
    constructor(tagName) {
        super();
        this.tagName = tagName;
        this.style = {};
        this.children = [];
        this.attributes = {};
    }
    appendChild(child) {
        this.children.push(child);
        child.parentElement = this;
    }
    setAttribute(name, value) { this.attributes[name] = value; }
    dispatchEvent(event) {
        const result = super.dispatchEvent(event);
        if (event.bubbles && !event.cancelBubble) this.parentElement?.dispatchEvent(event);
        return result;
    }
}

async function createFixture() {
    const state = {
        prompts: {
            Video: { Shared: { prompt: "video instruction" } },
            Image: {
                Shared: { prompt: "image instruction" },
                First: { prompt: "first instruction" },
                Empty: { prompt: "" },
                Missing: { thumbnail: "preview.png" },
                __meta__: { nsfw: false },
                _internal: "not a prompt",
            },
        },
        responseOk: true,
        fetches: [],
        browserCalls: [],
        notifications: [],
        callbacks: [],
        nativeMenuOpens: 0,
        selection: null,
    };
    const app = {
        registerExtension(extension) { state.extension = extension; },
        graph: { setDirtyCanvas() {} },
        extensionManager: { toast: { add(message) { state.notifications.push(message); } } },
    };
    const api = {
        addEventListener() {},
        async fetchApi(url, options) {
            state.fetches.push({ url, options });
            return { ok: state.responseOk, status: 503, json: async () => state.prompts };
        },
    };
    const showThumbnailBrowser = async (node, category, name, options) => {
        state.browserCalls.push({ category, name, options });
        await options.loadPromptsFn(node);
        return state.selection;
    };
    const setup = vm.runInNewContext(
        `${source.replace(/^import .*;\r?\n/gm, "")}\nsetupSystemPromptSelector;`,
        {
            app, api, showThumbnailBrowser, ...adapters,
            document: { createElement: tagName => new Element(tagName) },
            console: { log() {}, error() {} },
        },
    );
    class Generator {
        constructor() {
            this.inputs = [];
            this.widgets = [
                { name: "seed", value: 1234 },
                { name: "control_after_generate", value: "randomize" },
                {
                    name: "system_prompt", type: "combo", value: "Image / First",
                    options: { values: ["Image / First", "Image / Shared", "Video / Shared"] },
                    callback(value) { state.callbacks.push(value); },
                    onPointerDown() { state.nativeMenuOpens++; },
                },
                { name: "prompt", value: "user prompt" },
                { name: "format_as_json", value: false },
                { name: "enable_thinking", value: false },
                { name: "stop_server_after", value: true },
                { name: "clear_vram_on_run", value: true },
                { name: "model", value: "existing-model.gguf" },
            ];
        }
        setSize(size) { this.size = size; }
        setDirtyCanvas() {}
        addDOMWidget(name, type, element, options) {
            const widget = {
                name, type, element, options,
                get value() { return options.getValue(); },
                set value(value) { options.setValue(value); },
            };
            this.widgets.push(widget);
            return widget;
        }
        onConfigure(info) {
            this.widgets.forEach((widget, index) => { widget.value = info.widgets_values[index]; });
        }
    }
    await state.extension.beforeRegisterNodeDef(Generator, { name: "PromptGenerator" }, app);
    const node = new Generator();
    const originalWidgets = [...node.widgets];
    node.onNodeCreated();
    const widget = node.widgets[2];
    const canvasContainer = new Element("div");
    for (const eventName of ["pointerdown", "mousedown"]) {
        canvasContainer.addEventListener(eventName, () => originalWidgets[2].onPointerDown());
    }
    canvasContainer.appendChild(widget.element);
    const click = async (position, type = "click") => {
        const buttonIndex = position < 39 ? 0 : position > node.size[0] - 39 ? 2 : 1;
        widget.element.children[buttonIndex].dispatchEvent(new Event(type, { bubbles: true }));
        await new Promise(setImmediate);
    };
    return { state, node, widget, originalWidgets, click, setup };
}

test("replaces the native combo with a DOM widget in the same serialized slot", async () => {
    const { node, widget, originalWidgets, setup } = await createFixture();
    const values = originalWidgets.map(entry => entry.value);
    values[2] = "Image / Shared";
    node.onConfigure({ widgets_values: values });
    setup(node);
    assert.equal(node.widgets.length, originalWidgets.length);
    for (const [index, original] of originalWidgets.entries()) {
        if (index !== 2) assert.equal(node.widgets[index], original);
    }
    assert.ok(!node.widgets.includes(originalWidgets[2]));
    assert.equal(widget.onPointerDown, undefined);
    assert.deepEqual(node.widgets.map(entry => entry.value), values);
    assert.equal(widget.type, "system_prompt_selector");
    assert.equal(widget.name, "system_prompt");
    assert.equal(widget.serializeValue(), "Image / Shared");
    assert.equal(widget.element.children[1].textContent, "Image : Shared");
    assert.deepEqual(JSON.parse(JSON.stringify({ widgets_values: node.widgets.map(entry => entry.value) })).widgets_values, values);
});

test("arrows wrap across categories and omit metadata and empty prompts", async () => {
    const { widget, state, click } = await createFixture();
    await click(20);
    assert.equal(widget.value, "Video / Shared");
    await click(380);
    assert.equal(widget.value, "Image / First");
    await click(380);
    assert.equal(widget.value, "Image / Shared");
    await click(380);
    assert.equal(widget.value, "Video / Shared");
    assert.equal(state.callbacks.length, 4);
    assert.ok(state.fetches.every(request => request.url === "/prompt-generator/get-prompts"));
});

test("round-trips the bundled Generator workflow without shifting stored values", async () => {
    const workflow = JSON.parse(await readFile(new URL("../example_workflows/generator_workflow.json", import.meta.url), "utf8"));
    const savedNode = workflow.nodes.find(entry => entry.type === "PromptGenerator");
    assert.ok(savedNode);
    const { node, widget, click, originalWidgets } = await createFixture();
    originalWidgets[2].options.values.push(savedNode.widgets_values[2]);
    node.onConfigure(savedNode);
    const serialize = () => JSON.parse(JSON.stringify(node.widgets.map(entry => entry.value)));
    assert.deepEqual(serialize(), savedNode.widgets_values);
    await click(380);
    const expected = [...savedNode.widgets_values];
    expected[2] = widget.value;
    assert.deepEqual(serialize(), expected);
});

test("name opens single-select System Prompts with current category and name", async () => {
    const { widget, state, click } = await createFixture();
    widget.value = "Video / Shared / suffix";
    state.selection = { category: "Image", prompt: "Shared" };
    await click(200);
    const call = state.browserCalls[0];
    assert.equal(call.category, "Video");
    assert.equal(call.name, "Shared / suffix");
    assert.equal(call.options.endpointPrefix, "/prompt-generator");
    assert.equal(call.options.preferenceScope, "system");
    assert.equal(call.options.promptOnly, true);
    assert.equal(call.options.allowMultiSelect, false);
    assert.equal(widget.value, "Image / Shared");
    assert.deepEqual(state.callbacks, ["Image / Shared"]);
});

test("cancellation and empty libraries preserve the stored selection", async () => {
    const { widget, state, click } = await createFixture();
    await click(200);
    assert.equal(widget.value, "Image / First");
    state.prompts = {};
    await click(380);
    assert.equal(widget.value, "Image / First");
    assert.equal(state.callbacks.length, 0);
});

test("browser cancellation preserves manual values outside workflow loading", async () => {
    const { widget, click } = await createFixture();
    widget.value = "deleted prompt";
    await click(200);
    assert.equal(widget.value, "deleted prompt");
    await click(380);
    assert.equal(widget.value, "Image / First");
    widget.value = "legacy mode";
    await click(20);
    assert.equal(widget.value, "Video / Shared");
});

test("load failures notify without overwriting values and allow retry", async () => {
    const { widget, state, click } = await createFixture();
    state.responseOk = false;
    await click(380);
    assert.equal(widget.value, "Image / First");
    assert.equal(state.notifications.length, 1);
    state.responseOk = true;
    await click(380);
    assert.equal(widget.value, "Image / Shared");
});

test("DOM pointer events do not reach native combo handling; name click opens the browser", async () => {
    const { state, click } = await createFixture();
    await click(380, "pointerup");
    await click(380, "pointermove");
    await click(200, "pointerdown");
    await click(200, "mousedown");
    assert.equal(state.fetches.length, 0);
    await click(200);
    assert.equal(state.browserCalls.length, 1);
    assert.equal(state.nativeMenuOpens, 0);
});

test("long labels use a constrained DOM button and expose the full name as a tooltip", async () => {
    const { widget } = await createFixture();
    widget.value = `Image / ${"Long prompt name ".repeat(12)}`;
    const button = widget.element.children[1];
    assert.equal(button.textContent, widget.value.replace(" / ", " : "));
    assert.ok(button.title.includes(widget.value));
    assert.match(button.style.cssText, /min-width: 0/);
    assert.match(button.style.cssText, /text-overflow: ellipsis/);
    assert.equal(button.tagName, "button");
    assert.equal(button.attributes["aria-label"], "Browse System Prompts");
    assert.equal(widget.options.getMinHeight(), 28);
    assert.equal(widget.options.getMaxHeight(), 28);
});

test("migrates the exact legacy value stored by Krea2_Workflow-09 on configure", async () => {
    const { node, widget, originalWidgets, click, state } = await createFixture();
    originalWidgets[2].options.values = ["Audio / Enhance Prompt", "Image / Enhance Prompt", "Video / Enhance Prompt"];
    const values = node.widgets.map(entry => entry.value);
    values[2] = "Enhance Prompt (Image)";
    node.onConfigure({ widgets_values: values });
    assert.equal(widget.value, "Image / Enhance Prompt");
    assert.equal(widget.element.children[1].textContent, "Image : Enhance Prompt");
    assert.equal(widget.serializeValue(), "Image / Enhance Prompt");
    values[2] = "Image / Enhance Prompt";
    assert.deepEqual(node.widgets.map(entry => entry.value), values);
    await click(200);
    assert.equal(state.browserCalls[0].category, "Image");
    assert.equal(state.browserCalls[0].name, "Enhance Prompt");
});

test("matches legacy categories and unique bare names, defaulting missing or ambiguous names", async () => {
    const { node, widget, originalWidgets } = await createFixture();
    originalWidgets[2].options.values = [
        "Audio / Enhance Prompt", "Image / Enhance Prompt", "Video / Enhance Prompt",
        "Other / Analyze Image", "Image / Shared", "Video / Shared",
    ];
    for (const [saved, expected] of [
        ["Enhance Prompt (Video)", "Video / Enhance Prompt"],
        ["Enhance Prompt (Audio)", "Audio / Enhance Prompt"],
        ["Analyze Image", "Other / Analyze Image"],
        ["Video / Shared", "Video / Shared"],
        ["Deleted Prompt (Video)", "Image / Enhance Prompt"],
        ["Video / Deleted Prompt", "Image / Enhance Prompt"],
        ["Shared", "Image / Enhance Prompt"],
    ]) {
        const values = node.widgets.map(entry => entry.value);
        values[2] = saved;
        node.onConfigure({ widgets_values: values });
        assert.equal(widget.value, expected, saved);
    }
});

test("uses another available entry if Enhance Prompt was removed, preserving values when choices are unavailable", async () => {
    const { node, widget, originalWidgets } = await createFixture();
    const values = node.widgets.map(entry => entry.value);
    values[2] = "Deleted Prompt";
    node.onConfigure({ widgets_values: values });
    assert.equal(widget.value, "Image / First");
    originalWidgets[2].options.values = [];
    node.onConfigure({ widgets_values: values });
    assert.equal(widget.value, "Deleted Prompt");
});