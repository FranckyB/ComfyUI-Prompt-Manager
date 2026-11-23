// Prompt Manager's small form dialogs on the FB UI kit modal: text fields and
// selects in a column, Enter to confirm, Esc or a click outside to cancel.
//
// formDialog({ title, fields, confirmText }) -> Promise<{ [key]: value } | null>
//   fields: [{ key, label, type: "text" | "select" | "check", value, options }]
//   options: values or { value, label }, or (values) => options for a select
//   that follows the fields above it (a category list per prompt group).
import { button } from "./fbui/controls.js";
import { el } from "./fbui/dom.js";
import { openModal } from "./fbui/modal.js";
import { T } from "./fbui/theme.js";

function fillSelect(select, options, value) {
    select.replaceChildren(...options.map((option) => {
        const entry = typeof option === "object" && option ? option : { value: option, label: option };
        const item = el("option", "", String(entry.label ?? entry.value ?? ""));
        item.value = String(entry.value ?? "");
        return item;
    }));
    if (options.length) select.value = String(value ?? "");
    if (!select.value && select.options.length) select.selectedIndex = 0;
}

export function formDialog({ title, fields, confirmText = "OK" }) {
    return new Promise((resolve) => {
        let answer = null;
        const controls = new Map();
        const values = () => Object.fromEntries([...controls].map(([key, control]) => [key, control.type === "checkbox" ? control.checked : control.value]));
        const refreshDependent = () => {
            for (const field of fields) {
                if (field.type === "select" && typeof field.options === "function") {
                    const select = controls.get(field.key);
                    const keep = select.value || field.value;
                    fillSelect(select, field.options(values()), keep);
                }
            }
        };

        const body = fields.map((field, index) => {
            if (field.type === "check") {
                const box = el("input");
                box.type = "checkbox";
                box.checked = Boolean(field.value);
                box.style.cssText = `margin:0;accent-color:${T.brand}`;
                controls.set(field.key, box);
                const row = el("label");
                row.style.cssText = `display:flex;align-items:center;gap:8px;margin-top:${index ? 10 : 0}px;color:${T.inkSoft};font-size:12px;cursor:pointer`;
                row.append(box, el("span", "", field.label));
                return row;
            }
            const label = el("div", "", field.label);
            label.style.cssText = `margin:${index ? 10 : 0}px 0 5px;color:${T.label};font-size:12px`;
            let control;
            if (field.type === "select") {
                control = el("select", "fbui-select");
                if (typeof field.options !== "function") fillSelect(control, field.options ?? [], field.value);
                control.addEventListener("change", refreshDependent);
            } else {
                control = el("input", "fbui-text");
                control.type = "text";
                control.spellcheck = false;
                control.value = String(field.value ?? "");
            }
            control.style.width = "100%";
            // Enter moves to the next field, and confirms on the last one.
            control.addEventListener("keydown", (event) => {
                if (event.key !== "Enter") return;
                event.preventDefault();
                const next = [...controls.values()].slice([...controls.keys()].indexOf(field.key) + 1).find((other) => other.type !== "checkbox");
                if (next) next.focus();
                else accept();
            });
            controls.set(field.key, control);
            const row = el("div");
            row.append(label, control);
            return row;
        });

        const accept = () => {
            answer = values();
            dialog.close();
        };
        const dialog = openModal({
            title,
            body,
            width: 440,
            footer: [button("Cancel", { onClick: () => dialog.close() }), button(confirmText, { primary: true, onClick: accept })],
            onClose: () => resolve(answer),
        });
        refreshDependent();
        const first = controls.values().next().value;
        first?.focus();
        if (first?.type === "text") first.select();
    });
}
