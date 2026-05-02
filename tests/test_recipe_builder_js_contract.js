#!/usr/bin/env node
/*
  JS contract tests for Recipe Builder UI lock/persistence logic.
  Runs directly with Node and evaluates real functions from js/recipe_builder.js.

  Usage:
    node tests/test_recipe_builder_js_contract.js
*/

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert/strict");

const SOURCE_PATH = path.join(__dirname, "..", "js", "recipe_builder.js");
const SOURCE = fs.readFileSync(SOURCE_PATH, "utf8");

function extractFunction(name) {
  const marker = `function ${name}(`;
  const start = SOURCE.indexOf(marker);
  if (start < 0) {
    throw new Error(`Could not find function ${name} in ${SOURCE_PATH}`);
  }

  const bodyStart = SOURCE.indexOf("{", start);
  if (bodyStart < 0) {
    throw new Error(`Could not find function body start for ${name}`);
  }

  let depth = 0;
  let end = -1;
  for (let i = bodyStart; i < SOURCE.length; i += 1) {
    const ch = SOURCE[i];
    if (ch === "{") depth += 1;
    if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }

  if (end < 0) {
    throw new Error(`Could not find function body end for ${name}`);
  }

  return SOURCE.slice(start, end + 1);
}

const context = {
  console,
  requestAnimationFrame: (cb) => {
    if (typeof cb === "function") cb();
  },
  reflowNode: () => {},
  C: { textMuted: "#aaa", text: "#ccc" },
  cleanModelName: (v) => String(v ?? ""),
  normalizeSelectableFamily: (v) => String(v || "sdxl"),
  updateLoras: () => {},
  checkLoraAvailability: () => {},
  updateWanVisibility: () => {},
  _normalizeModelSlotKey: (v) => {
    const s = String(v || "model_a").toLowerCase();
    return ["model_a", "model_b", "model_c", "model_d"].includes(s) ? s : "model_a";
  },
  _normalizeSlotProfiles: (x) => (x && typeof x === "object" ? x : {}),
  document: {
    createElement: () => ({
      value: "",
      textContent: "",
      appendChild: () => {},
    }),
  },
};

vm.createContext(context);
vm.runInContext(extractFunction("mergeSlotProfileWithLocks"), context);
vm.runInContext(extractFunction("applyOverrides"), context);
vm.runInContext(extractFunction("syncHidden"), context);

function testMergeSlotProfileWithLocks() {
  const current = {
    ov: {
      model_a: "locked_model",
      positive_prompt: "locked_pos",
      negative_prompt: "unlocked_neg",
      steps_a: 77,
      loras_a: [{ name: "kept_lora" }],
      _section_locks: {
        model: true,
        sampler: true,
        positive: true,
        negative: false,
        loras: true,
      },
    },
    ls: { "a:kept_lora": { active: false, model_strength: 0.2, clip_strength: 0.2 } },
  };

  const incoming = {
    ov: {
      model_a: "incoming_model",
      positive_prompt: "incoming_pos",
      negative_prompt: "incoming_neg",
      steps_a: 10,
      loras_a: [{ name: "incoming_lora" }],
    },
    ls: { "a:incoming_lora": { active: true, model_strength: 1.0, clip_strength: 1.0 } },
  };

  const out = context.mergeSlotProfileWithLocks(current, incoming);

  assert.equal(out.ov.model_a, "locked_model", "model lock should preserve current model");
  assert.equal(out.ov.steps_a, 77, "sampler lock should preserve current sampler values");
  assert.equal(out.ov.positive_prompt, "locked_pos", "positive lock should preserve current positive prompt");
  assert.equal(out.ov.negative_prompt, "incoming_neg", "unlocked negative should adopt incoming value");
  assert.equal(
    JSON.stringify(out.ov.loras_a),
    JSON.stringify([{ name: "kept_lora" }]),
    "LoRA lock should preserve current LoRA list",
  );
  assert.equal(
    JSON.stringify(out.ls),
    JSON.stringify(current.ls),
    "LoRA lock should preserve current LoRA state map",
  );
}

function testApplyOverridesRestoresSectionLocks() {
  const calls = [];
  const node = {
    _weExtracted: {},
    _weUseSlotProfiles: false,
    _weSetSectionLock: (key, locked, options) => {
      calls.push({ key, locked, options });
    },
  };

  const ov = {
    _section_locks: {
      resolution: true,
      positive: true,
      loras: false,
    },
  };

  context.applyOverrides(node, JSON.stringify(ov), "{}");

  const byKey = Object.fromEntries(calls.map((c) => [c.key, c]));
  assert.equal(byKey.resolution.locked, true, "resolution lock should restore to true");
  assert.equal(byKey.positive.locked, true, "positive lock should restore to true");
  assert.equal(byKey.loras.locked, false, "loras lock should restore to false");
}

function testSyncHiddenPersistsLocksAndPrompts() {
  const widgets = [
    { name: "override_data", value: "" },
    { name: "lora_state", value: "" },
  ];

  const node = {
    widgets,
    properties: {},
    _weOverrides: {},
    _weSectionLocks: {
      resolution: true,
      positive: true,
      loras: false,
    },
    _weCollapsedSections: {
      resolution: false,
      model: true,
      sampler: false,
      positive: true,
      negative: false,
      loras: true,
    },
    _weUseSlotProfiles: false,
    _wePosBox: { value: "user_pos" },
    _weNegBox: { value: "user_neg" },
    _weExtracted: {
      loras_a: [],
      loras_b: [],
      lora_availability: {},
    },
  };

  context.syncHidden(node);

  const overrideJson = widgets.find((w) => w.name === "override_data").value;
  const ov = JSON.parse(overrideJson || "{}");

  assert.equal(ov.positive_prompt, "user_pos", "syncHidden should persist positive prompt from UI");
  assert.equal(ov.negative_prompt, "user_neg", "syncHidden should persist negative prompt from UI");
  assert.equal(ov._section_locks.resolution, true, "syncHidden should persist resolution lock");
  assert.equal(ov._section_locks.positive, true, "syncHidden should persist positive lock");
  assert.equal(Object.prototype.hasOwnProperty.call(ov._section_locks, "loras"), false, "unlocked section should not be persisted in non-slot mode");
  assert.equal(ov._section_collapsed.model, true, "syncHidden should persist global collapsed section state");
  assert.equal(ov._section_collapsed.positive, true, "syncHidden should persist positive collapsed state");
}

function run() {
  testMergeSlotProfileWithLocks();
  testApplyOverridesRestoresSectionLocks();
  testSyncHiddenPersistsLocksAndPrompts();
  console.log("JS contract tests passed");
}

run();
