// Configure view: renders user_configs.yaml as a form that auto-saves each field
// on commit. A field's value is written to user_configs.yaml as soon as it is
// committed (blur/Enter for inputs, instantly for toggles/dropdowns/picker). If
// the committed value is an *error* (not a warning) it is reverted to the field's
// previous value and not persisted. There is no Save button.
import { SCHEMA } from "../schema.js";
import { store } from "../store.js";
import { api } from "../api.js";
import { el, basename, relativePosix } from "../util.js";
import { openValidationEditor } from "./validation-editor.js";

let container;

export function renderConfigure(target) {
  container = target || container;
  container.innerHTML = "";
  container.appendChild(el("h1", { class: "view-title", text: "Configure your PrecisionTracker." }));
  container.appendChild(el("p", { class: "view-subtitle", text: "Every change is validated and saved automatically." }));
  SCHEMA.forEach((section) => {
    if (section.gate && !section.gate(store.config)) return;
    container.appendChild(renderSection(section));
  });
}

function renderSection(section) {
  const body = el("div", { class: "section-body" });
  section.fields.forEach((f) => body.appendChild(renderField(section, f)));

  const head = el("div", { class: "section-head" }, [
    el("h2", { class: "section-title" }, [section.title, section.sub ? el("span", { class: "section-sub", text: `  ·  ${section.sub}` }) : null]),
    el("span", { class: "section-chevron", text: "▾" }),
  ]);
  const card = el("div", { class: "section" }, [head, body]);
  head.addEventListener("click", () => card.classList.toggle("collapsed"));
  return card;
}

function renderField(section, f) {
  const fieldId = `${section.key}.${f.key}`;
  const label = el("div", { class: "field-label" }, [
    el("div", { class: "field-name", text: f.key }),
    f.help ? el("div", { class: "field-help", text: f.help }) : null,
  ]);
  const control = el("div", { class: "field-control" });

  if (f.type === "bool") control.appendChild(renderBool(section, f, fieldId));
  else if (f.type === "text") control.appendChild(renderText(section, f, fieldId));
  else if (f.type === "number") control.appendChild(renderNumber(section, f, fieldId, control));
  else if (f.type === "path") renderPath(section, f, fieldId, control);
  else if (f.type === "num_subjects") return renderNumSubjects(section, f, fieldId, label);

  return el("div", { class: "field" }, [el("div", { class: "field-row" }, [label, control])]);
}

// ------------------------------------------------------------------- autosave
// Persist the full config after a field was committed into `store`. On an error
// the field is reverted via `revert()`. Returns true when persisted.
async function autosave(fieldId, input, badge, revert) {
  try {
    const res = await api.saveField(fieldId, store.config);
    if (!res.saved) {
      revert();
      setBadge(input, badge, "error", res.message);
      if (res.message) window.notify({ level: "error", title: `Invalid: ${fieldId}`, message: res.message });
      return false;
    }
    const level = res.level || "ok";
    setBadge(input, badge, level, res.message);
    if (level === "warning" && res.message) window.notify({ level: "warning", title: fieldId, message: res.message });
    (res.created_dirs || []).forEach((d) => window.notify({ level: "success", title: "Directory created", message: d }));
    return true;
  } catch (e) {
    revert();
    window.notify({ level: "error", title: "Save failed", message: e.message });
    return false;
  }
}

// ----------------------------------------------------------------- widgets
function renderBool(section, f, fieldId) {
  let prev = store.get(section.key, f.key) === true;
  const input = el("input", { type: "checkbox" });
  input.checked = prev;
  input.addEventListener("change", async () => {
    const newVal = input.checked;
    store.set(section.key, f.key, newVal);
    // Cascade: offline correction refinement needs validation on, so turning
    // validation off also turns it off.
    let cascadePrev = null;
    if (section.key === "booleans" && f.key === "with_validation" && newVal === false && store.get("booleans", "with_offline_correction_refinement") === true) {
      cascadePrev = true;
      store.set("booleans", "with_offline_correction_refinement", false);
    }
    if (section.key === "booleans") renderConfigure(); // gated sections may appear/disappear
    const ok = await autosave(fieldId, null, null, () => {
      store.set(section.key, f.key, prev);
      input.checked = prev;
      if (cascadePrev !== null) store.set("booleans", "with_offline_correction_refinement", cascadePrev);
      if (section.key === "booleans") renderConfigure();
    });
    if (ok) prev = newVal;
  });
  const track = el("span", { class: "track" }, [el("span", { class: "thumb" })]);
  return el("label", { class: "toggle" }, [input, track]);
}

function renderText(section, f, fieldId) {
  let prev = store.get(section.key, f.key) ?? "";
  const input = el("input", { type: "text" });
  input.value = prev;
  input.addEventListener("change", async () => {
    const newVal = input.value;
    if (newVal === prev) return;
    store.set(section.key, f.key, newVal);
    const ok = await autosave(fieldId, input, null, () => {
      store.set(section.key, f.key, prev);
      input.value = prev;
    });
    if (ok) prev = newVal;
  });
  return input;
}

function renderNumber(section, f, fieldId, control) {
  let prev = store.get(section.key, f.key) ?? "";
  const input = el("input", { type: "number" });
  if (f.min != null) input.min = f.min;
  input.value = prev;
  const badge = el("span", { class: "badge" });
  input.addEventListener("change", async () => {
    const newVal = input.value === "" ? "" : Number(input.value);
    store.set(section.key, f.key, newVal);
    const ok = await autosave(fieldId, input, badge, () => {
      store.set(section.key, f.key, prev);
      input.value = prev;
    });
    if (ok) prev = newVal;
  });
  control.appendChild(badge);
  return input;
}

function renderPath(section, f, fieldId, control) {
  let prev = store.get(section.key, f.key) ?? "";
  const input = el("input", { type: "text" });
  input.value = prev;
  const badge = el("span", { class: "badge" });
  const browse = el("button", { class: "btn", text: "Browse…" });
  const isMetainfo = section.key === "general" && f.key === "metainfo";

  const commit = async () => {
    const newVal = input.value;
    if (newVal === prev) return;
    store.set(section.key, f.key, newVal);
    if (isMetainfo) await store.refreshMetainfoClasses();
    const ok = await autosave(fieldId, input, badge, () => {
      store.set(section.key, f.key, prev);
      input.value = prev;
      if (isMetainfo) store.refreshMetainfoClasses();
    });
    if (ok) prev = newVal;
  };
  input.addEventListener("change", commit);
  browse.addEventListener("click", async () => {
    const chosen = await browseFor(section, f);
    if (!chosen) return;
    input.value = chosen;
    commit();
  });

  control.appendChild(input);
  control.appendChild(browse);
  if (f.editor) {
    const editBtn = el("button", { class: "btn", text: "Edit values" });
    editBtn.addEventListener("click", () => openValidationEditor(store.get(section.key, f.key)));
    control.appendChild(editBtn);
  }
  control.appendChild(badge);
}

async function browseFor(section, f) {
  let startPath = null;
  let baseAbs = store.paths.tools_dir;
  if (f.baseField) {
    const [bs, bk] = f.baseField.split(".");
    const baseVal = store.get(bs, bk);
    if (baseVal) {
      try {
        baseAbs = (await api.resolve(baseVal)).abs;
      } catch {
        /* fall back to tools_dir */
      }
    }
    startPath = baseAbs;
  } else {
    const cur = store.get(section.key, f.key);
    if (cur) {
      try {
        startPath = (await api.resolve(cur)).abs;
      } catch {
        /* fall back to tools_dir */
      }
    }
  }
  const chosen = await document.querySelector("pt-fs-picker").open({
    mode: f.picker.mode,
    exts: f.picker.exts || [],
    startPath,
  });
  if (!chosen) return null;

  if (f.store === "basename") return basename(chosen);
  if (f.store === "data_root") return relativePosix(baseAbs, chosen);
  return relativePosix(store.paths.tools_dir, chosen);
}

function renderNumSubjects(section, f, fieldId, label) {
  const wrap = el("div", { class: "field-control", style: "flex-direction:column; align-items:stretch;" });
  const rows = el("div");
  const badge = el("span", { class: "badge" });

  const value = () => store.get(section.key, f.key) || {};
  const clone = (o) => JSON.parse(JSON.stringify(o));
  let prev = clone(value());

  const commit = async (obj) => {
    store.set(section.key, f.key, obj);
    const ok = await autosave(fieldId, null, badge, () => {
      store.set(section.key, f.key, prev);
      rebuild();
    });
    if (ok) prev = clone(obj);
  };

  function rebuild() {
    rows.innerHTML = "";
    Object.entries(value()).forEach(([cls, count]) => rows.appendChild(row(cls, count)));
  }

  function row(cls, count) {
    const classes = store.metainfoClasses || [];
    let clsInput;
    if (classes.length) {
      clsInput = el("select");
      classes.forEach((c) => clsInput.appendChild(el("option", { value: c, text: c })));
      if (cls && !classes.includes(cls)) clsInput.appendChild(el("option", { value: cls, text: cls }));
      clsInput.value = cls;
    } else {
      clsInput = el("input", { type: "text" });
      clsInput.value = cls;
    }
    const countInput = el("input", { type: "number" });
    countInput.value = count;
    const del = el("button", { class: "btn btn-ghost", text: "✕" });

    const r = el("div", { class: "ns-row" }, [clsInput, countInput, del]);
    const sync = () => {
      const obj = {};
      rows.querySelectorAll(".ns-row").forEach((rr) => {
        const k = rr.children[0].value.trim();
        const v = rr.children[1].value;
        if (k !== "") obj[k] = v === "" ? "" : parseInt(v, 10);
      });
      commit(obj);
    };
    clsInput.addEventListener("change", sync);
    countInput.addEventListener("change", sync);
    del.addEventListener("click", () => {
      r.remove();
      sync();
    });
    return r;
  }

  const add = el("button", { class: "btn", text: "+ Add class" });
  add.addEventListener("click", () => {
    const obj = value();
    const candidate = (store.metainfoClasses || []).find((c) => !(c in obj)) || "";
    rows.appendChild(row(candidate, 1));
  });

  rebuild();
  wrap.appendChild(rows);
  wrap.appendChild(el("div", {}, [add, badge]));
  return el("div", { class: "field" }, [el("div", { class: "field-row" }, [label, wrap])]);
}

// --------------------------------------------------------------- badges
function setBadge(input, badge, level, message) {
  if (badge) {
    badge.textContent = level === "ok" ? "✓" : level === "warning" ? "!" : "✕";
    badge.className = `badge ${level === "ok" ? "ok" : level === "warning" ? "warn" : "err"}`;
    badge.title = message || "";
  }
  if (input) {
    input.classList.toggle("valid", level === "ok");
    input.classList.toggle("invalid", level === "error");
  }
}
