// Modal editor for the validation configuration file (appearance | aruco) and,
// for appearance, the linked ReID metainfo identities / disabled_identities.
import { api } from "../api.js";
import { store } from "../store.js";
import { el } from "../util.js";

const APPEARANCE = "AppearanceValidation";
const ARUCO = "ArucoValidation";

let host;

function ensureHost() {
  if (!host) {
    host = document.createElement("div");
    document.body.appendChild(host);
  }
  return host;
}

function getNested(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}
function setNested(obj, path, val) {
  const keys = path.split(".");
  let o = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    if (o[keys[i]] == null || typeof o[keys[i]] !== "object") o[keys[i]] = {};
    o = o[keys[i]];
  }
  o[keys[keys.length - 1]] = val;
}

export async function openValidationEditor(path) {
  if (!path) {
    window.notify({ level: "error", title: "No file", message: "Set a validation configuration file first." });
    return;
  }
  let loaded;
  try {
    loaded = await api.getValidationConfig(path);
  } catch (e) {
    window.notify({ level: "error", title: "Load failed", message: e.message });
    return;
  }
  if (!loaded.ok) {
    window.notify({ level: "error", title: "Load failed", message: loaded.message });
    return;
  }
  new ValidationEditor(path, loaded.config || {}).render();
}

class ValidationEditor {
  constructor(path, config) {
    this.path = path;
    this.config = config;
    if (!this.config.type) this.config.type = APPEARANCE;
  }

  async render() {
    const h = ensureHost();
    h.innerHTML = `
      <div class="modal-backdrop">
        <div class="modal" style="width:680px;">
          <div class="modal-head">
            <span class="modal-title">Validation configuration</span>
            <button class="btn btn-ghost" data-act="cancel">Close</button>
          </div>
          <div class="modal-path">${this.path}</div>
          <div class="modal-list" data-body style="padding:16px 20px;"></div>
          <div class="modal-foot" style="justify-content:flex-end;">
            <button class="btn btn-primary" data-act="save">Save validation config</button>
          </div>
        </div>
      </div>`;
    h.querySelector('[data-act="cancel"]').addEventListener("click", () => (h.innerHTML = ""));
    h.querySelector('[data-act="save"]').addEventListener("click", () => this.save());
    this.body = h.querySelector("[data-body]");
    this.renderBody();
  }

  renderBody() {
    this.body.innerHTML = "";
    this.body.appendChild(this.strategyRow());
    if (this.config.type === ARUCO) this.renderAruco();
    else this.renderAppearance();
  }

  strategyRow() {
    const select = el("select");
    [APPEARANCE, ARUCO].forEach((s) => select.appendChild(el("option", { value: s, text: s })));
    select.value = this.config.type;
    select.addEventListener("change", async () => {
      const keepClasses = this.config.validated_classes || [];
      const tpl = (await api.validationTemplate(select.value)).config;
      tpl.validated_classes = keepClasses;
      this.config = tpl;
      this.renderBody();
    });
    return this.row("type (strategy)", select, "Appearance-based ReID or the Tailtag (ArUco) system.");
  }

  // -------------------------------------------------------------- widgets
  row(label, control, help) {
    return el("div", { class: "field" }, [
      el("div", { class: "field-row" }, [
        el("div", { class: "field-label" }, [el("div", { class: "field-name", text: label }), help ? el("div", { class: "field-help", text: help }) : null]),
        el("div", { class: "field-control" }, control),
      ]),
    ]);
  }

  textField(label, dotted, { help, exts, picker } = {}) {
    const input = el("input", { type: "text" });
    input.value = getNested(this.config, dotted) ?? "";
    input.addEventListener("change", () => setNested(this.config, dotted, input.value));
    const controls = [input];
    if (picker) {
      const browse = el("button", { class: "btn", text: "Browse…" });
      browse.addEventListener("click", async () => {
        const chosen = await document.querySelector("pt-fs-picker").open({ mode: "file", exts: exts || [] });
        if (chosen) {
          input.value = chosen;
          setNested(this.config, dotted, chosen);
        }
      });
      controls.push(browse);
    }
    return this.row(label, controls, help);
  }

  numberField(label, dotted, { help, step } = {}) {
    const input = el("input", { type: "number" });
    if (step) input.step = step;
    input.value = getNested(this.config, dotted) ?? "";
    input.addEventListener("change", () => setNested(this.config, dotted, input.value === "" ? null : Number(input.value)));
    return this.row(label, input, help);
  }

  selectField(label, dotted, options, help) {
    const select = el("select");
    options.forEach((o) => select.appendChild(el("option", { value: o, text: o })));
    select.value = getNested(this.config, dotted) ?? options[0];
    select.addEventListener("change", () => setNested(this.config, dotted, select.value));
    return this.row(label, select, help);
  }

  listField(label, dotted, { help, numeric } = {}) {
    const cur = getNested(this.config, dotted) || [];
    const input = el("input", { type: "text" });
    input.value = cur.join(", ");
    input.placeholder = numeric ? "e.g. 0, 1, 2" : "comma separated";
    input.addEventListener("change", () => {
      const parts = input.value.split(",").map((s) => s.trim()).filter((s) => s !== "");
      setNested(this.config, dotted, numeric ? parts.map(Number) : parts);
    });
    return this.row(label, input, help);
  }

  classesField() {
    return this.listField("validated_classes", "validated_classes", {
      help: `Classes to re-identify. Known classes: ${(store.metainfoClasses || []).join(", ") || "—"}`,
    });
  }

  // ------------------------------------------------------------ strategies
  renderAppearance() {
    this.body.appendChild(this.selectField("data_preprocessor.type", "data_preprocessor.type", ["WildLifeReIDPreprocessor"]));
    this.body.appendChild(this.textField("re_identificator.metainfo", "re_identificator.metainfo", { picker: true, exts: [".yaml", ".yml"], help: "ReID model metadata YAML." }));
    this.body.appendChild(this.textField("re_identificator.checkpoint", "re_identificator.checkpoint", { picker: true, exts: [".onnx", ".engine", ".pth"], help: "ReID ONNX checkpoint." }));
    this.body.appendChild(this.classesField());
    this.renderReidIdentities();
  }

  renderAruco() {
    this.body.appendChild(this.classesField());
    this.body.appendChild(this.numberField("num_tags", "num_tags"));
    this.body.appendChild(this.numberField("tags_size", "tags_size"));
    this.body.appendChild(this.textField("predefined_dict", "predefined_dict", { help: 'OpenCV dict name (e.g. "DICT_4X4_50") or leave empty for null.' }));
    this.body.appendChild(this.selectField("refinement", "refinement", ["none", "contour", "subpix", "apriltag"]));
    this.body.appendChild(this.numberField("tag_kpt", "tag_kpt"));
    this.body.appendChild(this.numberField("kpt_conf_thr", "kpt_conf_thr", { step: "0.01" }));
    this.body.appendChild(this.numberField("estimation_range", "estimation_range"));
    this.body.appendChild(this.numberField("timeout_after", "timeout_after", { step: "0.001" }));
    this.body.appendChild(this.numberField("min_sample_size", "min_sample_size"));
    this.body.appendChild(this.listField("valid_tags", "valid_tags", { numeric: true, help: "Tag IDs physically present." }));
    const params = ["minMarkerPerimeterRate", "maxMarkerPerimeterRate", "adaptiveThreshWinSizeMin", "adaptiveThreshWinSizeMax", "adaptiveThreshWinSizeStep", "polygonalApproxAccuracyRate", "minOtsuStdDev", "perspectiveRemovePixelPerCell", "perspectiveRemoveIgnoredMarginPerCell"];
    this.body.appendChild(el("div", { class: "field" }, [el("div", { class: "field-name", text: "parameters (OpenCV ArUco detector)" })]));
    params.forEach((p) => this.body.appendChild(this.numberField(p, `parameters.${p}`, { step: "0.01" })));
  }

  async renderReidIdentities() {
    const metaPath = getNested(this.config, "re_identificator.metainfo");
    const sep = el("div", { class: "field" }, [el("div", { class: "field-name", text: "ReID identities (metainfo file)" })]);
    this.body.appendChild(sep);
    if (!metaPath) {
      this.body.appendChild(this.row("", el("div", { class: "field-help", text: "Set re_identificator.metainfo to edit identities." })));
      return;
    }
    let data;
    try {
      data = await api.getReidMetainfo(metaPath);
    } catch (e) {
      this.body.appendChild(this.row("", el("div", { class: "field-help", text: `Could not load: ${e.message}` })));
      return;
    }
    if (!data.ok) {
      this.body.appendChild(this.row("", el("div", { class: "field-help", text: data.message })));
      return;
    }
    this.reid = { path: data.path, identities: [...data.identities], disabled: new Set(data.disabled_identities) };
    this.reidBox = el("div");
    this.body.appendChild(this.reidBox);
    this.renderReidBox();
  }

  renderReidBox() {
    this.reidBox.innerHTML = "";
    this.reid.identities.forEach((id, i) => {
      const nameInput = el("input", { type: "text" });
      nameInput.value = id;
      nameInput.addEventListener("change", () => {
        const old = this.reid.identities[i];
        this.reid.identities[i] = nameInput.value.trim();
        if (this.reid.disabled.has(old)) {
          this.reid.disabled.delete(old);
          this.reid.disabled.add(nameInput.value.trim());
        }
      });
      const disabled = el("input", { type: "checkbox" });
      disabled.checked = this.reid.disabled.has(id);
      disabled.addEventListener("change", () => {
        const name = this.reid.identities[i];
        if (disabled.checked) this.reid.disabled.add(name);
        else this.reid.disabled.delete(name);
      });
      const del = el("button", { class: "btn btn-ghost", text: "✕" });
      del.addEventListener("click", () => {
        this.reid.disabled.delete(this.reid.identities[i]);
        this.reid.identities.splice(i, 1);
        this.renderReidBox();
      });
      this.reidBox.appendChild(
        el("div", { class: "ns-row" }, [nameInput, el("label", { class: "field-help", style: "flex:0 0 auto; display:flex; gap:6px; align-items:center;" }, [disabled, "disabled"]), del])
      );
    });
    const add = el("button", { class: "btn", text: "+ Add identity" });
    add.addEventListener("click", () => {
      this.reid.identities.push("");
      this.renderReidBox();
    });
    this.reidBox.appendChild(add);
  }

  // ------------------------------------------------------------------ save
  async save() {
    // normalize predefined_dict empty -> null
    if (this.config.type === ARUCO && (this.config.predefined_dict === "" || this.config.predefined_dict === undefined)) {
      this.config.predefined_dict = null;
    }
    try {
      await api.saveValidationConfig(this.path, this.config);
      if (this.config.type === APPEARANCE && this.reid) {
        const identities = this.reid.identities.filter((s) => s !== "");
        const disabled = [...this.reid.disabled].filter((s) => s !== "");
        const res = await api.saveReidMetainfo(this.reid.path, identities, disabled);
        if (!res.saved) {
          window.notify({ level: "error", title: "ReID identities not saved", message: res.message });
          return;
        }
      }
      window.notify({ level: "success", title: "Saved", message: "Validation configuration updated." });
      host.innerHTML = "";
    } catch (e) {
      window.notify({ level: "error", title: "Save failed", message: e.message });
    }
  }
}
