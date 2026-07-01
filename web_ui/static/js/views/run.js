// Run view: pick a tool, set its flags, launch it, watch live output.
import { api } from "../api.js";
import { store } from "../store.js";
import { el } from "../util.js";

// Tools grouped into sections (top to bottom). AR/MART tools listed in AR_TOOLS
// are hidden when with_action_recognition is off.
const SECTIONS = [
  { title: "Tracking", tools: ["track.py", "batch_track.py", "batch_track_directory.py"] },
  { title: "Visualization", tools: ["visualize.py", "plot_profiles.py"] },
  { title: "Training", tools: ["train_detection.py", "train_action_recognition.py"] },
  { title: "Testing", tools: ["test_detection.py", "test_tracking.py", "test_action_recognition.py"] },
];
const AR_TOOLS = new Set(["train_action_recognition.py", "test_action_recognition.py"]);

let container;
let tools = null;
let selected = null;
const values = {}; // per-tool { argName: value }
let terminal, pill, stopBtn, runBtn;
let pollTimer = null;

export async function renderRun(target) {
  container = target || container;
  if (!tools) {
    try {
      tools = (await api.getTools()).tools;
    } catch (e) {
      window.notify({ level: "error", title: "Could not load tools", message: e.message });
      return;
    }
  }
  container.innerHTML = "";
  container.appendChild(el("h1", { class: "view-title", text: "1) Pick a tool.  2) Set its options.  3) Run it." }));
  container.appendChild(el("p", { class: "view-subtitle", text: "Output streams live below." }));
  container.appendChild(renderSections());
  container.appendChild(renderArgsPanel());
  container.appendChild(renderRunner());
  startPolling();
  reconcile();
}

function renderSections() {
  const withAR = store.get("booleans", "with_action_recognition") === true;
  const wrap = el("div");
  let shown = 0;
  SECTIONS.forEach((section) => {
    const names = section.tools.filter((n) => tools[n] && (withAR || !AR_TOOLS.has(n)));
    if (!names.length) return;
    const row = el("div", { class: "tool-row" }, names.map(toolCard));
    wrap.appendChild(
      el("div", { class: `tool-section ${shown > 0 ? "divided" : ""}` }, [el("div", { class: "tool-section-label", text: section.title }), row])
    );
    shown += 1;
  });
  return wrap;
}

function toolCard(name) {
  const spec = tools[name];
  const card = el("button", { class: `tool-card ${selected === name ? "selected" : ""}` }, [
    el("div", { class: "t-label", text: spec.label }),
    el("div", { class: "t-desc", text: spec.description }),
  ]);
  card.addEventListener("click", () => {
    selected = name;
    if (!values[name]) values[name] = defaultsFor(spec);
    renderRun();
  });
  return card;
}

function defaultsFor(spec) {
  const v = {};
  spec.flags.forEach((f) => {
    if ("default" in f) v[f.name] = f.default;
  });
  return v;
}

function renderArgsPanel() {
  const panel = el("div", { class: "section" });
  if (!selected) {
    panel.appendChild(el("div", { class: "section-body" }, [el("p", { class: "field-help", text: "Select a tool above." })]));
    return panel;
  }
  const spec = tools[selected];
  const body = el("div", { class: "section-body" });
  if (spec.warning) {
    body.appendChild(el("div", { class: "field-warning" }, [el("strong", { text: "⚠ Warning " }), document.createTextNode(spec.warning)]));
  }
  spec.positionals.forEach((p) => body.appendChild(argRow(p, true)));
  spec.flags.forEach((f) => body.appendChild(argRow(f, false)));
  panel.appendChild(el("div", { class: "section-head" }, [el("h2", { class: "section-title", text: `${spec.label} — options` })]));
  panel.appendChild(body);
  return panel;
}

function argRow(arg, required) {
  const label = el("div", { class: "field-label" }, [
    el("div", { class: "field-name", text: arg.name + (required ? " *" : "") }),
    arg.help ? el("div", { class: "field-help", text: arg.help }) : null,
  ]);
  const control = el("div", { class: "field-control" });
  const v = values[selected];

  if (arg.type === "bool") {
    const input = el("input", { type: "checkbox" });
    input.checked = v[arg.name] === true;
    input.addEventListener("change", () => (v[arg.name] = input.checked));
    control.appendChild(el("label", { class: "toggle" }, [input, el("span", { class: "track" }, [el("span", { class: "thumb" })])]));
  } else if (arg.type === "float" || arg.type === "number") {
    const input = el("input", { type: "number" });
    if (arg.type === "float") input.step = "any";
    input.value = v[arg.name] ?? "";
    input.addEventListener("change", () => (v[arg.name] = input.value === "" ? "" : Number(input.value)));
    control.appendChild(input);
  } else {
    // path / text
    const input = el("input", { type: "text" });
    input.value = v[arg.name] ?? "";
    input.addEventListener("change", () => (v[arg.name] = input.value));
    control.appendChild(input);
    if (arg.picker) {
      const browse = el("button", { class: "btn", text: "Browse…" });
      browse.addEventListener("click", async () => {
        // visualize.py reads tracking outputs from saving_directory, so start there.
        let startPath = null;
        if (selected === "visualize.py") {
          const sd = store.get("tracking", "saving_directory");
          if (sd) {
            try {
              startPath = (await api.resolve(sd)).abs;
            } catch {
              /* fall back to the repo-root default */
            }
          }
        }
        const chosen = await document.querySelector("pt-fs-picker").open({ mode: arg.picker.mode, exts: arg.picker.exts || [], startPath });
        if (chosen) {
          input.value = chosen;
          v[arg.name] = chosen;
        }
      });
      control.appendChild(browse);
    }
  }
  return el("div", { class: "field" }, [el("div", { class: "field-row" }, [label, control])]);
}

function renderRunner() {
  const wrap = el("div");
  pill = el("span", { class: "status-pill", text: "idle" });
  runBtn = el("button", { class: "btn btn-primary", text: "Run" });
  stopBtn = el("button", { class: "btn btn-danger", text: "Stop" });
  runBtn.disabled = !selected;
  runBtn.addEventListener("click", onRun);
  stopBtn.addEventListener("click", onStop);

  wrap.appendChild(el("div", { class: "run-bar" }, [runBtn, stopBtn, pill]));
  terminal = document.createElement("pt-terminal");
  wrap.appendChild(terminal);
  return wrap;
}

function setStatus(status) {
  if (!pill) return;
  pill.textContent = status;
  pill.className = `status-pill ${status}`;
  const running = status === "running";
  if (stopBtn) stopBtn.disabled = !running;
  if (runBtn) runBtn.disabled = running || !selected;
}

async function onRun() {
  const spec = tools[selected];
  for (const p of spec.positionals) {
    if (!values[selected][p.name]) {
      window.notify({ level: "error", title: "Missing argument", message: `${p.name} is required.` });
      return;
    }
  }
  try {
    await api.run(selected, values[selected]);
    terminal.connect((s) => setStatus(s || "done"));
    setStatus("running");
  } catch (e) {
    if (e.status === 409) window.notify({ level: "warning", title: "Already running", message: e.message });
    else window.notify({ level: "error", title: "Launch failed", message: e.message });
  }
}

async function onStop() {
  try {
    await api.stopRun();
  } catch (e) {
    window.notify({ level: "error", title: "Stop failed", message: e.message });
  }
}

// On (re)entering the view, reconcile with any job already running.
async function reconcile() {
  try {
    const data = await api.getRun();
    if (data.job) {
      setStatus(data.job.status);
      if (data.running) terminal.connect((s) => setStatus(s || "done"));
      else {
        terminal.clear();
        (data.tail || []).forEach((l) => terminal.commitLine(l));
        if (data.live) terminal.liveLine(data.live);
      }
    } else {
      setStatus("idle");
    }
  } catch {
    setStatus("idle");
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (!pill) return;
    try {
      const data = await api.getRun();
      if (data.job && pill.textContent !== data.job.status) setStatus(data.job.status);
    } catch {
      /* ignore */
    }
  }, 3000);
}
