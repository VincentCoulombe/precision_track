// Bootstrap: load components, fetch config, wire nav. The Configure view
// auto-saves each field, so there is no save bar.
import "./components/toaster.js";
import "./components/fs-picker.js";
import "./components/terminal.js";
import { store } from "./store.js";
import { renderConfigure } from "./views/configure.js";
import { renderRun } from "./views/run.js";

const views = {
  configure: document.getElementById("view-configure"),
  run: document.getElementById("view-run"),
};

function showView(name) {
  if (!views[name]) name = "configure";
  Object.entries(views).forEach(([k, node]) => node.classList.toggle("active", k === name));
  document.querySelectorAll(".nav-link").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  if (location.hash.slice(1) !== name) location.hash = name;
  if (name === "configure") renderConfigure(views.configure);
  if (name === "run") renderRun(views.run);
}

document.querySelectorAll(".nav-link").forEach((btn) => {
  btn.addEventListener("click", () => showView(btn.dataset.view));
});
window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

async function boot() {
  try {
    await store.load();
  } catch (e) {
    window.notify({ level: "error", title: "Could not load configuration", message: e.message });
    return;
  }
  showView(location.hash.slice(1) || "configure");
}

boot();
