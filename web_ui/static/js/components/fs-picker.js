// Modal file/folder selector. Usage:
//   const path = await document.querySelector("pt-fs-picker")
//       .open({ mode: "file"|"dir"|"save", exts: [".pth"], startPath, filename });
// Resolves to an absolute path string, or null if cancelled.
import { api } from "../api.js";

class PtFsPicker extends HTMLElement {
  open({ mode = "file", exts = [], startPath = null, filename = "" } = {}) {
    this.mode = mode;
    this.exts = exts;
    return new Promise((resolve) => {
      this._resolve = resolve;
      this._render(filename);
      this._navigate(startPath);
    });
  }

  _close(value) {
    if (this._onKey) {
      document.removeEventListener("keydown", this._onKey);
      this._onKey = null;
    }
    this.innerHTML = "";
    if (this._resolve) this._resolve(value);
    this._resolve = null;
  }

  _render(filename) {
    const dirsOnly = this.mode === "dir";
    const titleText = dirsOnly ? "Select a folder" : this.mode === "save" ? "Choose where to save" : "Select a file";
    this.innerHTML = `
      <div class="modal-backdrop">
        <div class="modal">
          <div class="modal-head">
            <span class="modal-title">${titleText}</span>
            <button class="btn btn-ghost" data-act="cancel">Cancel</button>
          </div>
          <div class="modal-path"></div>
          <div class="modal-list"></div>
          <div class="modal-foot">
            ${this.mode === "save" ? `<input type="text" class="save-name" placeholder="filename" />` : ""}
            ${dirsOnly || this.mode === "save" ? `<button class="btn btn-primary" data-act="choose"></button>` : ""}
          </div>
        </div>
      </div>`;
    this.querySelector('[data-act="cancel"]').addEventListener("click", () => this._close(null));
    this.querySelector(".modal-backdrop").addEventListener("click", (e) => {
      if (e.target.classList.contains("modal-backdrop")) this._close(null);
    });
    this._onKey = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        this._close(null);
      }
    };
    document.addEventListener("keydown", this._onKey);
    const choose = this.querySelector('[data-act="choose"]');
    if (choose) {
      choose.textContent = dirsOnly ? "Select this folder" : "Save here";
      choose.addEventListener("click", () => {
        if (this.mode === "save") {
          const name = this.querySelector(".save-name").value.trim();
          if (!name) return;
          this._close(`${this._cwd.replace(/\/$/, "")}/${name}`);
        } else {
          this._close(this._cwd);
        }
      });
    }
    if (filename) this.querySelector(".save-name").value = filename;
  }

  async _navigate(path) {
    let data;
    try {
      data = await api.fs(path, this.mode === "dir", this.mode === "save" ? [] : this.exts);
    } catch (e) {
      window.notify({ level: "error", title: "Browse failed", message: e.message });
      return;
    }
    this._cwd = data.cwd;
    this.querySelector(".modal-path").textContent = data.cwd;
    const list = this.querySelector(".modal-list");
    list.innerHTML = "";
    if (data.parent) list.appendChild(this._entry({ name: "..", path: data.parent, is_dir: true }, true));
    data.entries.forEach((e) => list.appendChild(this._entry(e, false)));
  }

  _entry(e, isParent) {
    const row = document.createElement("div");
    row.className = `fs-entry ${e.is_dir ? "dir" : "file"}`;
    row.innerHTML = `<span class="ic">${e.is_dir ? "📁" : "📄"}</span><span class="nm"></span>`;
    row.querySelector(".nm").textContent = e.name;
    row.addEventListener("click", () => {
      if (e.is_dir) this._navigate(e.path);
      else if (this.mode === "file") this._close(e.path);
      else if (this.mode === "save") this.querySelector(".save-name").value = e.name;
    });
    return row;
  }
}
customElements.define("pt-fs-picker", PtFsPicker);
