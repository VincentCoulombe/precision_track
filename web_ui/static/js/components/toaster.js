// Global notification host. Exposes window.notify({level, title, message}).

class PtToaster extends HTMLElement {
  connectedCallback() {
    window.notify = (opts) => this.show(opts);
  }

  show({ level = "success", title = "", message = "" }) {
    const el = document.createElement("div");
    el.className = `toast ${level}`;
    el.innerHTML = `
      <div class="toast-title">
        <span></span>
        <button class="toast-close" aria-label="Dismiss">&times;</button>
      </div>
      ${message ? `<div class="toast-msg"></div>` : ""}`;
    el.querySelector(".toast-title span").textContent =
      title || { error: "Error", warning: "Warning", success: "Done" }[level] || "";
    if (message) el.querySelector(".toast-msg").textContent = message;

    const close = () => {
      el.style.opacity = "0";
      setTimeout(() => el.remove(), 180);
    };
    el.querySelector(".toast-close").addEventListener("click", close);
    this.appendChild(el);

    const ttl = level === "error" ? 9000 : level === "warning" ? 7000 : 4000;
    setTimeout(close, ttl);
  }
}
customElements.define("pt-toaster", PtToaster);
