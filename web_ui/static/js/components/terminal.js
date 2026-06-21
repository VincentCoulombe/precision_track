// Embedded streaming terminal. Connects to /api/run/stream (SSE) and renders two
// event kinds: `line` (a committed line) and `live` (the current in-progress line,
// e.g. a progress bar, updated in place). A single "live line" element is reused
// for `live` events and finalized when the next `line` arrives, so a progress bar
// shows as one updating line instead of stacking.

class PtTerminal extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `<div class="terminal" tabindex="0"></div>`;
    this.out = this.querySelector(".terminal");
    this._live = null; // the live-line element, if any
  }

  clear() {
    if (this.out) this.out.innerHTML = "";
    this._live = null;
  }

  _styleFor(div, text) {
    if (text.startsWith("$ ")) div.className = "cmd";
    else if (text.startsWith("[web_ui]")) div.className = "sys";
    else div.className = "";
  }

  _scroll() {
    const atBottom = this.out.scrollHeight - this.out.scrollTop - this.out.clientHeight < 60;
    if (atBottom) this.out.scrollTop = this.out.scrollHeight;
  }

  // A finished line: finalize the live element if present, else append a new one.
  commitLine(text) {
    const div = this._live || document.createElement("div");
    this._styleFor(div, text);
    div.textContent = text;
    if (!this._live) this.out.appendChild(div);
    this._live = null;
    this._scroll();
  }

  // The current in-progress line: create/update the live element in place.
  liveLine(text) {
    if (!this._live) {
      this._live = document.createElement("div");
      this.out.appendChild(this._live);
    }
    this._styleFor(this._live, text);
    this._live.textContent = text;
    this._scroll();
  }

  // Open the live SSE stream. The stream is the single source of truth: it
  // first replays the committed tail (as `line` events) + the current `live`
  // line, then streams updates — so we must NOT pre-render the tail here, or it
  // would be drawn twice. onEnd(status) fires when the process finishes.
  connect(onEnd) {
    this.disconnect();
    this.clear();

    const es = new EventSource("/api/run/stream");
    this._es = es;
    es.addEventListener("line", (e) => this.commitLine(e.data));
    es.addEventListener("live", (e) => this.liveLine(e.data));
    es.addEventListener("end", (e) => {
      this.disconnect();
      if (onEnd) onEnd(e.data);
    });
    es.onerror = () => {
      // Stream closed (process ended or server gone). Let the poller reconcile.
      this.disconnect();
      if (onEnd) onEnd(null);
    };
  }

  disconnect() {
    if (this._es) {
      this._es.close();
      this._es = null;
    }
  }
}
customElements.define("pt-terminal", PtTerminal);
