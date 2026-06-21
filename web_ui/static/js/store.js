// Shared client state.
import { api } from "./api.js";

export const store = {
  config: {},
  metainfoClasses: [],
  paths: {},

  async load() {
    const d = await api.getConfig();
    this.config = d.config || {};
    this.metainfoClasses = d.metainfo_classes || [];
    this.paths = d.paths || {};
  },

  get(section, key) {
    return this.config[section] ? this.config[section][key] : undefined;
  },

  set(section, key, value) {
    if (!this.config[section]) this.config[section] = {};
    this.config[section][key] = value;
  },

  async refreshMetainfoClasses() {
    const path = this.get("general", "metainfo");
    if (!path) return;
    try {
      const r = await api.metainfoClasses(path);
      this.metainfoClasses = r.classes || [];
    } catch {
      /* keep previous classes */
    }
  },
};
