// Thin fetch helpers around the JSON API.

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await safeDetail(r)) || r.statusText);
  return r.json();
}

async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) {
    const detail = await safeDetail(r);
    const err = new Error(detail || r.statusText);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

async function safeDetail(r) {
  try {
    const j = await r.json();
    return j.detail || j.message || null;
  } catch {
    return null;
  }
}

export const api = {
  getConfig: () => jget("/api/config"),
  saveField: (field, config) => jpost("/api/config/field", { field, config }),
  validateField: (field, config) => jpost(`/api/validate/${field}`, { config }),
  metainfoClasses: (path) => jget(`/api/metainfo/classes?path=${encodeURIComponent(path)}`),
  resolve: (path, base) =>
    jget(`/api/resolve?path=${encodeURIComponent(path)}${base ? `&base=${encodeURIComponent(base)}` : ""}`),
  fs: (path, dirsOnly, exts) => {
    const p = new URLSearchParams();
    if (path) p.set("path", path);
    if (dirsOnly) p.set("dirs_only", "true");
    if (exts && exts.length) p.set("exts", exts.join(","));
    return jget(`/api/fs?${p.toString()}`);
  },
  getValidationConfig: (path) => jget(`/api/validation-config?path=${encodeURIComponent(path)}`),
  saveValidationConfig: (path, config) => jpost("/api/validation-config", { path, config }),
  validationTemplate: (strategy) => jget(`/api/validation-template?strategy=${encodeURIComponent(strategy)}`),
  getReidMetainfo: (path) => jget(`/api/reid-metainfo?path=${encodeURIComponent(path)}`),
  saveReidMetainfo: (path, identities, disabled_identities) =>
    jpost("/api/reid-metainfo", { path, identities, disabled_identities }),
  getTools: () => jget("/api/tools"),
  run: (tool, values) => jpost("/api/run", { tool, values }),
  getRun: () => jget("/api/run"),
  stopRun: () => jpost("/api/run/stop", {}),
};
