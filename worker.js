// RADI web model — computation worker.
// Runs the Python engine (radi_core.py + carbconst.py) in Pyodide, off the page's main thread,
// so the page stays responsive while a column is solved.
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/";
importScripts(PYODIDE_URL + "pyodide.js");

let pyodide = null;
let ready = false;
const VERSION = new URL(self.location.href).searchParams.get("v") || "dev";

async function init() {
  const t0 = performance.now();
  postMessage({ type: "status", text: "Loading the Python runtime…" });
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  postMessage({ type: "status", text: "Loading numpy and scipy…" });
  await pyodide.loadPackage(["numpy", "scipy"]);
  postMessage({ type: "status", text: "Loading the RADI engine…" });
  for (const f of ["carbconst.py", "radi_core.py"]) {
    const resp = await fetch(f + "?v=" + VERSION);
    if (!resp.ok) throw new Error("could not fetch " + f + " (" + resp.status + ")");
    pyodide.FS.writeFile(f, await resp.text());
  }
  pyodide.runPython("import sys\nif '.' not in sys.path: sys.path.insert(0, '.')\nimport radi_core");
  ready = true;
  postMessage({ type: "ready", seconds: (performance.now() - t0) / 1000,
                version: pyodide.runPython("radi_core.VERSION") });
}

self.onmessage = (e) => {
  const m = e.data;
  if (m.type !== "run") return;
  if (!ready) { postMessage({ type: "error", id: m.id, message: "The engine is still loading." }); return; }
  self.radi_progress = (s) => postMessage({ type: "progress", id: m.id, info: JSON.parse(s) });
  try {
    pyodide.globals.set("REQ", JSON.stringify(m.request));
    const out = pyodide.runPython(
      "import json, radi_core\n" +
      "from js import radi_progress\n" +
      "radi_core.run(REQ, progress=lambda info: radi_progress(json.dumps(info)))");
    postMessage({ type: "result", id: m.id, result: out });
  } catch (err) {
    let msg = String(err && err.message ? err.message : err);
    // keep the last line of a Python traceback (the actual error)
    const lines = msg.trim().split("\n");
    const last = lines[lines.length - 1];
    postMessage({ type: "error", id: m.id, message: last.replace(/^\w*Error: /, ""), detail: msg });
  }
};

init().catch((err) => postMessage({ type: "fatal", message: "The engine could not load: " + err }));
