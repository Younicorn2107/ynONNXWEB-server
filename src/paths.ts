/**
 * Where things live on the server's disk. Each one overridable by an environment variable, so a
 * deployment moves them without code.
 */

import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const cache = () => join(homedir(), ".cache", "ynonnxweb");

/**
 * The model store served to the browser: `<name>_<precision>/` folders, each with its graphs and a
 * `ynonnx_manifest.json`. The browser's own store, not the Python libraries' cache: those hold
 * variants chosen for native ONNX Runtime, some of which no browser backend runs.
 */
export function modelsRoot(): string {
  return process.env["YNONNXWEB_MODELS"] ?? join(cache(), "models");
}

/** The machine's Hugging Face cache (what Python already downloaded). */
export function hubCache(): string {
  return process.env["HF_HUB_CACHE"] ?? join(process.env["HF_HOME"] ?? join(homedir(), ".cache", "huggingface"), "hub");
}

/** Files the mirror fetched from the Hub itself. */
export function mirrorRoot(): string {
  return process.env["YNONNXWEB_HF_MIRROR"] ?? join(cache(), "hf-mirror");
}

/** The virtual environment this package installs its Python dependencies into. */
export function pythonEnvRoot(): string {
  return process.env["YNONNXWEB_PYTHON_ENV"] ?? join(cache(), "python");
}

/** The Python scripts shipped with this package (`python/` beside `dist/`). */
export function scriptsRoot(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "python");
}
