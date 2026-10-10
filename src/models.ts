/** The model routes, as connect middleware (Vite, Express, a bare `http` server): */

import { promises as fs } from "node:fs";
import { join, normalize, resolve, sep } from "node:path";

import { SAFE_NAME, sendJson, sendText, serveFile, type Logger, type Middleware } from "./http.js";
import { modelsRoot } from "./paths.js";
import { ensureModel } from "./prepare.js";
import { PythonRuntime } from "./python.js";
import { browserBlocker, cachedEntries, resolveEntry } from "./store.js";

/** A model name `prepare` accepts: a catalogue name, or a Hub `org/repo`. */
const MODEL_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*(\/[A-Za-z0-9][A-Za-z0-9._-]*)?$/;

export interface ModelsOptions {
  /** URL prefix (`/models/`). */
  mount?: string;
  /** The store's folder; `YNONNXWEB_MODELS` or `~/.cache/ynonnxweb/models`. */
  root?: string;
  /**
   * Make the models the store lacks (`prepare`): a Python runtime to do it with, `true` for the
   * default one, `false` to answer 404 instead.
   */
  prepare?: PythonRuntime | boolean;
  logger?: Logger | null;
}

export function modelsMiddleware(options: ModelsOptions = {}): Middleware {
  const mount = options.mount ?? "/models/";
  const root = options.root ?? modelsRoot();
  const logger = options.logger ?? null;
  const runtime = options.prepare === false ? null : options.prepare instanceof PythonRuntime ? options.prepare : new PythonRuntime({ logger });

  return async (request, response, next) => {
    if (!request.url?.startsWith(mount)) return next();
    const [rawPath = "", rawQuery = ""] = request.url.slice(mount.length).split("?");
    const path = decodeURIComponent(rawPath);

    if (path === "prepare") {
      const query = new URLSearchParams(rawQuery);
      const name = query.get("name") ?? "";
      const precision = query.get("computeType") ?? "float32";
      if (!MODEL_NAME.test(name) || !SAFE_NAME.test(precision)) return sendJson(response, 400, { ready: false, error: "bad name" });
      const answer = await ensureModel(name, precision, { runtime, root, logger });
      return sendJson(response, answer.ready ? 200 : 422, answer);
    }

    if (path === "catalog.json" || path === "") {
      response.setHeader("Content-Type", "application/json");
      response.setHeader("Cache-Control", "no-cache");
      response.end(JSON.stringify({ root, entries: await cachedEntries(root) }, null, 2));
      return;
    }

    const parts = path.split("/").filter(Boolean);
    if (parts.length !== 2) return sendText(response, 404, `expected ${mount}{entry}/{filename}`);
    const [entry, filename] = parts as [string, string];
    if (!SAFE_NAME.test(entry) || !SAFE_NAME.test(filename)) return sendText(response, 400, "bad name");

    const folder = await resolveEntry(root, entry);
    const blocker = await browserBlocker(root, folder);
    if (blocker) {
      return sendText(response, 409, `${folder} is quantized as ${blocker}, which no browser backend can run (WebGPU rejects its float zero points). Serve the web variant of this model.`);
    }

    const file = resolve(join(root, folder, filename));
    if (!normalize(file).startsWith(resolve(root) + sep)) return sendText(response, 403, "outside the store");
    const stat = await fs.stat(file).catch(() => null);
    if (!stat?.isFile()) return sendText(response, 404, `${folder}/${filename} is not in ${root}`);
    serveFile(request, response, file, stat);
  };
}
