/** The model routes, as connect middleware (Vite, Express, a bare `http` server): */

import { promises as fs } from "node:fs";

import { BenchQueue } from "./bench.js";
import { BodyError, SAFE_NAME, contained, decodePath, readJson, sendJson, sendText, serveFile, type Logger, type Middleware } from "./http.js";
import { modelsRoot } from "./paths.js";
import { MODEL_NAME, ensureModel, validPrecision } from "./prepare.js";
import { PythonRuntime } from "./python.js";
import { browserBlocker, cachedEntries, resolveEntry } from "./store.js";

export interface ModelsOptions {
  /** URL prefix (`/models/`). */
  mount?: string;
  /** The store's folder; `YNONNXWEB_MODELS` or `~/.cache/ynonnxweb/models`. */
  root?: string;
  /**
   * Make the models the store lacks (`prepare`): a Python runtime to do it with, `true` for the
   * default one, `false` (the default) to refuse.
   */
  prepare?: PythonRuntime | boolean;
  /** Serve the bench queue under `{mount}bench/` (off by default: benches only). */
  bench?: boolean;
  logger?: Logger | null;
}

export function modelsMiddleware(options: ModelsOptions = {}): Middleware {
  const mount = options.mount ?? "/models/";
  const root = options.root ?? modelsRoot();
  const logger = options.logger ?? null;
  const runtime = options.prepare instanceof PythonRuntime ? options.prepare : options.prepare === true ? new PythonRuntime({ logger }) : null;
  const bench = options.bench === true ? new BenchQueue() : null;

  return async (request, response, next) => {
    if (!request.url?.startsWith(mount)) return next();
    const [rawPath = ""] = request.url.slice(mount.length).split("?");
    const path = decodePath(rawPath);
    if (path === null) return sendText(response, 400, "bad path");
    const method = request.method ?? "GET";

    if (path.startsWith("bench/")) {
      if (!bench) return sendText(response, 404, "no bench queue here");
      return bench.handle(request, response, path.slice("bench/".length)).catch((error: unknown) => sendJson(response, error instanceof BodyError ? error.status : 400, { error: (error as Error).message }));
    }

    if (path === "prepare") {
      if (method !== "POST") return sendJson(response, 405, { ready: false, error: "prepare is a POST with a JSON body" });
      let body: unknown;
      try {
        body = await readJson(request, 4096);
      } catch (error) {
        return sendJson(response, error instanceof BodyError ? error.status : 400, { ready: false, error: (error as Error).message });
      }
      const { name, computeType = "float32" } = (body ?? {}) as { name?: unknown; computeType?: unknown };
      if (typeof name !== "string" || !MODEL_NAME.test(name) || typeof computeType !== "string" || !validPrecision(computeType)) {
        return sendJson(response, 400, { ready: false, error: "bad name or computeType" });
      }
      const answer = await ensureModel(name, computeType, { runtime, root, logger });
      return sendJson(response, answer.ready ? 200 : 422, answer);
    }

    if (method !== "GET" && method !== "HEAD") return sendText(response, 405, "read only");

    if (path === "catalog.json" || path === "") {
      response.setHeader("Content-Type", "application/json");
      response.setHeader("Cache-Control", "no-cache");

      const entries = (await cachedEntries(root)).filter((e) => SAFE_NAME.test(e.key));
      response.end(JSON.stringify({ entries }, null, 2));
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

    const file = contained(root, folder, filename);
    if (!file) return sendText(response, 403, "outside the store");
    const stat = await fs.stat(file).catch(() => null);
    if (!stat?.isFile()) return sendText(response, 404, `${folder}/${filename} is not in the store`);
    serveFile(request, response, file, stat);
  };
}
