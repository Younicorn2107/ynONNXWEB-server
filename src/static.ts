/**
 * A folder of runtime files served at a mount: a second onnxruntime-web, for instance the 64-bit
 * WebAssembly build (`.mjs`, `.wasm`) a page imports for a model too large for the 32-bit one's 4
 * GB.
 */

import { promises as fs } from "node:fs";

import { SAFE_NAME, contained, decodePath, sendText, serveFile, type Middleware } from "./http.js";

export function staticMiddleware(options: { mount: string; root: string; missing?: string }): Middleware {
  return async (request, response, next) => {
    const url = (request.url ?? "").split("?")[0] ?? "";
    if (!url.startsWith(options.mount)) return next();
    if (request.method !== "GET" && request.method !== "HEAD") return sendText(response, 405, "read only");
    const name = decodePath(url.slice(options.mount.length));
    if (name === null || !SAFE_NAME.test(name)) return sendText(response, 400, "bad path");
    const file = contained(options.root, name);
    const stat = file ? await fs.stat(file).catch(() => null) : null;
    if (!file || !stat?.isFile()) return sendText(response, 404, `${name}: not found${options.missing ? ` (${options.missing})` : ""}`);
    response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
    serveFile(request, response, file, stat);
  };
}
