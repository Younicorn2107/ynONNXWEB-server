/**
 * A folder of runtime files served at a mount: a second onnxruntime-web, for instance the 64-bit
 * WebAssembly build (`.mjs`, `.wasm`) a page imports for a model too large for the 32-bit one's 4
 * GB.
 */

import { promises as fs } from "node:fs";
import { join, normalize, resolve, sep } from "node:path";

import { sendText, serveFile, type Middleware } from "./http.js";

export function staticMiddleware(options: { mount: string; root: string; missing?: string }): Middleware {
  const root = resolve(options.root);
  return async (request, response, next) => {
    const url = (request.url ?? "").split("?")[0] ?? "";
    if (!url.startsWith(options.mount)) return next();
    const name = normalize(decodeURIComponent(url.slice(options.mount.length)));
    const file = resolve(join(root, name));
    if (!file.startsWith(root + sep)) return sendText(response, 400, "bad path");
    const stat = await fs.stat(file).catch(() => null);
    if (!stat?.isFile()) return sendText(response, 404, `${name}: not in ${root}${options.missing ? ` (${options.missing})` : ""}`);
    response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
    serveFile(request, response, file, stat);
  };
}
