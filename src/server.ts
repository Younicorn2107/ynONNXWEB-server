/**
 * The routes as one standalone HTTP server, for a deployment that is not a Vite dev server (or to
 * put the models on another origin than the page).
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";

import { sendText, type Logger, type Middleware } from "./http.js";
import { hfMirrorMiddleware, type MirrorOptions } from "./mirror.js";
import { modelsMiddleware, type ModelsOptions } from "./models.js";
import { staticMiddleware } from "./static.js";

export interface ServerOptions {
  /** The model routes (`/models/`); `false` for none. */
  models?: ModelsOptions | false;
  /** The Hub mirror (`/hf/`); `false` for none. */
  mirror?: MirrorOptions | false;
  /** Folders of runtime files (a 64-bit onnxruntime-web: `{ mount: "/ort-wasm64/", root }`). */
  statics?: { mount: string; root: string; missing?: string }[];
  /** Origins allowed to read (`*` by default); `false` to send no CORS headers. */
  cors?: string | false;
  logger?: Logger | null;
}

/** Every middleware the options ask for, in the order they are tried. */
export function middlewares(options: ServerOptions = {}): Middleware[] {
  const logger = options.logger ?? null;
  return [
    ...(options.models === false ? [] : [modelsMiddleware({ logger, ...options.models })]),
    ...(options.mirror === false ? [] : [hfMirrorMiddleware(options.mirror ?? {})]),
    ...(options.statics ?? []).map((s) => staticMiddleware(s)),
  ];
}

/** The middlewares chained into one request handler; `fallback` when none answers. */
export function handler(list: readonly Middleware[], fallback: (request: IncomingMessage, response: ServerResponse) => void) {
  return (request: IncomingMessage, response: ServerResponse) => {
    let at = 0;
    const next = (error?: unknown): void => {
      if (error) return sendText(response, 500, String(error));
      const middleware = list[at++];
      if (!middleware) return fallback(request, response);
      Promise.resolve(middleware(request, response, next)).catch((e: unknown) => sendText(response, 500, String(e)));
    };
    next();
  };
}

export function createModelServer(options: ServerOptions = {}): Server {
  const cors = options.cors ?? "*";
  const chain = handler(middlewares(options), (_request, response) => sendText(response, 404, "not found"));
  return createServer((request, response) => {
    if (cors !== false) {
      response.setHeader("Access-Control-Allow-Origin", cors);
      response.setHeader("Access-Control-Expose-Headers", "content-range, content-length, accept-ranges");
      response.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
      if (request.method === "OPTIONS") {
        response.setHeader("Access-Control-Allow-Headers", "range, content-type");
        response.statusCode = 204;
        return void response.end();
      }
    }
    chain(request, response);
  });
}
