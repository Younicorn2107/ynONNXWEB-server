/**
 * The routes as one standalone HTTP server, for a deployment that is not a Vite dev server (or to
 * put the models on another origin than the page).
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";

import { checkRequest, type GuardOptions } from "./guard.js";
import { sendText, type Logger, type Middleware } from "./http.js";
import { corpusMiddleware, hfMirrorMiddleware, type MirrorOptions } from "./mirror.js";
import { modelsMiddleware, type ModelsOptions } from "./models.js";
import { staticMiddleware } from "./static.js";

export interface ServerOptions extends GuardOptions {
  /** The model routes (`/models/`); `false` for none. */
  models?: ModelsOptions | false;
  /** The Hub mirror (`/hf/`); `false` for none. */
  mirror?: MirrorOptions | false;
  /** The bench corpus (`/corpus/`); off unless given. */
  corpus?: { mount?: string; root?: string };
  /** Folders of runtime files (a 64-bit onnxruntime-web: `{ mount: "/ort-wasm64/", root }`). */
  statics?: { mount: string; root: string; missing?: string }[];
  logger?: Logger | null;
}

/** Every middleware the options ask for, in the order they are tried. */
export function middlewares(options: ServerOptions = {}): Middleware[] {
  const logger = options.logger ?? null;
  return [
    ...(options.models === false ? [] : [modelsMiddleware({ logger, ...options.models })]),
    ...(options.mirror === false ? [] : [hfMirrorMiddleware(options.mirror ?? {})]),
    ...(options.corpus ? [corpusMiddleware(options.corpus)] : []),
    ...(options.statics ?? []).map((s) => staticMiddleware(s)),
  ];
}

/**
 * A middleware that never rejects: an exception becomes a 500 (or cuts a response already
 * streaming) instead of an unhandled rejection that ends the process — connect does not await async
 * middleware.
 */
export function safe(middleware: Middleware, logger: Logger | null = null): Middleware {
  return (request, response, next) => {
    try {
      Promise.resolve(middleware(request, response, next)).catch((error: unknown) => {
        logger?.warn(`  onnxweb-server: ${request.method} ${request.url}: ${(error as Error).message}`);
        sendText(response, 500, "internal error");
      });
    } catch (error) {
      logger?.warn(`  onnxweb-server: ${request.method} ${request.url}: ${(error as Error).message}`);
      sendText(response, 500, "internal error");
    }
  };
}

/** The middlewares chained into one request handler; `fallback` when none answers. */
export function handler(list: readonly Middleware[], fallback: (request: IncomingMessage, response: ServerResponse) => void, logger: Logger | null = null) {
  const wrapped = list.map((m) => safe(m, logger));
  return (request: IncomingMessage, response: ServerResponse) => {
    let at = 0;
    const next = (error?: unknown): void => {
      if (error) return sendText(response, 500, "internal error");
      const middleware = wrapped[at++];
      if (!middleware) return fallback(request, response);
      middleware(request, response, next);
    };
    next();
  };
}

/** The guard as a middleware (for the Vite plugin, or any connect stack). */
export function guardMiddleware(options: GuardOptions = {}): Middleware {
  return (request, response, next) => {
    const verdict = checkRequest(request, options);
    if (!verdict.ok) return sendText(response, verdict.status, verdict.reason);
    next();
  };
}

export function createModelServer(options: ServerOptions = {}): Server {
  const chain = handler(middlewares(options), (_request, response) => sendText(response, 404, "not found"), options.logger ?? null);
  return createServer((request, response) => {
    const verdict = checkRequest(request, options);
    if (!verdict.ok) return sendText(response, verdict.status, verdict.reason);
    if (verdict.origin) {
      response.setHeader("Access-Control-Allow-Origin", verdict.origin);
      response.setHeader("Vary", "Origin");
      response.setHeader("Access-Control-Expose-Headers", "content-range, content-length, accept-ranges, etag");
      response.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
      if (request.method === "OPTIONS") {
        response.setHeader("Access-Control-Allow-Methods", "GET, HEAD, POST");
        response.setHeader("Access-Control-Allow-Headers", "range, content-type, authorization");
        response.setHeader("Access-Control-Max-Age", "600");
        response.statusCode = 204;
        return void response.end();
      }
    }
    chain(request, response);
  });
}
