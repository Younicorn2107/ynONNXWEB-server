/**
 * The routes inside Vite's dev server: one process, one origin, so no CORS and no CORP to get
 * wrong, and `npm run dev` is the whole instruction.
 */

import type { GuardOptions } from "./guard.js";
import type { Logger, Middleware } from "./http.js";
import { modelsRoot } from "./paths.js";
import { guardMiddleware, middlewares, safe, type ServerOptions } from "./server.js";
import { cachedEntries } from "./store.js";

/** The slice of Vite's plugin and server types used here (Vite stays optional). */
interface ViteDevServer {
  middlewares: { use(handler: (...args: never[]) => unknown): void };
  config: { logger: Logger; server?: { allowedHosts?: unknown } };
}

export interface VitePlugin {
  name: string;
  configureServer(server: ViteDevServer): void;
  configurePreviewServer(server: ViteDevServer): void;
}

/** The mounts the options define: the guard only looks at requests for them. */
function mountsOf(options: ServerOptions): string[] {
  return [
    ...(options.models === false ? [] : [options.models?.mount ?? "/models/"]),
    ...(options.mirror === false ? [] : [options.mirror?.mount ?? "/hf/"]),
    ...(options.corpus ? [options.corpus.mount ?? "/corpus/"] : []),
    ...(options.statics ?? []).map((s) => s.mount),
  ];
}

export function onnxwebServer(options: Omit<ServerOptions, "logger"> = {}): VitePlugin {
  const mount = (server: ViteDevServer) => {
    const logger = server.config.logger;

    const viteHosts = server.config.server?.allowedHosts;
    const guard: GuardOptions = {
      ...options,
      ...(Array.isArray(viteHosts) ? { allowedHosts: [...(options.allowedHosts ?? []), ...viteHosts.map(String)] } : {}),
      ...(viteHosts === true ? { anyHost: true } : {}),
    };
    const mounts = mountsOf(options);
    const check = guardMiddleware(guard);
    const guarded: Middleware = (request, response, next) => {
      if (!mounts.some((m) => request.url?.startsWith(m))) return next();
      check(request, response, next);
    };
    server.middlewares.use(safe(guarded, logger) as never);
    for (const middleware of middlewares({ ...options, logger })) server.middlewares.use(safe(middleware, logger) as never);
    if (options.models === false) return;
    const root = options.models?.root ?? modelsRoot();
    const at = options.models?.mount ?? "/models/";
    void cachedEntries(root).then((entries) => {
      const total = entries.reduce((sum, e) => sum + e.sizeMb, 0);
      const making = options.models && options.models.prepare ? "; missing ones are made on demand" : "";
      logger.info(
        entries.length === 0
          ? `  onnxweb-server: the store is empty (YNONNXWEB_MODELS moves it)${making}`
          : `  onnxweb-server: ${entries.length} models at ${at} (${(total / 1000).toFixed(1)} GB)${making}`,
      );
    });
  };
  return { name: "onnxweb-server", configureServer: mount, configurePreviewServer: mount };
}
