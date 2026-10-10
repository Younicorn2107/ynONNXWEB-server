/**
 * The routes inside Vite's dev server: one process, one origin, so no CORS and no CORP to get
 * wrong, and `npm run dev` is the whole instruction.
 */

import type { Logger } from "./http.js";
import { middlewares, type ServerOptions } from "./server.js";
import { cachedEntries } from "./store.js";
import { modelsRoot } from "./paths.js";

/** The slice of Vite's plugin and server types used here (Vite stays optional). */
interface ViteDevServer {
  middlewares: { use(handler: (...args: never[]) => unknown): void };
  config: { logger: Logger };
}

export interface VitePlugin {
  name: string;
  configureServer(server: ViteDevServer): void;
  configurePreviewServer(server: ViteDevServer): void;
}

export function onnxwebServer(options: Omit<ServerOptions, "cors" | "logger"> = {}): VitePlugin {
  const mount = (server: ViteDevServer) => {
    const logger = server.config.logger;
    for (const middleware of middlewares({ ...options, logger })) server.middlewares.use(middleware as never);
    if (options.models === false) return;
    const root = options.models?.root ?? modelsRoot();
    const at = options.models?.mount ?? "/models/";
    void cachedEntries(root).then((entries) => {
      const total = entries.reduce((sum, e) => sum + e.sizeMb, 0);
      logger.info(
        entries.length === 0
          ? `  onnxweb-server: the store ${root} is empty; models are made on demand (YNONNXWEB_MODELS moves it)`
          : `  onnxweb-server: ${entries.length} models at ${at} (${(total / 1000).toFixed(1)} GB, ${root}); missing ones are made on demand`,
      );
    });
  };
  return { name: "onnxweb-server", configureServer: mount, configurePreviewServer: mount };
}
