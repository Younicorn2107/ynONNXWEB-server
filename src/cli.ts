#!/usr/bin/env node
/**
 * onnxweb-server serve [--port 8765] [--host 127.0.0.1] [--token T] [--allow-origin URL]...
 * [--allow-host NAME]... [--root DIR] [--prepare] [--bench] [--offline] [--mirror-repo
 * org/repo|org/*]... onnxweb-server prepare <name> [--precision float32] [--root DIR]
 * onnxweb-server cache [--sizes n,s] [--no-hub] onnxweb-server doctor
 */

import { randomBytes } from "node:crypto";
import { parseArgs } from "node:util";

import { isLoopback } from "./guard.js";
import { modelsRoot } from "./paths.js";
import { PRECISIONS, prepareModel, validPrecision } from "./prepare.js";
import { PythonRuntime } from "./python.js";
import { createModelServer } from "./server.js";
import { listEntries } from "./store.js";

const logger = { info: (m: string) => console.log(m), warn: (m: string) => console.warn(m) };

async function main(): Promise<number> {
  const [command, ...rest] = process.argv.slice(2);
  const { values, positionals } = parseArgs({
    args: rest,
    allowPositionals: true,
    options: {
      port: { type: "string", default: process.env["PORT"] ?? "8765" },
      host: { type: "string", default: "127.0.0.1" },
      token: { type: "string" },
      "allow-origin": { type: "string", multiple: true },
      "allow-host": { type: "string", multiple: true },
      "mirror-repo": { type: "string", multiple: true },
      root: { type: "string" },
      precision: { type: "string", default: "float32" },
      "chunk-mb": { type: "string" },
      prepare: { type: "boolean", default: false },
      bench: { type: "boolean", default: false },
      offline: { type: "boolean", default: false },
      sizes: { type: "string", default: "n" },
      "no-hub": { type: "boolean", default: false },
    },
  });
  const root = values.root ?? modelsRoot();
  const runtime = new PythonRuntime({ logger });

  switch (command) {
    case "serve": {
      const port = Number(values.port);
      if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error(`--port must be 1 to 65535 (got ${values.port})`);
      const host = values.host;
      let token = values.token ?? process.env["ONNXWEB_SERVER_TOKEN"] ?? null;
      if (!isLoopback(host) && !token) {
        token = randomBytes(24).toString("base64url");
        console.log(`onnxweb-server listens on ${host}: requests need "Authorization: Bearer ${token}"`);
      }
      const server = createModelServer({
        models: { mount: "/models/", root, prepare: values.prepare ? runtime : false, bench: values.bench },
        mirror: { mount: "/hf/", offline: values.offline, ...(values["mirror-repo"] ? { repos: values["mirror-repo"] } : {}) },
        allowedOrigins: values["allow-origin"] ?? [],
        allowedHosts: [...(values["allow-host"] ?? []), ...(isLoopback(host) ? [] : [host])],
        token,
        logger,
      });
      server.listen(port, host, () => {
        const shown = host.includes(":") ? `[${host}]` : host;
        console.log(`onnxweb-server  →  http://${shown}:${port}/models/${values.prepare ? "  (models made on demand)" : ""}`);
        console.log(`Hub mirror      →  http://${shown}:${port}/hf/`);
        void listEntries(root).then((entries) => {
          for (const e of entries) console.log(`  ${e.key.padEnd(40)} ${String(e.sizeMb).padStart(6)} MB`);
        });
      });
      return new Promise(() => undefined);
    }
    case "prepare": {
      const name = positionals[0];
      if (!name) throw new Error("prepare needs a model name");
      if (!validPrecision(values.precision)) throw new Error(`--precision must be one of ${PRECISIONS.join(", ")}`);
      const chunk = values["chunk-mb"] ? Number(values["chunk-mb"]) : undefined;
      const answer = await prepareModel(name, values.precision, { runtime, root, logger, ...(chunk ? { chunkMb: chunk } : {}) });
      console.log(answer.ready ? `ready ${answer.entry}` : `error ${answer.error}`);
      return answer.ready ? 0 : 1;
    }
    case "cache": {
      if (!/^[nslmx](,[nslmx])*$/.test(values.sizes)) throw new Error("--sizes is a comma-separated list of n, s, m, l, x");
      const args = ["--sizes", values.sizes, ...(values["no-hub"] ? ["--no-hub"] : [])];
      const result = await runtime.run("cache_models.py", args, { needs: ["yolo"], env: { YNONNXWEB_MODELS: root }, onLine: (l) => console.log(l) });
      return result.code;
    }
    case "doctor": {
      const report = await runtime.describe();
      console.log(JSON.stringify({ ...report, store: root }, null, 2));
      return 0;
    }
    default:
      console.log("usage: onnxweb-server serve | prepare <name> | cache | doctor  (see the README)");
      return command ? 2 : 0;
  }
}

main().then(
  (code) => process.exit(code),
  (error: unknown) => {
    console.error(`onnxweb-server: ${(error as Error).message}`);
    process.exit(1);
  },
);
