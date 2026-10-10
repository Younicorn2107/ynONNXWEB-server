#!/usr/bin/env node
/**
 * onnxweb-server serve [--port 8765] [--root DIR] [--no-prepare] [--offline] onnxweb-server prepare
 * <name> [--precision float32] [--root DIR] onnxweb-server cache [--sizes n,s] [--no-hub]
 * onnxweb-server doctor
 */

import { parseArgs } from "node:util";

import { modelsRoot } from "./paths.js";
import { prepareModel } from "./prepare.js";
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
      root: { type: "string" },
      precision: { type: "string", default: "float32" },
      "chunk-mb": { type: "string" },
      "no-prepare": { type: "boolean", default: false },
      offline: { type: "boolean", default: false },
      sizes: { type: "string", default: "n" },
      "no-hub": { type: "boolean", default: false },
    },
  });
  const root = values.root ?? modelsRoot();
  const runtime = new PythonRuntime({ logger });

  switch (command) {
    case "serve": {
      const server = createModelServer({
        models: { mount: "/models/", root, prepare: values["no-prepare"] ? false : runtime },
        mirror: { mount: "/hf/", offline: values.offline },
        logger,
      });
      const port = Number(values.port);
      server.listen(port, () => {
        console.log(`onnxweb-server  →  http://localhost:${port}/models/  (store ${root})`);
        console.log(`Hub mirror      →  http://localhost:${port}/hf/`);
        void listEntries(root).then((entries) => {
          for (const e of entries) console.log(`  ${e.key.padEnd(40)} ${String(e.sizeMb).padStart(6)} MB`);
        });
      });
      return new Promise(() => undefined);
    }
    case "prepare": {
      const name = positionals[0];
      if (!name) throw new Error("prepare needs a model name");
      const chunk = values["chunk-mb"] ? Number(values["chunk-mb"]) : undefined;
      const answer = await prepareModel(name, values.precision, { runtime, root, logger, ...(chunk ? { chunkMb: chunk } : {}) });
      console.log(answer.ready ? `ready ${answer.entry}` : `error ${answer.error}`);
      return answer.ready ? 0 : 1;
    }
    case "cache": {
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
