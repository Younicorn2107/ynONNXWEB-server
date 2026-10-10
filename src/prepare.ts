/** Models made on demand. */

import { promises as fs } from "node:fs";
import { join } from "node:path";

import type { Logger } from "./http.js";
import { type PythonRuntime, type RequirementSet } from "./python.js";
import { entryKey, resolveEntry } from "./store.js";

export type Prepared = { ready: true; entry: string } | { ready: false; error: string };

/** The Python requirement set a recipe needs, by model name (the script holds the recipes). */
export function requirementsFor(name: string): RequirementSet[] {
  if (/^yolo26[nslmx](-|$)/.test(name)) return ["yolo"];
  if (name.startsWith("onnx-community/")) return ["hub"];
  return ["export"];
}

export interface PrepareOptions {
  runtime: PythonRuntime;
  /** The store the entry is written into. */
  root: string;
  logger?: Logger | null;
  /** The largest weight file written, in MB (1024). */
  chunkMb?: number;
}

const running = new Map<string, Promise<Prepared>>();

/** `name` at `precision` made into the store; the entry's folder name when it is ready. */
export function prepareModel(name: string, precision: string, options: PrepareOptions): Promise<Prepared> {
  const key = `${options.root}|${name}|${precision}`;
  const existing = running.get(key);
  if (existing) return existing;
  const args = [name, "--precision", precision, ...(options.chunkMb ? ["--chunk-mb", String(options.chunkMb)] : [])];
  const run = options.runtime
    .run("prepare_model.py", args, {
      needs: requirementsFor(name),
      env: { YNONNXWEB_MODELS: options.root },
      onLine: (line) => {
        if (/^(step|ready|error) /.test(line)) options.logger?.info(`  onnxweb-server prepare ${name}: ${line}`);
      },
    })
    .then(({ code, lines }): Prepared => {
      const last = [...lines].reverse().find((l) => /^(ready|error) /.test(l));
      if (code === 0 && last?.startsWith("ready ")) return { ready: true, entry: last.slice(6) };
      return { ready: false, error: last?.slice(6) ?? (lines.slice(-3).join(" | ") || `exit ${code}`) };
    })
    .catch((error: unknown): Prepared => ({ ready: false, error: (error as Error).message }))
    .finally(() => running.delete(key));
  running.set(key, run);
  return run;
}

/**
 * The entry for `name` at `precision`: the one in the store (a blend included), else made now when
 * a runtime is given. Checked and made under one lock per model, so a request arriving as another
 * one's export ends finds the entry instead of exporting it again.
 */
export function ensureModel(name: string, precision: string, options: Omit<PrepareOptions, "runtime"> & { runtime: PythonRuntime | null }): Promise<Prepared> {
  const key = `${options.root}|${name}|${precision}|ensure`;
  const existing = running.get(key);
  if (existing) return existing;
  const run = (async (): Promise<Prepared> => {
    const folder = await resolveEntry(options.root, entryKey(name, precision));
    if (await fs.stat(join(options.root, folder, "ynonnx_manifest.json")).catch(() => null)) return { ready: true, entry: folder };
    if (!options.runtime) return { ready: false, error: `${folder} is not in the store, and this server does not make models` };
    return prepareModel(name, precision, { ...options, runtime: options.runtime });
  })().finally(() => running.delete(key));
  running.set(key, run);
  return run;
}
