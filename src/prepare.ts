/** Models made on demand. */

import { promises as fs } from "node:fs";
import { join } from "node:path";

import type { Logger } from "./http.js";
import { type PythonRuntime, type RequirementSet } from "./python.js";
import { entryKey, resolveEntry } from "./store.js";

export type Prepared = { ready: true; entry: string } | { ready: false; error: string };

/**
 * A model name `prepare` accepts: a catalogue name, or a Hub `org/repo` (each part starting with a
 * letter or digit).
 */
export const MODEL_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*(\/[A-Za-z0-9][A-Za-z0-9._-]*)?$/;

/** The precisions the export script knows (never a value argparse could read as an option). */
export const PRECISIONS = ["float32", "float16", "int8", "int4", "int4_float16"] as const;

export function validPrecision(value: string): boolean {
  return (PRECISIONS as readonly string[]).includes(value);
}

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
  /** Killed after this long (ms); 2 hours by default. */
  timeoutMs?: number;
}

export const MAX_EXPORTS = 1;
export const MAX_QUEUED = 4;
export const EXPORT_TIMEOUT_MS = 2 * 60 * 60 * 1000;

const running = new Map<string, Promise<Prepared>>();
let active = 0;
const waiting: (() => void)[] = [];

/** A slot among `MAX_EXPORTS`, or null when the queue is full. */
function acquire(): Promise<() => void> | null {
  const release = () => {
    active -= 1;
    waiting.shift()?.();
  };
  if (active < MAX_EXPORTS) {
    active += 1;
    return Promise.resolve(release);
  }
  if (waiting.length >= MAX_QUEUED) return null;
  return new Promise((resolve) =>
    waiting.push(() => {
      active += 1;
      resolve(release);
    }),
  );
}

/** `name` at `precision` made into the store; the entry's folder name when it is ready. */
export function prepareModel(name: string, precision: string, options: PrepareOptions): Promise<Prepared> {
  if (!MODEL_NAME.test(name)) return Promise.resolve({ ready: false, error: "bad model name" });
  if (!validPrecision(precision)) return Promise.resolve({ ready: false, error: `precision must be one of ${PRECISIONS.join(", ")}` });
  const key = `${options.root}|${name}|${precision}`;
  const existing = running.get(key);
  if (existing) return existing;
  const slot = acquire();
  if (!slot) return Promise.resolve({ ready: false, error: "the server is busy exporting other models; try again later" });
  const chunk = options.chunkMb !== undefined && Number.isInteger(options.chunkMb) && options.chunkMb > 0 ? ["--chunk-mb", String(options.chunkMb)] : [];

  const args = ["--precision", precision, ...chunk, "--", name];
  const run = slot
    .then(async (release) => {
      try {
        const { code, lines } = await options.runtime.run("prepare_model.py", args, {
          needs: requirementsFor(name),
          env: { YNONNXWEB_MODELS: options.root },
          signal: AbortSignal.timeout(options.timeoutMs ?? EXPORT_TIMEOUT_MS),
          onLine: (line) => {
            if (/^(step|ready|error) /.test(line)) options.logger?.info(`  onnxweb-server prepare ${name}: ${line}`);
          },
        });
        const last = [...lines].reverse().find((l) => /^(ready|error) /.test(l));
        if (code === 0 && last?.startsWith("ready ")) return { ready: true, entry: last.slice(6) } as Prepared;
        return { ready: false, error: last?.slice(6) ?? `export failed (exit ${code})` } as Prepared;
      } finally {
        release();
      }
    })
    .catch((error: unknown): Prepared => {
      const aborted = (error as Error).name === "AbortError" || (error as Error).name === "TimeoutError";
      options.logger?.warn(`  onnxweb-server prepare ${name}: ${(error as Error).message}`);
      return { ready: false, error: aborted ? "the export took too long and was stopped" : "the export could not run (see the server log)" };
    })
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
  if (!MODEL_NAME.test(name) || !validPrecision(precision)) return Promise.resolve({ ready: false, error: "bad model name or precision" });
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
