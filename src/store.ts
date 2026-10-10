/**
 * The model store, read: its entries (`<name>_<precision>/` with a `ynonnx_manifest.json`), what
 * each weighs, and which folder answers a request.
 */

import { promises as fs } from "node:fs";
import { join } from "node:path";

import { modelsRoot } from "./paths.js";

interface ManifestGraph {
  filename: string;
  role: string;
  size_mb?: number;
  external_data?: string[];
}

export interface StoreEntry {
  /** The folder name, `<name>_<precision>`. */
  key: string;
  name: string;
  computeType: string;
  /** What a run actually downloads, measured on disk. */
  sizeMb: number;
  graphs: string[];
  vocab: number | null;
  architecture: string | null;
  /** Some weights sit in files beside the graphs. */
  external: boolean;
}

/**
 * The graphs a run actually fetches. An entry often ships a merged decoder and the split pair it
 * replaces — three files where two phases are needed — and summing them all reports a download
 * twice the real size.
 */
export function neededGraphs<T extends { role: string }>(graphs: readonly T[]): T[] {
  const byRole = new Map(graphs.map((g) => [g.role, g]));
  const merged = byRole.get("decoder_merged");
  const decoder = merged ? [merged] : [byRole.get("decoder"), byRole.get("decoder_with_past")];
  const pair = [byRole.get("encoder"), ...decoder].filter((g): g is T => g !== undefined);

  return pair.length > 0 ? pair : graphs.filter((g) => g.role !== "decoder" && g.role !== "decoder_with_past");
}

/**
 * An entry's name and precision. Split on the FIRST underscore: a precision can carry one of its
 * own (`int8_float16`, `float8_e4m3`).
 */
export function splitKey(key: string): { name: string; computeType: string } {
  const at = key.indexOf("_");
  return at > 0 ? { name: key.slice(0, at), computeType: key.slice(at + 1) } : { name: key, computeType: "float32" };
}

export async function listEntries(root = modelsRoot()): Promise<StoreEntry[]> {
  const names = await fs.readdir(root).catch(() => [] as string[]);
  const entries: StoreEntry[] = [];
  for (const key of names) {
    const directory = join(root, key);
    let raw: { version?: number; graphs?: ManifestGraph[]; logits_vocab_size?: number; architecture?: string };
    try {
      raw = JSON.parse(await fs.readFile(join(directory, "ynonnx_manifest.json"), "utf-8")) as typeof raw;
    } catch {
      continue;
    }
    const needed = neededGraphs(raw.graphs ?? []);

    const recorded = (raw.version ?? 0) >= 2;
    let bytes = 0;
    let external = false;
    for (const graph of needed) {
      const stat = await fs.stat(join(directory, graph.filename)).catch(() => null);
      bytes += stat ? stat.size : Math.round((graph.size_mb ?? 0) * 1e6);
      const sidecars = recorded ? (graph.external_data ?? []) : ["_data", ".data"].map((suffix) => graph.filename + suffix);
      for (const name of sidecars) {
        const sidecar = await fs.stat(join(directory, name)).catch(() => null);
        if (sidecar) {
          bytes += sidecar.size;
          external = true;
        }
      }
    }
    entries.push({
      key,
      ...splitKey(key),
      sizeMb: Math.round(bytes / 1e6),
      graphs: needed.map((g) => g.role),
      vocab: raw.logits_vocab_size ?? null,
      architecture: raw.architecture ?? null,
      external,
    });
  }
  return entries.sort((a, b) => a.sizeMb - b.sizeMb || a.key.localeCompare(b.key));
}

let listed: { root: string; mtimeMs: number; entries: Promise<StoreEntry[]> } = { root: "", mtimeMs: -1, entries: Promise.resolve([]) };

export async function cachedEntries(root = modelsRoot()): Promise<StoreEntry[]> {
  const stat = await fs.stat(root).catch(() => null);
  const mtimeMs = stat ? stat.mtimeMs : 0;
  if (listed.root !== root || listed.mtimeMs !== mtimeMs) listed = { root, mtimeMs, entries: listEntries(root) };
  return listed.entries;
}

/**
 * Quantizations no browser backend runs. `algo-int4-hqq` writes float zero points, flat; WebGPU's
 * `MatMulNBits` wants them as packed integers and refuses (`zeroPoints input size error`). Packing
 * them was measured and destroys the model: 1.46 % word error rate becomes 122.68 %.
 */
const NOT_RUNNABLE_IN_BROWSER = ["algo-int4-hqq"];

/**
 * The variant of an entry no browser can run, or null: a store may be the Python libraries' own
 * cache, chosen for native ONNX Runtime. Refused before the download, with the cause named, rather
 * than inside a shader after it.
 */
export async function browserBlocker(root: string, folder: string): Promise<string | null> {
  for (const file of ["ynonnx_manifest.json", "onnx_manifest.json"]) {
    try {
      const raw = await fs.readFile(join(root, folder, file), "utf8");
      const key = String((JSON.parse(raw) as { model_key?: unknown }).model_key ?? "");
      return NOT_RUNNABLE_IN_BROWSER.find((variant) => key.includes(variant)) ?? null;
    } catch {
      // Not fatal: carry on.
    }
  }
  return null;
}

/**
 * The folder that serves `entry`: itself, or the one blend a request for that precision produced.
 * The Python side does not always store a precision under its own name: `small` asked at int8 is
 * kept as `small_mix_decoder-int4_encoder-int8`, a rule that lives in the Python catalogue and that
 * nothing publishes. Only when exactly one folder matches: two would mean guessing, and a wrong
 * guess loads other weights under the name asked for.
 */
export async function resolveEntry(root: string, entry: string): Promise<string> {
  if (await fs.stat(join(root, entry)).catch(() => null)) return entry;
  const { name, computeType } = splitKey(entry);
  if (name === entry) return entry;
  const names = await fs.readdir(root).catch(() => [] as string[]);
  const candidates = names.filter((n) => n.startsWith(`${name}_mix_`) && n.includes(`-${computeType}`));
  return candidates.length === 1 ? candidates[0]! : entry;
}

/**
 * The folder name a model and a precision are stored under (`/` and `_` in a name become `--` and
 * `-`).
 */
export function entryKey(name: string, precision: string): string {
  return `${name.replaceAll("/", "--").replaceAll("_", "-")}_${precision}`;
}
