import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  createModelServer,
  entryKey,
  listEntries,
  neededGraphs,
  parseRange,
  PythonRuntime,
  requirementsFor,
  resolveEntry,
  splitKey,
  type RunOptions,
  type RunResult,
} from "../../src/index.js";

/** A runtime that "prepares" by writing the entry itself: no Python needed. */
class FakeRuntime extends PythonRuntime {
  calls: { script: string; args: readonly string[]; options: RunOptions }[] = [];
  constructor(private readonly root: string) {
    super({ interpreter: "none" });
  }
  override async run(script: string, args: readonly string[], options: RunOptions = {}): Promise<RunResult> {
    this.calls.push({ script, args, options });
    const key = entryKey(args[0]!, args[2]!);
    if (args[0] === "broken") return { code: 1, lines: ["step trying", "error no recipe for 'broken'"] };
    mkdirSync(join(this.root, key), { recursive: true });
    writeFileSync(join(this.root, key, "ynonnx_manifest.json"), JSON.stringify({ version: 2, graphs: [{ filename: "model.onnx", role: "single" }] }));
    writeFileSync(join(this.root, key, "model.onnx"), "x".repeat(10));
    return { code: 0, lines: ["step exporting", `ready ${key}`] };
  }
}

let root: string;
let hub: string;
let server: Server;
let base: string;
let runtime: FakeRuntime;

beforeAll(async () => {
  root = mkdtempSync(join(tmpdir(), "onnxweb-server-"));
  hub = mkdtempSync(join(tmpdir(), "onnxweb-hub-"));
  const entry = (key: string, manifest: object, files: Record<string, string>) => {
    mkdirSync(join(root, key), { recursive: true });
    writeFileSync(join(root, key, "ynonnx_manifest.json"), JSON.stringify(manifest));
    for (const [name, data] of Object.entries(files)) writeFileSync(join(root, key, name), data);
  };
  entry(
    "small_mix_decoder-int4_encoder-int8",
    { version: 2, model_key: "small_mix", graphs: [{ filename: "encoder_model.onnx", role: "encoder" }, { filename: "decoder_model_merged.onnx", role: "decoder_merged", external_data: ["decoder_model_merged.onnx_data"] }, { filename: "decoder_model.onnx", role: "decoder" }] },
    { "encoder_model.onnx": "e".repeat(1000), "decoder_model_merged.onnx": "d".repeat(500), "decoder_model_merged.onnx_data": "w".repeat(2_000_000), "decoder_model.onnx": "z".repeat(9_000_000) },
  );
  entry("turbo_int4", { version: 2, model_key: "turbo_algo-int4-hqq", graphs: [] }, { "model.onnx": "x" });
  entry("tiny_float32", { version: 2, graphs: [{ filename: "model.onnx", role: "single" }] }, { "model.onnx": "0123456789" });
  mkdirSync(join(root, "not-an-entry"));
  const snapshot = join(hub, "models--org--repo", "snapshots", "abc", "onnx");
  mkdirSync(snapshot, { recursive: true });
  writeFileSync(join(snapshot, "model.onnx"), "hub bytes");

  runtime = new FakeRuntime(root);
  server = createModelServer({ models: { root, prepare: runtime }, mirror: { hubCache: hub, offline: true } });
  await new Promise<void>((done) => server.listen(0, done));
  const address = server.address();
  base = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
});

afterAll(() => {
  server.close();
  rmSync(root, { recursive: true, force: true });
  rmSync(hub, { recursive: true, force: true });
});

describe("store", () => {
  it("lists entries with the size a run downloads, skipping the replaced pair and non-entries", async () => {
    const entries = await listEntries(root);
    expect(entries.map((e) => e.key)).toEqual(["tiny_float32", "turbo_int4", "small_mix_decoder-int4_encoder-int8"]);
    const small = entries.find((e) => e.key.startsWith("small"))!;

    expect(small.sizeMb).toBe(2);
    expect(small.graphs).toEqual(["encoder", "decoder_merged"]);
    expect(small.external).toBe(true);
  });

  it("splits a key on its first underscore", () => {
    expect(splitKey("turbo_int8_float16")).toEqual({ name: "turbo", computeType: "int8_float16" });
    expect(splitKey("plain")).toEqual({ name: "plain", computeType: "float32" });
    expect(entryKey("onnx-community/Qwen3_0.6B", "int4")).toBe("onnx-community--Qwen3-0.6B_int4");
  });

  it("answers a precision with the one blend that produced it", async () => {
    expect(await resolveEntry(root, "small_int8")).toBe("small_mix_decoder-int4_encoder-int8");
    expect(await resolveEntry(root, "small_float16")).toBe("small_float16");
  });

  it("keeps the merged decoder over the pair", () => {
    expect(neededGraphs([{ role: "encoder" }, { role: "decoder" }, { role: "decoder_with_past" }, { role: "decoder_merged" }]).map((g) => g.role)).toEqual(["encoder", "decoder_merged"]);
    expect(neededGraphs([{ role: "single" }]).map((g) => g.role)).toEqual(["single"]);
  });
});

describe("ranges", () => {
  it("parses ranges, suffixes and refuses the impossible", () => {
    expect(parseRange(undefined, 10)).toBeNull();
    expect(parseRange("bytes=2-5", 10)).toEqual([2, 5]);
    expect(parseRange("bytes=4-", 10)).toEqual([4, 9]);
    expect(parseRange("bytes=-3", 10)).toEqual([7, 9]);
    expect(parseRange("bytes=8-100", 10)).toEqual([8, 9]);
    expect(parseRange("bytes=10-", 10)).toBe("invalid");
  });
});

describe("model routes", () => {
  it("serves the catalogue", async () => {
    const body = (await (await fetch(`${base}/models/catalog.json`)).json()) as { entries: { key: string }[] };
    expect(body.entries.length).toBe(3);
  });

  it("serves a file whole and by range, with CORS for another origin", async () => {
    const whole = await fetch(`${base}/models/tiny_float32/model.onnx`);
    expect(await whole.text()).toBe("0123456789");
    expect(whole.headers.get("access-control-allow-origin")).toBe("*");
    expect(whole.headers.get("cross-origin-resource-policy")).toBe("cross-origin");
    const part = await fetch(`${base}/models/tiny_float32/model.onnx`, { headers: { range: "bytes=3-5" } });
    expect(part.status).toBe(206);
    expect(part.headers.get("content-range")).toBe("bytes 3-5/10");
    expect(await part.text()).toBe("345");
    expect((await fetch(`${base}/models/tiny_float32/model.onnx`, { headers: { range: "bytes=20-" } })).status).toBe(416);
  });

  it("refuses names outside the whitelist and variants no browser runs", async () => {
    expect((await fetch(`${base}/models/bad%20name/model.onnx`)).status).toBe(400);
    expect((await fetch(`${base}/models/tiny_float32/..%2F..%2Fsecret`)).status).toBe(404);
    expect((await fetch(`${base}/models/turbo_int4/model.onnx`)).status).toBe(409);
    expect((await fetch(`${base}/models/tiny_float32/missing.onnx`)).status).toBe(404);
  });

  it("makes a missing model once, then answers from the store", async () => {
    const [a, b] = await Promise.all([fetch(`${base}/models/prepare?name=solon-embeddings-large&computeType=int8`), fetch(`${base}/models/prepare?name=solon-embeddings-large&computeType=int8`)]);
    expect(await a.json()).toEqual({ ready: true, entry: "solon-embeddings-large_int8" });
    expect((await b.json()) as object).toEqual({ ready: true, entry: "solon-embeddings-large_int8" });
    expect(runtime.calls.length).toBe(1);
    expect(runtime.calls[0]!.options.needs).toEqual(["export"]);
    expect(runtime.calls[0]!.options.env).toEqual({ YNONNXWEB_MODELS: root });
    await fetch(`${base}/models/prepare?name=solon-embeddings-large&computeType=int8`);
    expect(runtime.calls.length).toBe(1);
    expect(await (await fetch(`${base}/models/solon-embeddings-large_int8/model.onnx`)).text()).toBe("x".repeat(10));
  });

  it("reports why a model could not be made", async () => {
    const answer = await fetch(`${base}/models/prepare?name=broken&computeType=float32`);
    expect(answer.status).toBe(422);
    expect(await answer.json()).toEqual({ ready: false, error: "no recipe for 'broken'" });
    expect((await fetch(`${base}/models/prepare?name=../x&computeType=float32`)).status).toBe(400);
  });

  it("takes a Hub language model's name", async () => {
    const answer = await fetch(`${base}/models/prepare?name=onnx-community/Qwen3-0.6B-ONNX&computeType=int4`);
    expect(await answer.json()).toEqual({ ready: true, entry: "onnx-community--Qwen3-0.6B-ONNX_int4" });
    expect(runtime.calls.at(-1)!.options.needs).toEqual(["hub"]);
  });
});

describe("Hub mirror", () => {
  it("serves a file from the Hub cache, any revision", async () => {
    const answer = await fetch(`${base}/hf/org/repo/resolve/main/onnx/model.onnx`);
    expect(await answer.text()).toBe("hub bytes");
  });

  it("answers 404 offline for what is not on disk", async () => {
    expect((await fetch(`${base}/hf/org/repo/resolve/main/onnx/other.onnx`)).status).toBe(404);
    expect((await fetch(`${base}/hf/org/repo/blob/main/x`)).status).toBe(404);
  });
});

describe("python runtime", () => {
  it("picks the requirement set from the model name", () => {
    expect(requirementsFor("yolo26s-seg")).toEqual(["yolo"]);
    expect(requirementsFor("onnx-community/Qwen3-0.6B-ONNX")).toEqual(["hub"]);
    expect(requirementsFor("monot5-base")).toEqual(["export"]);
  });

  it("names what it tried when no interpreter answers, and never installs one", async () => {
    const missing = new PythonRuntime({ interpreter: join(tmpdir(), "no-such-python") });
    expect(missing.managed).toBe(false);
    await expect(missing.interpreter()).rejects.toThrow(/no Python 3\.10\+ found .*Install Python \(this package does not\)/);
  });
});
