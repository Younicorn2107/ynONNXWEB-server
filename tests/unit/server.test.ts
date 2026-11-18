import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { createServer, request as httpRequest, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  BenchQueue,
  checkRequest,
  contained,
  createModelServer,
  decodePath,
  entryKey,
  hostName,
  listEntries,
  neededGraphs,
  parseRange,
  PythonRuntime,
  repoAllowed,
  requirementsFor,
  resolveEntry,
  SAFE_NAME,
  SAFE_SEGMENT,
  splitKey,
  validPrecision,
  whichOnPath,
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
    const name = args.at(-1)!;
    const key = entryKey(name, args[1]!);
    if (name === "broken") return { code: 1, lines: ["step trying", "error no recipe for 'broken'"] };
    mkdirSync(join(this.root, key), { recursive: true });
    writeFileSync(join(this.root, key, "ynonnx_manifest.json"), JSON.stringify({ version: 2, graphs: [{ filename: "model.onnx", role: "single" }] }));
    writeFileSync(join(this.root, key, "model.onnx"), "x".repeat(10));
    return { code: 0, lines: ["step exporting", `ready ${key}`] };
  }
}

const COMMIT_A = "a".repeat(40);
const COMMIT_B = "b".repeat(40);

let root: string;
let hub: string;
let secret: string;
let server: Server;
let port = 0;
let base: string;
let runtime: FakeRuntime;

/**
 * A request sent exactly as written: no URL normalisation, any header (what curl --path-as-is
 * does).
 */
function raw(path: string, options: { method?: string; headers?: Record<string, string>; body?: string } = {}): Promise<{ status: number; body: string; headers: Record<string, unknown> }> {
  return new Promise((resolve, reject) => {
    const req = httpRequest({ host: "127.0.0.1", port, path, method: options.method ?? "GET", headers: { host: `127.0.0.1:${port}`, ...(options.body !== undefined ? { "content-length": String(Buffer.byteLength(options.body)) } : {}), ...options.headers } }, (res) => {
      let body = "";
      res.on("data", (c: Buffer) => (body += c.toString()));
      res.on("end", () => resolve({ status: res.statusCode ?? 0, body, headers: res.headers }));
    });
    req.on("error", reject);
    if (options.body !== undefined) req.write(options.body);
    req.end();
  });
}

const post = (path: string, body: unknown, headers: Record<string, string> = {}) =>
  raw(path, { method: "POST", body: JSON.stringify(body), headers: { "content-type": "application/json", ...headers } });

beforeAll(async () => {
  root = mkdtempSync(join(tmpdir(), "onnxweb-server-"));
  hub = mkdtempSync(join(tmpdir(), "onnxweb-hub-"));

  secret = join(hub, "token");
  writeFileSync(secret, "hf_SECRET");
  writeFileSync(join(root, "..secret"), "nope");
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
  entry("tiny_float32", { version: 2, graphs: [{ filename: "model.onnx", role: "single" }] }, { "model.onnx": "0123456789", ".hidden": "h" });
  mkdirSync(join(root, "not-an-entry"));

  const repo = join(hub, "models--org--repo");
  mkdirSync(join(repo, "refs"), { recursive: true });
  writeFileSync(join(repo, "refs", "main"), COMMIT_A);
  for (const [commit, text] of [[COMMIT_A, "main bytes"], [COMMIT_B, "old bytes"]] as const) {
    mkdirSync(join(repo, "snapshots", commit, "onnx"), { recursive: true });
    writeFileSync(join(repo, "snapshots", commit, "onnx", "model.onnx"), text);
  }

  runtime = new FakeRuntime(root);
  server = createModelServer({ models: { root, prepare: runtime }, mirror: { hubCache: hub, offline: true }, allowedOrigins: ["https://app.example"] });
  await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
  const address = server.address();
  port = typeof address === "object" && address ? address.port : 0;
  base = `http://127.0.0.1:${port}`;
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
  });
});

describe("names and paths", () => {
  it("refuses dot segments and hidden names at the whitelist", () => {
    for (const bad of [".", "..", ".ssh", ".hidden", "a/b", "a\\b", "", "C:", "nul\0"]) {
      expect(SAFE_NAME.test(bad), bad).toBe(false);
      expect(SAFE_SEGMENT.test(bad), bad).toBe(false);
    }
    for (const good of ["model.onnx", "tiny_float32", "decoder_model_merged.onnx_data", "-x"]) expect(SAFE_NAME.test(good)).toBe(true);
    expect(SAFE_SEGMENT.test("v1@main")).toBe(true);
  });

  it("contains every join after resolve", () => {
    expect(contained(root, "tiny_float32", "model.onnx")).toBe(join(root, "tiny_float32", "model.onnx"));
    for (const parts of [[".."], ["..", "x"], ["a", "..", "..", "x"], ["/etc/passwd"], ["C:\\Windows"], [""]]) expect(contained(root, ...parts), parts.join("/")).toBeNull();
  });

  it("decodes paths without throwing", () => {
    expect(decodePath("%2e%2e")).toBe("..");
    expect(decodePath("%")).toBeNull();
    expect(decodePath("%zz")).toBeNull();
  });

  it("parses ranges, suffixes and refuses the impossible", () => {
    expect(parseRange(undefined, 10)).toBeNull();
    expect(parseRange("bytes=2-5", 10)).toEqual([2, 5]);
    expect(parseRange("bytes=4-", 10)).toEqual([4, 9]);
    expect(parseRange("bytes=-3", 10)).toEqual([7, 9]);
    expect(parseRange("bytes=8-100", 10)).toEqual([8, 9]);
    expect(parseRange("bytes=10-", 10)).toBe("invalid");
    expect(parseRange("bytes=-", 10)).toBe("invalid");
  });

  it("checks precisions against the known set", () => {
    expect(validPrecision("int4_float16")).toBe(true);
    for (const bad of ["--help", "-x", "int3", "float32;rm"]) expect(validPrecision(bad)).toBe(false);
  });

  it("matches repository allow-lists", () => {
    expect(repoAllowed("onnx-community", "x", ["onnx-community/*"])).toBe(true);
    expect(repoAllowed("evil", "x", ["onnx-community/*"])).toBe(false);
    expect(repoAllowed("evil", "x", undefined)).toBe(true);
  });

  it("finds interpreters on PATH only, never in the working folder", () => {
    const cwd = process.cwd();
    const found = whichOnPath("python-that-does-not-exist", { PATH: cwd });
    expect(found).toBeNull();
    expect(whichOnPath("x", { PATH: "relative/dir" })).toBeNull();
  });
});

describe("traversal, every route", () => {
  const attempts = [
    "/models/tiny_float32/../../token",
    "/models/tiny_float32/%2e%2e%2f%2e%2e%2ftoken",
    "/models/../..secret",
    "/models/tiny_float32/.hidden",
    "/models/%2e%2e/model.onnx",
    "/models/%",
    "/models/tiny_float32/%zz",
    "/hf/org/repo/resolve/main/../../../token",
    "/hf/org/repo/resolve/main/%2e%2e/%2e%2e/%2e%2e/token",
    "/hf/org/repo/resolve/main/onnx/..%2f..%2f..%2f..%2ftoken",
    "/hf/org/repo/resolve/../../token",
    "/hf/../../token",
    "/hf/api/../org/repo/resolve/main/onnx/model.onnx",
    "/hf/api/models/org/repo/resolve/main/x",
    "/hf/%",
  ];
  for (const path of attempts) {
    it(`never serves a file outside: ${path}`, async () => {
      const answer = await raw(path);
      expect(answer.status, path).toBeGreaterThanOrEqual(400);
      expect(answer.body).not.toContain("hf_SECRET");
      expect(answer.body).not.toContain("nope");
    });
  }

  it("survives malformed URLs (the server still answers)", async () => {
    await raw("/models/%");
    expect((await raw("/models/catalog.json")).status).toBe(200);
  });
});

describe("guard", () => {
  it("refuses unknown hosts (DNS rebinding) and cross-site requests", async () => {
    expect((await raw("/models/catalog.json", { headers: { host: "evil.example" } })).status).toBe(403);
    expect((await raw("/models/catalog.json", { headers: { "sec-fetch-site": "cross-site" } })).status).toBe(403);
    expect((await raw("/models/catalog.json", { headers: { origin: "https://evil.example" } })).status).toBe(403);
    expect((await raw("/models/catalog.json", { headers: { origin: "null" } })).status).toBe(403);
  });

  it("answers an allowed origin with its own name, never *", async () => {
    const answer = await raw("/models/tiny_float32/model.onnx", { headers: { origin: "https://app.example" } });
    expect(answer.status).toBe(200);
    expect(answer.headers["access-control-allow-origin"]).toBe("https://app.example");
    expect(String(answer.headers["access-control-expose-headers"])).toContain("etag");
    const same = await raw("/models/tiny_float32/model.onnx");
    expect(same.headers["access-control-allow-origin"]).toBeUndefined();
  });

  it("requires the token when one is set, in constant time", () => {
    const req = (headers: Record<string, string>) => ({ headers: { host: "localhost:1", ...headers }, method: "GET" }) as never;
    expect(checkRequest(req({}), { token: "t0k" })).toMatchObject({ ok: false, status: 401 });
    expect(checkRequest(req({ authorization: "Bearer nope" }), { token: "t0k" })).toMatchObject({ ok: false, status: 401 });
    expect(checkRequest(req({ authorization: "Bearer t0k" }), { token: "t0k" })).toMatchObject({ ok: true });
    expect(hostName("[::1]:8765")).toBe("[::1]");
    expect(hostName("LocalHost:80")).toBe("localhost");
  });
});

describe("model routes", () => {
  it("serves the catalogue without any absolute path", async () => {
    const answer = await raw("/models/catalog.json");
    const body = JSON.parse(answer.body) as { entries: { key: string }[]; root?: string };
    expect(body.entries.length).toBe(3);
    expect(body.root).toBeUndefined();
    expect(answer.body).not.toContain(root);
  });

  it("serves a file whole, by range, and answers a matching ETag with 304", async () => {
    const whole = await fetch(`${base}/models/tiny_float32/model.onnx`);
    expect(await whole.text()).toBe("0123456789");
    const part = await fetch(`${base}/models/tiny_float32/model.onnx`, { headers: { range: "bytes=3-5" } });
    expect(part.status).toBe(206);
    expect(await part.text()).toBe("345");
    expect((await fetch(`${base}/models/tiny_float32/model.onnx`, { headers: { range: "bytes=20-" } })).status).toBe(416);
    const again = await fetch(`${base}/models/tiny_float32/model.onnx`, { headers: { "if-none-match": whole.headers.get("etag")! } });
    expect(again.status).toBe(304);
  });

  it("refuses variants no browser runs, and missing files without naming the folder on disk", async () => {
    expect((await raw("/models/turbo_int4/model.onnx")).status).toBe(409);
    const missing = await raw("/models/tiny_float32/missing.onnx");
    expect(missing.status).toBe(404);
    expect(missing.body).not.toContain(root);
  });

  it("makes a missing model once, through a JSON POST only", async () => {
    expect((await raw("/models/prepare?name=solon-embeddings-large&computeType=int8")).status).toBe(405);

    expect((await raw("/models/prepare", { method: "POST", body: '{"name":"x"}', headers: { "content-type": "text/plain" } })).status).toBe(415);
    const [a, b] = await Promise.all([post("/models/prepare", { name: "solon-embeddings-large", computeType: "int8" }), post("/models/prepare", { name: "solon-embeddings-large", computeType: "int8" })]);
    expect(JSON.parse(a.body)).toEqual({ ready: true, entry: "solon-embeddings-large_int8" });
    expect(JSON.parse(b.body)).toEqual({ ready: true, entry: "solon-embeddings-large_int8" });
    expect(runtime.calls.length).toBe(1);
    const call = runtime.calls[0]!;
    expect(call.args).toEqual(["--precision", "int8", "--", "solon-embeddings-large"]);
    expect(call.options.needs).toEqual(["export"]);
    expect(call.options.signal).toBeDefined();
    await post("/models/prepare", { name: "solon-embeddings-large", computeType: "int8" });
    expect(runtime.calls.length).toBe(1);
  });

  it("refuses bad names, option-like precisions and oversized bodies", async () => {
    for (const body of [{ name: "../x" }, { name: "-rf" }, { name: "x", computeType: "--help" }, { name: "x", computeType: "int3" }, { name: 3 }]) {
      expect((await post("/models/prepare", body)).status, JSON.stringify(body)).toBe(400);
    }
    expect((await post("/models/prepare", { name: "x".repeat(10_000) })).status).toBe(413);
  });

  it("reports why a model could not be made", async () => {
    const answer = await post("/models/prepare", { name: "broken", computeType: "float32" });
    expect(answer.status).toBe(422);
    expect(JSON.parse(answer.body)).toEqual({ ready: false, error: "no recipe for 'broken'" });
  });

  it("has no bench queue unless asked", async () => {
    expect((await raw("/models/bench/status")).status).toBe(404);
  });
});

describe("Hub mirror", () => {
  it("serves main through refs/main, and a commit through its own snapshot", async () => {
    expect((await raw("/hf/org/repo/resolve/main/onnx/model.onnx")).body).toBe("main bytes");
    expect((await raw(`/hf/org/repo/resolve/${COMMIT_B}/onnx/model.onnx`)).body).toBe("old bytes");
  });

  it("refuses other revisions, writes, partial files and other API routes", async () => {
    expect((await raw("/hf/org/repo/resolve/refs%2Fpr%2F1/onnx/model.onnx")).status).toBe(400);
    expect((await raw("/hf/org/repo/resolve/dev/onnx/model.onnx")).status).toBe(400);
    expect((await raw("/hf/org/repo/resolve/main/onnx/model.onnx", { method: "POST" })).status).toBe(405);
    expect((await raw("/hf/org/repo/resolve/main/onnx/model.onnx.partial")).status).toBe(404);
    expect((await raw("/hf/api/spaces/org/repo")).status).toBe(404);
  });

  it("answers 404 offline for what is not on disk", async () => {
    expect((await raw("/hf/org/repo/resolve/main/onnx/other.onnx")).status).toBe(404);
  });
});

describe("bench queue", () => {
  it("hands jobs out, collects results, bounds what it keeps and writes off the abandoned", async () => {
    const queue = new BenchQueue(1000);
    const srv = createServer((req, res) => void queue.handle(req, res, req.url!.slice(1)).catch(() => res.end()));
    await new Promise<void>((done) => srv.listen(0, "127.0.0.1", done));
    const address = srv.address();
    const at = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
    const json = { "content-type": "application/json" };
    await fetch(`${at}/submit`, { method: "POST", headers: json, body: JSON.stringify([{ id: "a", model: "m" }, { id: "b", model: "n" }]) });
    expect(((await (await fetch(`${at}/job`)).json()) as { id: string }).id).toBe("a");
    expect(((await (await fetch(`${at}/job`)).json()) as { id: string }).id).toBe("b");
    expect((await fetch(`${at}/job`)).status).toBe(204);
    await fetch(`${at}/result`, { method: "POST", headers: json, body: JSON.stringify({ id: "a", status: "ok" }) });
    expect((await fetch(`${at}/submit`, { method: "POST", headers: json, body: JSON.stringify([{ model: "no id" }]) })).status).toBe(400);
    await fetch(`${at}/close`, { method: "POST" });
    queue.sweep(Date.now() + 5000);
    const status = (await (await fetch(`${at}/status`)).json()) as { finished: boolean; results: { id: string; error?: string }[] };
    expect(status.finished).toBe(true);
    expect(status.results.find((r) => r.id === "b")!.error).toMatch(/never answered/);
    expect((await fetch(`${at}/job`)).status).toBe(205);
    srv.close();
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
