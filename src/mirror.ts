/** A local mirror of the Hugging Face Hub. */

import { createWriteStream, promises as fs, type Stats } from "node:fs";
import type { ServerResponse } from "node:http";
import { dirname } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ReadableStream as WebReadableStream } from "node:stream/web";

import { SAFE_NAME, SAFE_SEGMENT, contained, decodePath, sendText, serveFile, type Middleware } from "./http.js";
import { corpusRoot, hubCache, mirrorRoot } from "./paths.js";

const UPSTREAM = "https://huggingface.co";
const COMMIT = /^[0-9a-f]{40}$/;
/** The largest API answer relayed (a folder listing is a few kilobytes). */
const MAX_API_BYTES = 4 * 2 ** 20;

type Found = { file: string; stat: Stats } | { status: number };

/** `org/repo` matched against patterns like `onnx-community/*` or an exact name. */
export function repoAllowed(org: string, repo: string, patterns: readonly string[] | undefined): boolean {
  if (!patterns) return true;
  const name = `${org}/${repo}`;
  return patterns.some((p) => (p.endsWith("/*") ? name.startsWith(p.slice(0, -1)) : p === name));
}

/**
 * The file in the Hub cache, for `revision`: a commit's own snapshot, or the snapshot
 * `refs/<branch>` points at. Null when the cache does not hold it.
 */
export async function fromHubCache(org: string, repo: string, revision: string, path: readonly string[], cache = hubCache()): Promise<{ file: string; stat: Stats } | null> {
  const base = contained(cache, `models--${org}--${repo}`);
  if (!base) return null;
  let commit = revision;
  if (!COMMIT.test(revision)) {
    const ref = contained(base, "refs", revision);
    const text = ref ? await fs.readFile(ref, "utf8").catch(() => null) : null;
    if (!text || !COMMIT.test(text.trim())) return null;
    commit = text.trim();
  }
  const snapshot = contained(base, "snapshots", commit);
  const file = snapshot ? contained(snapshot, ...path) : null;
  if (!file) return null;
  const stat = await fs.stat(file).catch(() => null);
  return stat?.isFile() ? { file, stat } : null;
}

const commits = new Map<string, { commit: string; at: number }>();

async function commitOf(upstream: string, org: string, repo: string, revision: string): Promise<string | null> {
  if (COMMIT.test(revision)) return revision;
  const key = `${org}/${repo}@${revision}`;
  const known = commits.get(key);
  if (known && Date.now() - known.at < 60_000) return known.commit;
  const answer = await fetch(`${upstream}/api/models/${org}/${repo}/revision/${revision}`, { signal: AbortSignal.timeout(30_000) }).catch(() => null);
  if (!answer?.ok) return null;
  const sha = ((await answer.json().catch(() => ({}))) as { sha?: unknown }).sha;
  if (typeof sha !== "string" || !COMMIT.test(sha)) return null;
  commits.set(key, { commit: sha, at: Date.now() });
  return sha;
}

const downloads = new Map<string, Promise<number>>();

async function fromUpstream(root: string, upstream: string, org: string, repo: string, revision: string, path: readonly string[]): Promise<Found> {
  const commit = await commitOf(upstream, org, repo, revision);
  if (!commit) return { status: 404 };

  const file = contained(root, org, repo, commit, ...path);
  if (!file) return { status: 400 };
  const stat = await fs.stat(file).catch(() => null);
  if (stat?.isFile()) return { file, stat };
  let download = downloads.get(file);
  if (!download) {
    download = (async () => {
      const response = await fetch(`${upstream}/${org}/${repo}/resolve/${commit}/${path.map(encodeURIComponent).join("/")}`);
      if (!response.ok || !response.body) return response.status;
      await fs.mkdir(dirname(file), { recursive: true });
      const partial = `${file}.partial`;
      try {
        await pipeline(Readable.fromWeb(response.body as WebReadableStream), createWriteStream(partial));
        await fs.rename(partial, file);
      } catch (error) {
        await fs.rm(partial, { force: true });
        throw error;
      }
      return 200;
    })().finally(() => downloads.delete(file));
    downloads.set(file, download);
  }
  const status = await download;
  if (status !== 200) return { status };
  return { file, stat: await fs.stat(file) };
}

export interface MirrorOptions {
  mount?: string;
  /** The Hub cache read first; `HF_HUB_CACHE` / `HF_HOME` / `~/.cache/huggingface/hub`. */
  hubCache?: string;
  /** Where files fetched by the mirror are kept; `YNONNXWEB_HF_MIRROR`. */
  root?: string;
  /** The Hub (`https://huggingface.co`). */
  upstream?: string;
  /** Never fetch: serve what is on disk only. */
  offline?: boolean;
  /** Repositories served (`org/repo` or `org/*`); all when absent. */
  repos?: readonly string[];
  /** Revision names accepted besides `main` and 40-hex commits. */
  revisions?: readonly string[];
}

/** The listing route of the API, relayed with a size cap. */
async function relayListing(upstream: string, parts: readonly string[], query: string, response: ServerResponse): Promise<void> {
  const answer = await fetch(`${upstream}/${parts.map(encodeURIComponent).join("/")}${query ? `?${query}` : ""}`, { signal: AbortSignal.timeout(60_000) });
  response.statusCode = answer.status;
  response.setHeader("Content-Type", "application/json");
  if (!answer.body) return void response.end();
  let size = 0;
  const cap = new Transform({
    transform(chunk: Buffer, _encoding, done) {
      size += chunk.length;
      done(size > MAX_API_BYTES ? new Error("listing too large") : null, chunk);
    },
  });
  await pipeline(Readable.fromWeb(answer.body as WebReadableStream), cap, response).catch(() => response.destroy());
}

export function hfMirrorMiddleware(options: MirrorOptions = {}): Middleware {
  const mount = options.mount ?? "/hf/";
  const upstream = options.upstream ?? UPSTREAM;
  const revisionOk = (r: string) => r === "main" || COMMIT.test(r) || (options.revisions ?? []).includes(r);
  return async (request, response, next) => {
    if (!request.url?.startsWith(mount)) return next();
    if (request.method !== "GET" && request.method !== "HEAD") return sendText(response, 405, "read only");
    const [rawPath = "", query = ""] = request.url.slice(mount.length).split("?");
    const decoded = rawPath.split("/").filter(Boolean).map(decodePath);
    if (decoded.some((p) => p === null)) return sendText(response, 400, "bad path");
    const parts = decoded as string[];
    if (!parts.every((p) => SAFE_SEGMENT.test(p))) return sendText(response, 400, "bad path");
    try {
      if (parts[0] === "api") {
        const [, kind, org, repo, verb, revision] = parts;
        if (kind !== "models" || !org || !repo || verb !== "tree" || !revision || !revisionOk(revision)) return sendText(response, 404, "not relayed");
        if (!repoAllowed(org, repo, options.repos)) return sendText(response, 403, "repository not served");
        if (options.offline) return sendText(response, 404, "offline mirror: no Hub API");
        const safeQuery = new URLSearchParams(query).get("recursive") === "true" ? "recursive=true" : "";
        return await relayListing(upstream, parts, safeQuery, response);
      }
      const [org, repo, verb, revision, ...path] = parts;
      if (!org || !repo || verb !== "resolve" || !revision || path.length === 0) {
        return sendText(response, 404, `expected ${mount}{org}/{repo}/resolve/{revision}/{path}`);
      }
      if (!revisionOk(revision)) return sendText(response, 400, "revision must be main or a commit");
      if (!repoAllowed(org, repo, options.repos)) return sendText(response, 403, "repository not served");
      if (path.at(-1)!.endsWith(".partial")) return sendText(response, 404, "not found");
      const found: Found =
        (await fromHubCache(org, repo, revision, path, options.hubCache)) ??
        (options.offline ? { status: 404 } : await fromUpstream(options.root ?? mirrorRoot(), upstream, org, repo, revision, path));
      if (!("file" in found)) return sendText(response, found.status || 404, `${org}/${repo}/${path.join("/")}: HTTP ${found.status}`);
      return serveFile(request, response, found.file, found.stat);
    } catch {
      return sendText(response, 502, "upstream failed");
    }
  };
}

/** The mirror's own folder layout, for tests and tools: `root/org/repo/<commit>/path`. */
export function mirrorPath(root: string, org: string, repo: string, commit: string, path: readonly string[]): string | null {
  return contained(root, org, repo, commit, ...path) ?? null;
}

/**
 * `{mount}{file}` from the bench corpus folder; for benches, never part of a deployment (off unless
 * configured).
 */
export function corpusMiddleware(options: { mount?: string; root?: string } = {}): Middleware {
  const mount = options.mount ?? "/corpus/";
  return async (request, response, next) => {
    if (!request.url?.startsWith(mount)) return next();
    const name = decodePath(request.url.slice(mount.length).split("?")[0] ?? "");
    if (name === null || !SAFE_NAME.test(name)) return sendText(response, 400, "bad name");
    const file = contained(options.root ?? corpusRoot(), name);
    const stat = file ? await fs.stat(file).catch(() => null) : null;
    if (!file || !stat?.isFile()) return sendText(response, 404, "not in the corpus folder");
    return serveFile(request, response, file, stat);
  };
}
