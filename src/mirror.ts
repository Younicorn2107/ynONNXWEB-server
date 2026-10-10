/** A local mirror of the Hugging Face Hub. */

import { createWriteStream, promises as fs, type Stats } from "node:fs";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ReadableStream as WebReadableStream } from "node:stream/web";

import { SAFE_SEGMENT, sendText, serveFile, type Middleware } from "./http.js";
import { hubCache, mirrorRoot } from "./paths.js";

const UPSTREAM = "https://huggingface.co";

type Found = { file: string; stat: Stats } | { status: number };

/** The file in the Hub cache's snapshots, any revision, or null. */
export async function fromHubCache(org: string, repo: string, path: readonly string[], cache = hubCache()): Promise<{ file: string; stat: Stats } | null> {
  const snapshots = join(cache, `models--${org}--${repo}`, "snapshots");
  for (const revision of await fs.readdir(snapshots).catch(() => [] as string[])) {
    const file = join(snapshots, revision, ...path);
    const stat = await fs.stat(file).catch(() => null);
    if (stat?.isFile()) return { file, stat };
  }
  return null;
}

const downloads = new Map<string, Promise<number>>();

async function fromUpstream(root: string, upstream: string, org: string, repo: string, revision: string, path: readonly string[]): Promise<Found> {
  const file = join(root, org, repo, ...path);
  const stat = await fs.stat(file).catch(() => null);
  if (stat?.isFile()) return { file, stat };
  let download = downloads.get(file);
  if (!download) {
    download = (async () => {
      const response = await fetch(`${upstream}/${org}/${repo}/resolve/${revision}/${path.join("/")}`);
      if (!response.ok || !response.body) return response.status;
      await fs.mkdir(dirname(file), { recursive: true });
      const partial = `${file}.partial`;
      await pipeline(Readable.fromWeb(response.body as WebReadableStream), createWriteStream(partial));
      await fs.rename(partial, file);
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
}

export function hfMirrorMiddleware(options: MirrorOptions = {}): Middleware {
  const mount = options.mount ?? "/hf/";
  const upstream = options.upstream ?? UPSTREAM;
  return async (request, response, next) => {
    if (!request.url?.startsWith(mount)) return next();
    const parts = (request.url.slice(mount.length).split("?")[0] ?? "").split("/").filter(Boolean).map(decodeURIComponent);
    if (!parts.every((p) => SAFE_SEGMENT.test(p))) return sendText(response, 400, "bad path");
    try {
      if (parts[0] === "api") {
        if (options.offline) return sendText(response, 404, "offline mirror: no Hub API");
        const answer = await fetch(`${upstream}/${parts.join("/")}`);
        response.statusCode = answer.status;
        response.setHeader("Content-Type", answer.headers.get("content-type") ?? "application/json");
        return void response.end(Buffer.from(await answer.arrayBuffer()));
      }
      const [org, repo, verb, revision, ...path] = parts;
      if (!org || !repo || verb !== "resolve" || !revision || path.length === 0) {
        return sendText(response, 404, `expected ${mount}{org}/{repo}/resolve/{revision}/{path}`);
      }
      const found: Found =
        (await fromHubCache(org, repo, path, options.hubCache)) ??
        (options.offline ? { status: 404 } : await fromUpstream(options.root ?? mirrorRoot(), upstream, org, repo, revision, path));
      if (!("file" in found)) return sendText(response, found.status || 404, `${org}/${repo}/${path.join("/")}: HTTP ${found.status}`);
      return serveFile(request, response, found.file, found.stat);
    } catch (error) {
      return sendText(response, 502, String(error));
    }
  };
}

