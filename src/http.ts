/**
 * The HTTP pieces every route shares: connect-style middleware types, JSON answers, request bodies
 * read under a limit, paths decoded and contained, and files served with byte ranges.
 */

import { createReadStream, type Stats } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { resolve, sep } from "node:path";
import { pipeline } from "node:stream";

export type Next = (error?: unknown) => void;
export type Middleware = (request: IncomingMessage, response: ServerResponse, next: Next) => void | Promise<void>;

/** What logs: Vite's logger has this shape, and so does `console`. */
export interface Logger {
  info(message: string): void;
  warn(message: string): void;
}

/**
 * A file or folder name a URL may name. Whitelisted, not sanitised, and never starting with a dot:
 * `.` and `..` would otherwise pass (Node does not normalise `request.url`, so a raw client sends
 * them as they are), and a leading dot names hidden files (`.cache`, `.ssh`). Joining URL segments
 * onto a directory is how a static server turns into a file-read primitive; every join is also
 * contained after `resolve` (`contained`).
 */
export const SAFE_NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]*$/;
/** The same for a Hub path segment, which may hold an `@` (revisions). */
export const SAFE_SEGMENT = /^[A-Za-z0-9_@-][A-Za-z0-9._@-]*$/;

/**
 * `base` joined with `parts`, or null when the result is not strictly inside `base` (whatever the
 * parts hold: separators, dot segments, drive letters).
 */
export function contained(base: string, ...parts: readonly string[]): string | null {
  const root = resolve(base);
  const file = resolve(root, ...parts);
  return file.startsWith(root + sep) ? file : null;
}

/**
 * A URL path component decoded, or null when it is malformed (`%`, `%zz`): a 400, never an
 * exception.
 */
export function decodePath(raw: string): string | null {
  try {
    return decodeURIComponent(raw);
  } catch {
    return null;
  }
}

/**
 * A response not yet started gets `body`; one already streaming is cut (headers cannot change any
 * more).
 */
function finish(response: ServerResponse, status: number, type: string, body: string): void {
  if (response.headersSent) {
    response.destroy();
    return;
  }
  response.statusCode = status;
  response.setHeader("Content-Type", type);
  response.setHeader("Cache-Control", "no-store");
  response.end(body);
}

export function sendJson(response: ServerResponse, status: number, body: unknown): void {
  finish(response, status, "application/json", JSON.stringify(body));
}

export function sendText(response: ServerResponse, status: number, body: string): void {
  finish(response, status, "text/plain; charset=utf-8", body);
}

const TYPES: Record<string, string> = {
  ".json": "application/json",
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".txt": "text/plain; charset=utf-8",
};

/**
 * The `[start, end]` (inclusive) a `Range` header asks for, `"invalid"`, or null for the whole
 * file.
 */
export function parseRange(header: string | undefined, size: number): [number, number] | "invalid" | null {
  const match = header ? /^bytes=(\d*)-(\d*)$/.exec(header.trim()) : null;
  if (!match) return null;
  let start: number;
  let end: number;
  if (match[1] === "" && match[2] === "") return "invalid";
  if (match[1] === "") {
    start = Math.max(0, size - Number(match[2]));
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
  }
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start >= size || start > end) return "invalid";
  return [start, end];
}

/**
 * A file, with byte ranges: a model that cannot resume is a download that starts over on every
 * reload, and the browser libraries take a weight file larger than one buffer in ranged parts.
 * Streamed with `pipeline`, so a read error (the file removed between `stat` and `open`) ends this
 * response instead of the process, and a client that goes away closes the file.
 */
export function serveFile(request: IncomingMessage, response: ServerResponse, file: string, stat: Stats, cacheControl = "no-cache"): void {
  const dot = file.lastIndexOf(".");
  const etag = `"${stat.size}-${Math.round(stat.mtimeMs)}"`;
  response.setHeader("Content-Type", TYPES[dot >= 0 ? file.slice(dot) : ""] ?? "application/octet-stream");
  response.setHeader("Accept-Ranges", "bytes");
  response.setHeader("ETag", etag);

  response.setHeader("Cache-Control", cacheControl);
  if (request.headers["if-none-match"] === etag && !request.headers.range) {
    response.statusCode = 304;
    response.end();
    return;
  }
  const range = parseRange(request.headers.range, stat.size);
  if (range === "invalid") {
    response.statusCode = 416;
    response.setHeader("Content-Range", `bytes */${stat.size}`);
    response.end();
    return;
  }
  const [start, end] = range ?? [0, stat.size - 1];
  if (range) {
    response.statusCode = 206;
    response.setHeader("Content-Range", `bytes ${start}-${end}/${stat.size}`);
  }
  response.setHeader("Content-Length", String(stat.size === 0 ? 0 : end - start + 1));
  if (request.method === "HEAD" || stat.size === 0) return void response.end();
  pipeline(createReadStream(file, { start, end }), response, () => undefined);
}

/** The largest JSON body a route reads by default: the routes take a few hundred bytes. */
export const DEFAULT_BODY_LIMIT = 64 * 1024;

export class BodyError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

/**
 * The request body as JSON, at most `limit` bytes. `Content-Type` must be `application/json`: a
 * cross-site page can only send that type after a CORS preflight, so a blind `fetch(..., { body })`
 * from another site is refused here rather than acted on. A body past the limit destroys the
 * request (a client streaming forever must not grow the heap).
 */
export function readJson(request: IncomingMessage, limit = DEFAULT_BODY_LIMIT): Promise<unknown> {
  return new Promise((resolveBody, reject) => {
    const type = String(request.headers["content-type"] ?? "").split(";")[0]!.trim().toLowerCase();
    if (type !== "application/json") return reject(new BodyError(415, "expected Content-Type: application/json"));
    const declared = Number(request.headers["content-length"] ?? 0);
    if (declared > limit) {
      request.resume();
      return reject(new BodyError(413, "payload too large"));
    }
    let size = 0;
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > limit) {
        request.destroy();
        reject(new BodyError(413, "payload too large"));
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => {
      try {
        resolveBody(size > 0 ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : {});
      } catch {
        reject(new BodyError(400, "malformed JSON"));
      }
    });
    request.on("error", () => reject(new BodyError(400, "request aborted")));
  });
}
