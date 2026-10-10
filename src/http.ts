/**
 * The HTTP pieces every route shares: connect-style middleware types, JSON answers, and files
 * served with byte ranges.
 */

import { createReadStream, type Stats } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";

export type Next = (error?: unknown) => void;
export type Middleware = (request: IncomingMessage, response: ServerResponse, next: Next) => void | Promise<void>;

/** What logs: Vite's logger has this shape, and so does `console`. */
export interface Logger {
  info(message: string): void;
  warn(message: string): void;
}

/**
 * A file or folder name a URL may name. Whitelisted, not sanitised: a rule about what a name may
 * contain is checkable, while stripping what it must not is a game of catch-up — and joining URL
 * segments onto a directory is how a static server turns into a file-read primitive.
 */
export const SAFE_NAME = /^[A-Za-z0-9._-]+$/;
/** The same for a Hub path segment, which may hold an `@` (revisions). */
export const SAFE_SEGMENT = /^[A-Za-z0-9._@-]+$/;

export function sendJson(response: ServerResponse, status: number, body: unknown): void {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json");
  response.setHeader("Cache-Control", "no-store");
  response.end(JSON.stringify(body));
}

export function sendText(response: ServerResponse, status: number, body: string): void {
  response.statusCode = status;
  response.setHeader("Content-Type", "text/plain; charset=utf-8");
  response.end(body);
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
  const match = header ? /bytes=(\d*)-(\d*)/.exec(header) : null;
  if (!match) return null;
  let start: number;
  let end: number;
  if (match[1] === "" && match[2] !== "") {
    start = Math.max(0, size - Number(match[2]));
    end = size - 1;
  } else {
    start = match[1] ? Number(match[1]) : 0;
    end = match[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
  }
  if (start >= size || start > end) return "invalid";
  return [start, end];
}

/**
 * A file, with byte ranges: a model that cannot resume is a download that starts over on every
 * reload, and the browser libraries take a weight file larger than one buffer in ranged parts.
 */
export function serveFile(request: IncomingMessage, response: ServerResponse, file: string, stat: Stats, cacheControl = "no-cache"): void {
  const dot = file.lastIndexOf(".");
  response.setHeader("Content-Type", TYPES[dot >= 0 ? file.slice(dot) : ""] ?? "application/octet-stream");
  response.setHeader("Accept-Ranges", "bytes");
  response.setHeader("ETag", `"${stat.size}-${Math.round(stat.mtimeMs)}"`);

  response.setHeader("Cache-Control", cacheControl);
  const range = parseRange(request.headers.range, stat.size);
  if (range === "invalid") {
    response.statusCode = 416;
    response.setHeader("Content-Range", `bytes */${stat.size}`);
    response.end();
    return;
  }
  if (range) {
    const [start, end] = range;
    response.statusCode = 206;
    response.setHeader("Content-Length", String(end - start + 1));
    response.setHeader("Content-Range", `bytes ${start}-${end}/${stat.size}`);
    if (request.method === "HEAD") return void response.end();
    createReadStream(file, { start, end }).pipe(response);
    return;
  }
  response.setHeader("Content-Length", String(stat.size));
  if (request.method === "HEAD") return void response.end();
  createReadStream(file).pipe(response);
}

/** The request body as JSON (at most `limit` bytes). */
export function readJson(request: IncomingMessage, limit = 1e7): Promise<unknown> {
  return new Promise((resolve, reject) => {
    let raw = "";
    request.on("data", (chunk: Buffer) => {
      raw += chunk;
      if (raw.length > limit) reject(new Error("payload too large"));
    });
    request.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch (error) {
        reject(error as Error);
      }
    });
    request.on("error", reject);
  });
}
