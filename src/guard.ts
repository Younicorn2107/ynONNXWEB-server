/** Who may talk to the server at all, checked before any route. */

import { timingSafeEqual } from "node:crypto";
import type { IncomingMessage } from "node:http";

export interface GuardOptions {
  /**
   * Host names (without port) besides the loopback ones: `localhost`, `127.0.0.1`, `[::1]`, `::1`.
   */
  allowedHosts?: readonly string[];
  /** `true` accepts any `Host` (behind a reverse proxy that checks it). */
  anyHost?: boolean;
  /**
   * Origins (`https://app.example`) allowed to call cross-origin; the server's own origin always
   * is.
   */
  allowedOrigins?: readonly string[];
  /** Required as `Authorization: Bearer <token>` on every request when set. */
  token?: string | null;
}

const LOOPBACK = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

/** The host name of a `Host` header, port removed (`[::1]:8765` → `[::1]`). */
export function hostName(header: string | undefined): string | null {
  if (!header) return null;
  const value = header.trim().toLowerCase();
  if (value.startsWith("[")) {
    const end = value.indexOf("]");
    return end > 0 ? value.slice(0, end + 1) : null;
  }
  const colon = value.lastIndexOf(":");
  return colon > 0 ? value.slice(0, colon) : value;
}

function sameToken(given: string, expected: string): boolean {
  const a = Buffer.from(given);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

export type GuardVerdict = { ok: true; origin: string | null } | { ok: false; status: number; reason: string };

/**
 * Whether `request` may be served; `origin` is the allowed cross-origin caller to echo in CORS
 * headers.
 */
export function checkRequest(request: IncomingMessage, options: GuardOptions = {}): GuardVerdict {
  const host = hostName(request.headers.host);
  if (!options.anyHost) {
    const allowed = host !== null && (LOOPBACK.has(host) || (options.allowedHosts ?? []).some((h) => h.toLowerCase() === host));
    if (!allowed) return { ok: false, status: 403, reason: "unknown Host" };
  }
  const origin = typeof request.headers.origin === "string" ? request.headers.origin : null;
  let crossOrigin: string | null = null;
  if (origin && origin !== "null") {
    let originHost: string | null = null;
    try {
      originHost = new URL(origin).host.toLowerCase();
    } catch {
      return { ok: false, status: 403, reason: "malformed Origin" };
    }
    if (originHost !== String(request.headers.host ?? "").toLowerCase()) {
      if (!(options.allowedOrigins ?? []).includes(origin)) return { ok: false, status: 403, reason: "origin not allowed" };
      crossOrigin = origin;
    }
  } else if (origin === "null") {
    return { ok: false, status: 403, reason: "opaque origin" };
  }
  if (request.headers["sec-fetch-site"] === "cross-site" && crossOrigin === null) {
    return { ok: false, status: 403, reason: "cross-site request" };
  }

  if (options.token && request.method !== "OPTIONS") {
    const header = String(request.headers.authorization ?? "");
    const given = header.startsWith("Bearer ") ? header.slice(7) : "";
    if (!sameToken(given, options.token)) return { ok: false, status: 401, reason: "missing or wrong token" };
  }
  return { ok: true, origin: crossOrigin };
}

/** Whether an address the server binds to is the loopback only. */
export function isLoopback(address: string): boolean {
  return LOOPBACK.has(address) || address === "127.0.0.1" || /^127\./.test(address);
}
