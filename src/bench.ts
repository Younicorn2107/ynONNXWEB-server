/** The bench queue. */

import type { IncomingMessage, ServerResponse } from "node:http";

import { readJson, sendJson } from "./http.js";

interface Job {
  id: string;
  model?: string;
  backend?: string;
  [key: string]: unknown;
}

/**
 * How long a handed-out job may stay unanswered before it is written off. Longer than the page's
 * own deadline on purpose: when both are alive the page times out first and reports which model
 * hung, the more useful row. This one only catches the case where nobody is left to report
 * anything.
 */
const ABANDON_MS = 25 * 60 * 1000;

/** Bounds: a bench run is a few hundred jobs; anything past these is not a bench. */
export const MAX_PENDING = 2000;
export const MAX_RESULTS = 5000;
const MAX_RESULT_BYTES = 1024 * 1024;

export class BenchQueue {
  pending: Job[] = [];
  /**
   * Handed to the page, not answered for yet. Without it a job that leaves the queue and never
   * comes back is simply gone, and a loop waiting for the end waits for an end that cannot arrive
   * (a killed tab, a crashed worker, a driver reset).
   */
  inflight = new Map<string, { job: Job; since: number }>();
  results = new Map<string, Record<string, unknown>>();
  client: unknown = null;
  closed = false;

  constructor(private readonly abandonMs = ABANDON_MS) {}

  /** Every timed-out in-flight job moved into the results, so the run can end. */
  sweep(now = Date.now()): void {
    for (const [id, { job, since }] of this.inflight) {
      if (now - since < this.abandonMs) continue;
      this.inflight.delete(id);

      this.results.set(id, { id, status: "error", model: job.model, backend: job.backend, error: `client never answered within ${Math.round(this.abandonMs / 60000)} min` });
    }
  }

  get finished(): boolean {
    return this.closed && this.pending.length === 0 && this.inflight.size === 0;
  }

  /** `{mount}bench/<route>`. */
  async handle(request: IncomingMessage, response: ServerResponse, route: string): Promise<void> {
    const method = request.method ?? "GET";

    if (route === "hello" && method === "POST") {
      this.client = await readJson(request, 16 * 1024);
      return sendJson(response, 200, { ok: true });
    }
    if (route === "job" && method === "GET") {
      this.sweep();

      if (this.finished) return void ((response.statusCode = 205), response.end());
      const job = this.pending.shift();
      if (!job) return void ((response.statusCode = 204), response.end());
      this.inflight.set(job.id, { job, since: Date.now() });
      return sendJson(response, 200, job);
    }
    if (route === "result" && method === "POST") {
      const body = (await readJson(request, MAX_RESULT_BYTES)) as { id?: unknown } | null;
      if (typeof body?.id === "string" && (this.results.has(body.id) || this.results.size < MAX_RESULTS)) {
        this.inflight.delete(body.id);
        this.results.set(body.id, body as Record<string, unknown>);
      }
      return sendJson(response, 200, { ok: true });
    }

    if (route === "submit" && method === "POST") {
      const body = await readJson(request, MAX_RESULT_BYTES);
      const jobs = (Array.isArray(body) ? body : [body]) as Job[];
      if (!jobs.every((j) => j && typeof j === "object" && typeof (j as { id?: unknown }).id === "string")) return sendJson(response, 400, { error: "every job needs a string id" });
      if (this.pending.length + jobs.length > MAX_PENDING) return sendJson(response, 429, { error: `at most ${MAX_PENDING} pending jobs` });
      this.pending.push(...jobs);
      return sendJson(response, 200, { queued: jobs.length });
    }
    if (route === "status" && method === "GET") {
      this.sweep();
      return sendJson(response, 200, {
        client: this.client,
        pending: this.pending.length,

        inflight: this.inflight.size,
        done: this.results.size,
        closed: this.closed,

        finished: this.finished,
        results: [...this.results.values()],
      });
    }
    if (route === "close" && method === "POST") {
      this.closed = true;
      return sendJson(response, 200, { ok: true });
    }
    if (route === "reset" && method === "POST") {
      this.pending = [];
      this.inflight.clear();
      this.results.clear();
      this.closed = false;
      return sendJson(response, 200, { ok: true });
    }
    return sendJson(response, 404, { error: `unknown bench route '${route}'` });
  }
}
