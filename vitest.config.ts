import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Node: the middleware is exercised over a real HTTP server on a free port,
    // and the Python runtime is faked by a script of a few lines.
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
