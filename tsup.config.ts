import { defineConfig } from "tsup";

// Node only: its built-ins and Vite stay external. No `clean` here: the build
// script cleans first.
export default defineConfig([
  {
    entry: {
      index: "src/index.ts",
      vite: "src/vite.ts",
      cli: "src/cli.ts",
    },
    format: ["esm"],
    platform: "node",
    target: "node20",
    dts: { entry: { index: "src/index.ts", vite: "src/vite.ts" } },
    treeshake: true,
    external: ["vite"],
  },
]);
