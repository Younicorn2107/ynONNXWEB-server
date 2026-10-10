# onnxweb-server

The server side: serve models to the browser libraries, and make the ones nobody publishes in ONNX.

- **Model store** at `/models/`: byte ranges, a catalogue, and `prepare`, which exports a missing model on demand.
- **Hub mirror** at `/hf/`: files from the machine's Hugging Face cache, fetched once otherwise.
- **Embedded Python**: the export scripts ship with the package (Optimum, ONNX Runtime quantization, Ultralytics). Python 3.10+ must be installed — this package finds it and never installs it — and its dependencies go into a virtual environment of its own, installed on first need.
- **Mounts** as a Vite plugin, connect middleware or a standalone server; a CLI.

```ts
// vite.config.ts
import { onnxwebServer } from "@youni/onnxweb-server/vite";
export default defineConfig({ plugins: [onnxwebServer()] });
```

```sh
npx onnxweb-server serve --port 8765
npx onnxweb-server prepare monot5-base --precision int8
npx onnxweb-server doctor
```

## Install

Build and test:

```sh
npm install
npm run build   # dist/
npm test        # unit tests (Node, onnxruntime-web on WebAssembly)
```

## Licence

Apache-2.0 for the code. Model weights keep their own licence (listed above where it matters).
