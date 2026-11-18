# @youni/onnxweb-server

The server side of the `@youni/onnxweb-*` browser libraries: it serves models to the browser
(byte ranges, a catalogue, a Hugging Face Hub mirror) and can make the models nobody publishes
in ONNX (download, Optimum export, quantization, weights cut into browser-sized files) with a
Python it finds and an environment it manages.

```ts
// vite.config.ts (a local dev server)
import { onnxwebServer } from "@youni/onnxweb-server/vite";
export default defineConfig({ plugins: [onnxwebServer({ models: { prepare: true } })] });
```

```sh
npx onnxweb-server serve                       # 127.0.0.1:8765, read-only
npx onnxweb-server serve --prepare             # also makes missing models on demand
npx onnxweb-server prepare monot5-base --precision int8
npx onnxweb-server doctor
```

## Routes

| Route | What |
| --- | --- |
| `GET /models/catalog.json` | the store's entries (names only, no paths) |
| `GET /models/{entry}/{file}` | a file of an entry, byte ranges, `ETag` / `304` |
| `POST /models/prepare` | `{"name", "computeType"}` as JSON: the entry made if missing (off unless `--prepare` / `prepare: true`) |
| `/models/bench/*` | the bench queue (off unless `--bench` / `bench: true`) |
| `GET /hf/{org}/{repo}/resolve/{main or commit}/{path}` | a Hub file, from the machine's Hub cache or fetched once |
| `GET /hf/api/models/{org}/{repo}/tree/{rev}[/{path}]` | a folder listing, relayed |

## Security

Private by default; open it deliberately.

- **Loopback only.** `serve` binds `127.0.0.1`. `--host 0.0.0.0` opens it, and then a token is
  required on every request (`--token`, `ONNXWEB_SERVER_TOKEN`, or one generated and printed):
  `Authorization: Bearer <token>` (core's `serverResolver({ headers })` sends it).
- **Guard on every request**, in the standalone server and in the Vite plugin (whose
  middlewares run before Vite's own checks): the `Host` must be a loopback name or one allowed
  (`--allow-host`; DNS rebinding is refused), a cross-site request is refused (`Sec-Fetch-Site`,
  `Origin`), reads included. Another origin is served only when listed (`--allow-origin
  https://app.example`), and its answers carry that origin, never `*`.
- **Paths.** Every name is whitelisted (no `.`/`..`, no hidden names, no separators) and every
  joined path is contained in its folder after `resolve`. Malformed URLs answer 400.
- **Mirror.** The revision is `main` or a 40-hex commit; a branch is resolved to its commit and
  files are kept under that commit (a file of one revision never answers for another). In the
  Hub cache, `refs/<branch>` picks the snapshot. Only the listing API route is relayed, under a
  size cap. `--mirror-repo org/*` limits which repositories are served (the Hub cache may hold
  gated or private models downloaded with your token).
- **Exports** (`prepare`) are a JSON `POST` (a cross-site page cannot send one without a
  preflight), one at a time with a short queue, killed after two hours. Names and precisions are
  validated (a precision is one of a fixed set; the model name is passed after `--`). The
  Python requirement sets are pinned to exact versions and every Optimum recipe to a Hub commit.
  The interpreter is resolved on `PATH` to an absolute path, never from the working folder.
- **Bodies** are read under a small limit (64 KiB by default) and only as
  `application/json`; streams are piped with error handling, so a client that goes away closes
  the file.

## Licence

Apache-2.0. Ultralytics (YOLO exports) is AGPL-3.0.
