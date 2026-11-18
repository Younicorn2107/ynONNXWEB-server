/** The Python this package runs its scripts with. */

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, statSync, promises as fs } from "node:fs";
import { delimiter, dirname, isAbsolute, join, resolve } from "node:path";

import type { Logger } from "./http.js";
import { pythonEnvRoot, scriptsRoot } from "./paths.js";

/** The requirement sets shipped in `python/requirements/`. */
export type RequirementSet = "hub" | "export" | "yolo";

/** Which modules prove a set is installed, for an environment this package does not manage. */
const PROBES: Record<RequirementSet, string[]> = {
  hub: ["huggingface_hub", "onnx"],
  export: ["huggingface_hub", "onnx", "optimum.exporters.onnx", "onnxruntime", "transformers", "torch"],
  yolo: ["huggingface_hub", "onnx", "ultralytics", "onnxconverter_common", "onnxruntime"],
};

const MIN_VERSION: [number, number] = [3, 10];
/**
 * The newest Python the conversion stack ships wheels for. A machine with 3.14 and 3.12 installed
 * must get 3.12: onnxruntime had no 3.14 wheel when this was written, and pip then fails after
 * minutes of backtracking. A newer interpreter is still used when it is the only one, with a
 * warning.
 */
const MAX_PREFERRED: [number, number] = [3, 13];

export class PythonError extends Error {
  override name = "PythonError";
}

export interface PythonRuntimeOptions {
  /** An interpreter used as it is (no environment managed); `YNONNXWEB_PYTHON` by default. */
  interpreter?: string;
  /** The managed environment's folder; `YNONNXWEB_PYTHON_ENV` or `~/.cache/ynonnxweb/python`. */
  envRoot?: string;
  /** The folder of the scripts and `requirements/`; this package's `python/`. */
  scripts?: string;
  logger?: Logger | null;
}

export interface RunOptions {
  /** The requirement sets the script needs, made present first. */
  needs?: readonly RequirementSet[];
  /** Each line the script prints (stdout and stderr). */
  onLine?: (line: string) => void;
  signal?: AbortSignal;
  env?: Record<string, string>;
}

export interface RunResult {
  code: number;
  lines: string[];
}

interface Interpreter {
  command: string;
  args: string[];
  version: string;
}

/**
 * Lines of a child's output kept (the last ones): an export prints a lot, and only its end is read.
 */
const MAX_LINES = 2000;

/**
 * `name` found on `PATH`, as an absolute path, never in the current folder: on Windows a bare name
 * is looked up in the working directory first, so a server started in a folder holding a planted
 * `python.exe` would run it. Relative `PATH` entries are skipped for the same reason.
 */
export function whichOnPath(name: string, env: NodeJS.ProcessEnv = process.env): string | null {
  if (isAbsolute(name)) return existsSync(name) ? name : null;
  if (name.includes("/") || name.includes("\\")) return null;
  const cwd = resolve(process.cwd());
  const extensions = process.platform === "win32" ? (env["PATHEXT"] ?? ".EXE;.CMD;.BAT;.COM").split(";").filter(Boolean) : [""];
  for (const dir of (env["PATH"] ?? env["Path"] ?? "").split(delimiter)) {
    if (!dir || !isAbsolute(dir) || resolve(dir) === cwd) continue;
    for (const ext of extensions) {
      const candidate = join(dir, process.platform === "win32" && !name.toLowerCase().endsWith(ext.toLowerCase()) ? name + ext.toLowerCase() : name);
      try {
        if (statSync(candidate).isFile()) return candidate;
      } catch {
        // Not fatal: carry on.
      }
    }
  }
  return null;
}

/** A command's output (its last `MAX_LINES` lines); rejects when it cannot start or is aborted. */
function capture(command: string, args: readonly string[], options: { onLine?: (line: string) => void; signal?: AbortSignal; env?: Record<string, string> } = {}): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      windowsHide: true,
      env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1", PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1", ...options.env },
      ...(options.signal ? { signal: options.signal } : {}),
    });
    const lines: string[] = [];
    let pending = "";
    const read = (chunk: Buffer) => {
      pending += chunk.toString("utf8");
      const parts = pending.split(/\r?\n/);
      pending = parts.pop() ?? "";
      for (const line of parts) {
        if (!line.trim()) continue;
        lines.push(line);
        if (lines.length > MAX_LINES) lines.shift();
        options.onLine?.(line);
      }
    };
    child.stdout.on("data", read);
    child.stderr.on("data", read);
    child.on("error", reject);
    child.on("close", (code) => {
      if (pending.trim()) {
        lines.push(pending);
        options.onLine?.(pending);
      }
      resolve({ code: code ?? 1, lines });
    });
  });
}

/**
 * The candidates tried in order when no interpreter is named: the preferred versions newest first,
 * then whatever answers.
 */
function candidates(): [string, string[]][] {
  const minors: number[] = [];
  for (let minor = MAX_PREFERRED[1]; minor >= MIN_VERSION[1]; minor -= 1) minors.push(minor);
  return process.platform === "win32"
    ? [...minors.map((m): [string, string[]] => ["py", [`-3.${m}`]]), ["py", ["-3"]], ["python", []], ["python3", []]]
    : [...minors.map((m): [string, string[]] => [`python3.${m}`, []]), ["python3", []], ["python", []]];
}

const below = (version: readonly number[], limit: readonly number[]) => version[0]! < limit[0]! || (version[0] === limit[0] && version[1]! < limit[1]!);

export class PythonRuntime {
  readonly envRoot: string;
  readonly scripts: string;
  readonly managed: boolean;
  #named: string | null;
  #logger: Logger | null;
  #base: Promise<Interpreter> | null = null;

  #installing: Promise<unknown> = Promise.resolve();

  constructor(options: PythonRuntimeOptions = {}) {
    this.#named = options.interpreter ?? process.env["YNONNXWEB_PYTHON"] ?? null;
    this.managed = this.#named === null;
    this.envRoot = options.envRoot ?? pythonEnvRoot();
    this.scripts = options.scripts ?? scriptsRoot();
    this.#logger = options.logger ?? null;
  }

  /** The interpreter found on the machine (or the one named), its version checked. */
  interpreter(): Promise<Interpreter> {
    this.#base ??= this.#find();
    return this.#base;
  }

  async #find(): Promise<Interpreter> {
    const tried: string[] = [];
    const list: [string, string[]][] = this.#named ? [[this.#named, []]] : candidates();
    let newer: Interpreter | null = null;
    for (const [name, args] of list) {
      const label = [name, ...args].join(" ");
      const command = whichOnPath(name);
      if (!command) {
        tried.push(label);
        continue;
      }
      const probe = await capture(command, [...args, "-B", "-c", "import sys; print(sys.executable); print('%d.%d.%d' % sys.version_info[:3])"]).catch(() => null);
      const version = probe?.code === 0 ? probe.lines.at(-1)?.trim() : undefined;
      const executable = probe?.code === 0 ? probe.lines.at(-2)?.trim() : undefined;
      if (!version || !/^\d+\.\d+\.\d+$/.test(version)) {
        tried.push(label);
        continue;
      }
      const numbers = version.split(".").map(Number);
      if (below(numbers, MIN_VERSION)) {
        tried.push(`${label} (${version}, too old)`);
        continue;
      }

      const found: Interpreter = executable && isAbsolute(executable) && resolve(dirname(executable)) !== resolve(process.cwd()) ? { command: executable, args: [], version } : { command, args, version };
      if (!this.#named && !below(numbers, [MAX_PREFERRED[0], MAX_PREFERRED[1] + 1])) {
        newer ??= found;
        continue;
      }
      return found;
    }
    if (newer) {
      this.#logger?.warn(`  onnxweb-server: only Python ${newer.version} found; the conversion stack may have no wheels for it yet (3.${MAX_PREFERRED[1]} or older is safer)`);
      return newer;
    }
    throw new PythonError(
      `no Python ${MIN_VERSION.join(".")}+ found (tried ${tried.join(", ")}). ` +
        "Install Python (this package does not), or name one with YNONNXWEB_PYTHON.",
    );
  }

  /** The managed environment's interpreter. */
  get envPython(): string {
    return process.platform === "win32" ? join(this.envRoot, "Scripts", "python.exe") : join(this.envRoot, "bin", "python");
  }

  /** What `run` will execute: the managed environment's Python, or the named one. */
  async #python(): Promise<Interpreter> {
    if (!this.managed) return this.interpreter();
    return { command: this.envPython, args: [], version: (await this.interpreter()).version };
  }

  /**
   * The requirement sets present, installed when missing (managed environment) or reported with the
   * command that installs them (named interpreter).
   */
  async ensure(sets: readonly RequirementSet[]): Promise<void> {
    if (sets.length === 0) return;
    if (!this.managed) return this.#check(sets);
    const run = this.#installing.then(() => this.#install(sets));
    this.#installing = run.catch(() => undefined);
    return run;
  }

  async #check(sets: readonly RequirementSet[]): Promise<void> {
    const python = await this.interpreter();
    const modules = [...new Set(sets.flatMap((s) => PROBES[s]))];
    const script = `import importlib.util as u, sys\nmissing = [m for m in ${JSON.stringify(modules)} if u.find_spec(m.split('.')[0]) is None]\nprint(' '.join(missing))`;
    const probe = await capture(python.command, [...python.args, "-B", "-c", script]);
    const missing = probe.lines.at(-1)?.trim();
    if (missing) {
      const files = sets.map((s) => join(this.scripts, "requirements", `${s}.txt`));
      throw new PythonError(
        `${python.command} lacks ${missing.split(" ").join(", ")}: ` +
          `install them (${python.command} -m pip install ${files.map((f) => `-r "${f}"`).join(" ")}), ` +
          "or unset YNONNXWEB_PYTHON to let this package manage an environment of its own",
      );
    }
  }

  async #install(sets: readonly RequirementSet[]): Promise<void> {
    const base = await this.interpreter();
    if (!existsSync(this.envPython)) {
      this.#logger?.info(`  onnxweb-server: creating a Python environment in ${this.envRoot} (Python ${base.version})`);
      const made = await capture(base.command, [...base.args, "-B", "-m", "venv", this.envRoot]);
      if (made.code !== 0 || !existsSync(this.envPython)) {
        throw new PythonError(`could not create a virtual environment in ${this.envRoot}: ${made.lines.slice(-3).join(" | ")}`);
      }
    }
    const stampFile = join(this.envRoot, "onnxweb-server.json");
    const stamps = JSON.parse(await fs.readFile(stampFile, "utf8").catch(() => "{}")) as Record<string, string>;
    for (const set of sets) {
      const file = join(this.scripts, "requirements", `${set}.txt`);
      const digest = createHash("sha256").update(await this.#requirementText(file)).digest("hex");
      if (stamps[set] === digest) continue;
      this.#logger?.info(`  onnxweb-server: installing the '${set}' Python requirements (once)`);
      const installed = await capture(this.envPython, ["-B", "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "-r", file], {
        onLine: (line) => {
          if (/^(Successfully|ERROR)/.test(line)) this.#logger?.info(`    pip: ${line}`);
        },
      });
      if (installed.code !== 0) {
        throw new PythonError(`pip could not install the '${set}' requirements: ${installed.lines.slice(-4).join(" | ")}`);
      }
      stamps[set] = digest;
      await fs.writeFile(stampFile, JSON.stringify(stamps, null, 1));
    }
  }

  /** A requirement file with the files it includes (`-r`), so a change in one reinstalls. */
  async #requirementText(file: string): Promise<string> {
    const text = await fs.readFile(file, "utf8");
    const included = [...text.matchAll(/^-r\s+(\S+)/gm)].map((m) => join(file, "..", m[1]!));
    return text + (await Promise.all(included.map((f) => this.#requirementText(f)))).join("\n");
  }

  /** One of this package's scripts (`python/<script>`), its requirement sets made present first. */
  async run(script: string, args: readonly string[], options: RunOptions = {}): Promise<RunResult> {
    await this.ensure(options.needs ?? []);
    const python = await this.#python();
    return capture(python.command, [...python.args, "-B", join(this.scripts, script), ...args], options);
  }

  /** What this runtime would use, and what is installed: for `onnxweb-server doctor`. */
  async describe(): Promise<{ interpreter: string; version: string; managed: boolean; envRoot: string | null; installed: string[] }> {
    const base = await this.interpreter();
    const stamps = this.managed
      ? (JSON.parse(await fs.readFile(join(this.envRoot, "onnxweb-server.json"), "utf8").catch(() => "{}")) as Record<string, string>)
      : {};
    return {
      interpreter: [base.command, ...base.args].join(" "),
      version: base.version,
      managed: this.managed,
      envRoot: this.managed ? this.envRoot : null,
      installed: Object.keys(stamps),
    };
  }
}
