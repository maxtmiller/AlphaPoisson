import { spawn } from "child_process";
import http from "http";
import path from "path";
import fs from "fs";

export const runtime = "nodejs";

let serverProcess: ReturnType<typeof spawn> | null = null;
let serverReadyPromise: Promise<void> | null = null;
let watcher: fs.FSWatcher | null = null;
let restartTimeout: NodeJS.Timeout | null = null;
let stoppingPromise: Promise<void> | null = null;
let restarting = false;
let lastRestartAt: number | null = null;
let lastRestartReason: "manual" | "file-change" | null = null;

const servePort = Number(process.env.SERVE_PORT || "5058");

function getServerCwd() {
  return path.join(process.cwd(), "..");
}

function setupWatcher() {
  if (watcher) {
    return;
  }

  const serverCwd = getServerCwd();
  const srcDir = path.join(serverCwd, "src");

  if (!fs.existsSync(srcDir)) {
    return;
  }

  try {
    watcher = fs.watch(srcDir, { recursive: true }, () => {
      console.log(
        "Detected change in src directory, scheduling serve.py restart"
      );
      if (restartTimeout) {
        clearTimeout(restartTimeout);
      }
      restartTimeout = setTimeout(() => {
        restartServer("file-change");
      }, 300);
    });
  } catch (err) {
    console.error("Failed to watch src directory for changes", err);
  }
}

function stopServer(): Promise<void> {
  if (!serverProcess) {
    serverReadyPromise = null;
    return Promise.resolve();
  }

  if (stoppingPromise) {
    return stoppingPromise;
  }

  const current = serverProcess;

  stoppingPromise = new Promise<void>((resolve) => {
    current.once("exit", () => {
      serverProcess = null;
      serverReadyPromise = null;
      stoppingPromise = null;
      resolve();
    });

    if (!current.killed) {
      current.kill();
    }
  });

  return stoppingPromise;
}

function startServer() {
  if (serverProcess && !serverProcess.killed) {
    return;
  }

  const serverCwd = getServerCwd();
  const pythonExecutable =
    process.env.PYTHON_EXECUTABLE || process.env.PYTHON || "python3";

  setupWatcher();

  console.log("Starting serve.py process", { pythonExecutable, serverCwd });
  try {
    serverProcess = spawn(pythonExecutable, ["-u", "serve.py"], {
      cwd: serverCwd,
      stdio: ["ignore", "pipe", "pipe"],
      shell: false
    });

    if (serverProcess.stdout) {
      serverProcess.stdout.on("data", (data) => {
        const text = data.toString();
        if (text.trim().length > 0) {
          console.log("[serve.py stdout]", text.trimEnd());
        }
      });
    }

    if (serverProcess.stderr) {
      serverProcess.stderr.on("data", (data) => {
        const text = data.toString();
        if (text.trim().length > 0) {
          console.error("[serve.py stderr]", text.trimEnd());
        }
      });
    }
  } catch (err) {
    console.error("Failed to spawn serve.py", err);
    serverProcess = null;
    serverReadyPromise = null;
    throw err;
  }

  serverProcess.on("exit", () => {
    console.log("serve.py process exited");
    serverProcess = null;
    serverReadyPromise = null;
  });
}

function waitForServerReady(): Promise<void> {
  if (serverReadyPromise) return serverReadyPromise;

  console.log("Waiting for serve.py to become ready");
  serverReadyPromise = new Promise((resolve, reject) => {
    const maxAttempts = 30;
    const delayMs = 200;
    let attempts = 0;

    const check = () => {
      console.log("Checking serve.py readiness", { attempts });
      attempts += 1;

      const req = http.request(
      {
        host: "backend",
        port: servePort,
        path: "/", 
        method: "GET"
      },
      (res) => {
        // If the server responds at all (even a 404), it's technically "up"
        if (res.statusCode && res.statusCode < 500) { 
          console.log("serve.py is ready");
          resolve();
        } else if (attempts < maxAttempts) {
          setTimeout(check, delayMs);
        } else {
            reject(new Error("serve.py did not become ready"));
          }
        }
      );

      req.on("error", (err) => {
        console.error("Error checking serve.py readiness");
        if (attempts < maxAttempts) {
          setTimeout(check, delayMs);
        } else {
          reject(new Error("serve.py did not become ready"));
        }
      });

      req.end();
    };

    check();
  });

  return serverReadyPromise;
}

export async function ensureServerRunning() {
  if (!serverProcess || serverProcess.killed) {
    startServer();
  }
  await waitForServerReady();
}

export async function restartServer(reason?: "manual" | "file-change") {
  if (restarting) {
    return;
  }
  restarting = true;
  console.log("Restarting serve.py process");
  lastRestartAt = Date.now();
  lastRestartReason = reason ?? null;
  await stopServer();
  startServer();
  restarting = false;
}

export function getLastRestartInfo() {
  return {
    lastRestartAt,
    lastRestartReason
  };
}
