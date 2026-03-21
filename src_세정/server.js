// src/server.js
console.log("[SERVER.JS PATH]", new URL(import.meta.url).pathname);

import { spawn } from "child_process";
import http from "http";
import open from "open";
import path from "path";
import fs from "fs";

// ====== PATH 세팅 ======
// 현재 실행 위치 예시:
// C:\Users\user\Desktop\new\VideoPose3D\pose-web
// 이 기준에서 한 단계 위가 VideoPose3D 루트
const HERE = process.cwd();
const ROOT_DIR = path.resolve(HERE, "../"); // ✅ VideoPose3D 루트

// 루트에 있는 vp3d_ws_server.py를 실행하도록 고정
const PY_FILE_ABS = path.join(ROOT_DIR, "vp3d_ws_server.py");

// (선택) venv python을 확실히 쓰고 싶으면 아래를 우선 사용
// vpose_env가 루트에 있다는 가정: VideoPose3D/vpose_env/Scripts/python.exe
const VENV_PY_ABS = path.join(ROOT_DIR, "vpose_env", "Scripts", "python.exe");

// Windows면 python.exe 우선순위: venv python -> 시스템 python
const isWin = process.platform === "win32";
const PY_CMD = isWin && fs.existsSync(VENV_PY_ABS)
  ? VENV_PY_ABS
  : (isWin ? "python" : "python3");

// Vite
const VITE_CMD = isWin ? "npm.cmd" : "npm";
const VITE_PORT = 5173;
const VITE_URL = `http://localhost:${VITE_PORT}/`;

// ====== 디버그 로그 ======
console.log("[HERE]", HERE);
console.log("[ROOT_DIR]", ROOT_DIR);
console.log("[PY_FILE_ABS]", PY_FILE_ABS);
console.log("[VENV_PY_ABS]", VENV_PY_ABS);
console.log("[PY_CMD]", PY_CMD);

// ====== 유틸 ======
function existsOrDie(filePath, label) {
  if (!fs.existsSync(filePath)) {
    console.error(`\n[ERROR] ${label} not found:\n  ${filePath}\n`);
    process.exit(1);
  }
}

function runProc(cmd, args, name, cwd) {
  console.log(`[RUN] ${name}`);
  console.log(`      cmd: ${cmd}`);
  console.log(`      args: ${args.join(" ")}`);
  console.log(`      cwd: ${cwd}`);

  const p = spawn(cmd, args, {
    stdio: "inherit",
    shell: isWin,
    cwd,
  });

  p.on("close", (code) => console.log(`[${name}] exited: ${code}`));
  return p;
}

// ====== 사전 체크 ======
existsOrDie(PY_FILE_ABS, "vp3d_ws_server.py");
existsOrDie(path.join(ROOT_DIR, "common"), "VideoPose3D/common folder");

// ====== 1) VP3D WS 서버 (루트에서 실행) ======
const vp3d = runProc(PY_CMD, [PY_FILE_ABS], "VP3D", ROOT_DIR);

// ====== 2) Vite dev 서버 (현재 폴더에서 실행) ======
const vite = runProc(
  VITE_CMD,
  ["run", "dev", "--", "--port", String(VITE_PORT)],
  "VITE",
  HERE
);

// ====== 3) Vite 준비되면 브라우저 자동 오픈 ======
function waitForServer(url, timeoutMs = 30000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      http
        .get(url, (res) => {
          res.resume();
          resolve(true);
        })
        .on("error", () => {
          if (Date.now() - start > timeoutMs) {
            reject(new Error("Vite not ready"));
          } else {
            setTimeout(tick, 250);
          }
        });
    };
    tick();
  });
}

(async () => {
  try {
    await waitForServer(VITE_URL, 30000);
    await open(VITE_URL);
    console.log(`[OPEN] ${VITE_URL}`);
  } catch (e) {
    console.log("[OPEN] failed:", e.message);
    console.log("수동으로 열기:", VITE_URL);
  }
})();

// ====== 4) 종료 처리(같이 죽이기) ======
function shutdown() {
  console.log("\n[SHUTDOWN] stopping processes...");
  try {
    vp3d.kill("SIGINT");
  } catch {}
  try {
    vite.kill("SIGINT");
  } catch {}
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);