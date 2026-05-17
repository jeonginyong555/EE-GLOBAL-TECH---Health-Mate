/* eslint-disable no-console */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/** =========================
 * UI
 * ========================= */
const VIEW_W = 400;
const VIEW_H = 300;
const WEBCAM_DRAW_INTERVAL = 33; // 약 30fps

const app = document.querySelector("#app");

app.innerHTML = `
  <div style="font-family:sans-serif; padding:12px;">
    <h2 style="margin:0 0 8px;">Health-Mate Preview (WebYOLO + Gym Animation)</h2>

    <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start;">
      <!-- LEFT -->
      <div style="min-width:${VIEW_W}px;">
        <div style="margin:8px 0 6px; font-weight:600;">Video</div>
        <video id="video" autoplay playsinline style="width:${VIEW_W}px; border:1px solid #ddd;"></video>

        <div style="margin:10px 0 6px; font-weight:600;">Canvas (2D: YOLO)</div>
        <div style="position:relative; width:${VIEW_W}px; height:${VIEW_H}px; border:1px solid #ddd;">
          <canvas id="canvas" width="${VIEW_W}" height="${VIEW_H}" style="position:absolute; left:0; top:0;"></canvas>
          <canvas id="overlay" width="${VIEW_W}" height="${VIEW_H}" style="position:absolute; left:0; top:0; pointer-events:none;"></canvas>
        </div>

        <div style="margin-top:6px; font-size:12px; color:#666;">
          <span style="display:inline-block;width:10px;height:10px;background:#0f0;margin-right:6px;"></span>YOLO lines
          &nbsp;&nbsp;
          <span style="display:inline-block;width:10px;height:10px;background:#f00;margin-right:6px;"></span>YOLO joints
        </div>

        <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
          <button id="start">Start Webcam</button>
          <button id="stop">Stop</button>
          <button id="loadModel">Load ONNX (Worker)</button>
          <button id="infer">Infer 1 Frame</button>
          <button id="live">Live ON/OFF</button>
          <button id="clear">Clear</button>
        </div>

        <div style="margin-top:8px; font-size:12px; color:#666; line-height:1.4;">
          ONNX: <code>/models/yolo_pose.onnx</code> (public/models)<br/>
          Worker: <code>/src/yoloWorker.js</code><br/>
          입력 해상도: <code>640x640 유지</code><br/>
          차렷 성공 후: <code>YOLO 실시간 피드백 모드</code>
        </div>
      </div>

      <!-- RIGHT -->
      <div style="min-width:640px; flex:1;">
        <div style="margin:8px 0 6px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <button id="loadScene">Load Scene</button>
          <button id="toggleAnim">Anim ON/OFF</button>
          <button id="resetAnim">Anim Reset</button>
          <label style="display:flex; gap:6px; align-items:center;">
            <input id="autoRotate" type="checkbox" />
            Auto Rotate Camera
          </label>
          <span id="threeStatus" style="color:#555;">Ready</span>
        </div>

        <div id="threeWrap" style="width:920px; max-width:100%; height:520px; border:1px solid #ddd;"></div>

        <div style="margin-top:10px; font-weight:600;">Info</div>
        <pre id="info" style="background:#0f0f10; color:#9ef; padding:10px; border-radius:8px; max-width:920px; overflow:auto; height:220px;">(no data)</pre>

        <div style="margin-top:8px; color:#888; font-size:12px; line-height:1.4;">
          Gym GLB: <code>/models/Untitled_gym.glb</code><br/>
          Anim GLB: <code>/models/Untitled_squat.glb</code><br/>
          시작 조건: <code>차렷 자세 5초 유지</code><br/>
          차렷 후: <code>실시간 카운트/피드백 모드</code>
        </div>
      </div>
    </div>
  </div>
`;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const infoEl = document.getElementById("info");
const threeStatusEl = document.getElementById("threeStatus");
const autoRotateEl = document.getElementById("autoRotate");

let infoTick = 0;
function logInfo(msg) {
  infoEl.textContent = String(msg);
}

function logInfoThrottled(msg, every = 4) {
  infoTick += 1;
  if (infoTick % every !== 0) return;
  infoEl.textContent = String(msg);
}

function setThreeStatus(msg) {
  threeStatusEl.textContent = msg;
}

/** =========================
 * 상태 변수
 * ========================= */
let stream = null;
let rafId = null;
let webcamTimer = null;
let liveOn = false;
let liveTimer = null;
let workerBusy = false;

let lastYoloKpts = null;
let displayKpts = null;
let overlayAnimId = null;

let lastStableKpts = null;

let yoloWorker = null;
let workerReady = false;

/** =========================
 * 차렷 5초 게이트 상태
 * ========================= */
let animationPaused = true;
let animReadyGateOpen = false;
let standHoldStart = null;
const STAND_HOLD_MS = 5000;

/** =========================
 * 성능 튜닝
 * ========================= */
const INFER_INTERVAL_PRE_GATE = 80;
const INFER_INTERVAL_POST_GATE = 80;

/** =========================
 * 스쿼트 카운트/피드백 상태
 * ========================= */
let squatCount = 0;
let squatState = "UP";
let lastDepth = 0;
let lastKneeValgus = 0;
let lastTorsoLean = 0;
let lastFeedback = "대기 중";

/** =========================
 * COCO skeleton edges
 * ========================= */
const COCO_EDGES = [
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [11, 12],
  [5, 11],
  [6, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

/** =========================
 * Worker
 * ========================= */
function initWorkerIfNeeded() {
  if (yoloWorker) return;

  yoloWorker = new Worker(new URL("./yoloWorker.js", import.meta.url), {
    type: "module",
  });

  yoloWorker.onmessage = (e) => {
    const { type, payload } = e.data;

    if (type === "inited") {
      workerReady = true;
      setThreeStatus("ONNX loaded (worker)");
      logInfo("ONNX loaded in WebWorker");
      return;
    }

    if (type === "inferResult") {
      workerBusy = false;

      if (!payload.ok) {
        logInfoThrottled("사람 검출 실패 또는 conf 부족", 4);
        return;
      }

      lastYoloKpts = payload.kpts;
      lastYoloKpts = stabilizeKptsLight(lastYoloKpts);

      if (!animReadyGateOpen) {
        updateAttentionGate(lastYoloKpts);

        logInfoThrottled(
          `YOLO conf=${payload.conf.toFixed(3)}\n` +
            `infer=${payload.inferMs.toFixed(1)}ms\n` +
            `차렷 자세를 5초 유지하세요`,
          4
        );
        setThreeStatus(`차렷 대기 중 / YOLO ${payload.conf.toFixed(3)}`);
      } else {
        updateSquatCounterAndFeedback(lastYoloKpts);

        logInfoThrottled(
          `YOLO conf=${payload.conf.toFixed(3)}\n` +
            `infer=${payload.inferMs.toFixed(1)}ms\n` +
            `분석 모드: 실시간 피드백\n\n` +
            `squatCount=${squatCount}\n` +
            `depth=${lastDepth.toFixed(2)}\n` +
            `torsoLean=${lastTorsoLean.toFixed(2)}\n` +
            `kneeValgus=${lastKneeValgus.toFixed(2)}\n` +
            `feedback=${lastFeedback}`,
          4
        );

        setThreeStatus(`분석 중 / count ${squatCount} / ${lastFeedback}`);
      }

      return;
    }

    if (type === "error") {
      workerBusy = false;
      setThreeStatus("Worker error");
      logInfo(`Worker error: ${payload.message}`);
    }
  };
}

/** =========================
 * 2D Drawing
 * ========================= */
function clearOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawThresholdByIndex(idx) {
  if ([13, 14, 15, 16].includes(idx)) return 0.18;
  if ([11, 12].includes(idx)) return 0.22;
  return 0.28;
}

function drawSkeleton2D(kpts, lineColor, dotColor, lineWidth = 1.5, dotRadius = 3) {
  octx.lineWidth = lineWidth;
  octx.strokeStyle = lineColor;
  octx.fillStyle = dotColor;

  for (const [a, b] of COCO_EDGES) {
    const pa = kpts[a];
    const pb = kpts[b];
    if (!pa || !pb) continue;
    if (pa.s < drawThresholdByIndex(a) || pb.s < drawThresholdByIndex(b)) continue;

    octx.beginPath();
    octx.moveTo(pa.x, pa.y);
    octx.lineTo(pb.x, pb.y);
    octx.stroke();
  }

  for (let i = 0; i < kpts.length; i++) {
    const p = kpts[i];
    if (!p || p.s < drawThresholdByIndex(i)) continue;
    octx.beginPath();
    octx.arc(p.x, p.y, dotRadius, 0, Math.PI * 2);
    octx.fill();
  }
}

/** =========================
 * Overlay smoothing
 * ========================= */
function lerp(a, b, t) {
  return a + (b - a) * t;
}

function smoothDisplayKpts(targetKpts) {
  if (!targetKpts) return null;

  if (!displayKpts) {
    displayKpts = targetKpts.map((p) => ({ ...p }));
    return displayKpts;
  }

  for (let i = 0; i < targetKpts.length; i++) {
    const src = displayKpts[i];
    const dst = targetKpts[i];
    if (!src || !dst) continue;

    const alpha = [13, 14, 15, 16].includes(i) ? 0.75 : 0.92;

    const dx = dst.x - src.x;
    const dy = dst.y - src.y;
    const dist = Math.hypot(dx, dy);

    if (dist > 12) {
      src.x = dst.x;
      src.y = dst.y;
      src.s = dst.s;
    } else {
      src.x = lerp(src.x, dst.x, alpha);
      src.y = lerp(src.y, dst.y, alpha);
      src.s = lerp(src.s, dst.s, alpha);
    }
  }

  return displayKpts;
}

function animateOverlay() {
  overlayAnimId = requestAnimationFrame(animateOverlay);

  if (!lastYoloKpts) {
    clearOverlay();
    return;
  }

  const smoothed = smoothDisplayKpts(lastYoloKpts);
  if (!smoothed) return;

  clearOverlay();
  drawSkeleton2D(smoothed, "lime", "red", 1.5, 3);
}

function resetOverlayState() {
  lastYoloKpts = null;
  displayKpts = null;
  clearOverlay();
}

/** =========================
 * 공통 유틸
 * ========================= */
function dist2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function isConfident(p, th = 0.3) {
  return p && p.s >= th;
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

/** =========================
 * 키포인트 안정화
 * ========================= */
function stabilizeKptsLight(kpts) {
  if (!kpts) return kpts;

  const out = kpts.map((p) => ({ ...p }));

  if (!lastStableKpts) {
    lastStableKpts = out.map((p) => ({ ...p }));
    return out;
  }

  const importantLower = [13, 14, 15, 16];

  for (let i = 0; i < out.length; i++) {
    const cur = out[i];
    const prev = lastStableKpts[i];
    if (!cur || !prev) continue;

    const isLower = importantLower.includes(i);

    if (isLower && cur.s < 0.28) {
      cur.x = prev.x;
      cur.y = prev.y;
      cur.s = Math.max(cur.s, prev.s * 0.95);
    } else {
      const alpha = isLower ? 0.65 : 0.8;
      cur.x = prev.x * (1 - alpha) + cur.x * alpha;
      cur.y = prev.y * (1 - alpha) + cur.y * alpha;
      cur.s = prev.s * (1 - alpha) + cur.s * alpha;
    }
  }

  lastStableKpts = out.map((p) => ({ ...p }));
  return out;
}

function resetStableKpts() {
  lastStableKpts = null;
}

/** =========================
 * 차렷 자세 판정
 * ========================= */
function isAttentionPose(kpts) {
  if (!kpts || kpts.length < 17) return false;

  const lShoulder = kpts[5];
  const rShoulder = kpts[6];
  const lWrist = kpts[9];
  const rWrist = kpts[10];
  const lHip = kpts[11];
  const rHip = kpts[12];
  const lKnee = kpts[13];
  const rKnee = kpts[14];

  if (
    !isConfident(lShoulder) ||
    !isConfident(rShoulder) ||
    !isConfident(lWrist) ||
    !isConfident(rWrist) ||
    !isConfident(lHip) ||
    !isConfident(rHip) ||
    !isConfident(lKnee) ||
    !isConfident(rKnee)
  ) {
    return false;
  }

  const shoulderWidth = dist2D(lShoulder, rShoulder);
  if (shoulderWidth < 20) return false;

  const hipCenter = {
    x: (lHip.x + rHip.x) * 0.5,
    y: (lHip.y + rHip.y) * 0.5,
  };

  const shoulderCenter = {
    x: (lShoulder.x + rShoulder.x) * 0.5,
    y: (lShoulder.y + rShoulder.y) * 0.5,
  };

  const leftWristNearBody =
    Math.abs(lWrist.x - lHip.x) < shoulderWidth * 1.2 &&
    lWrist.y > lShoulder.y &&
    lWrist.y < lKnee.y;

  const rightWristNearBody =
    Math.abs(rWrist.x - rHip.x) < shoulderWidth * 1.2 &&
    rWrist.y > rShoulder.y &&
    rWrist.y < rKnee.y;

  const torsoLean = Math.abs(shoulderCenter.x - hipCenter.x);
  const torsoUpright = torsoLean < shoulderWidth * 0.7;

  return leftWristNearBody && rightWristNearBody && torsoUpright;
}

function updateAttentionGate(kpts) {
  if (animReadyGateOpen) return;

  const now = performance.now();
  const ok = isAttentionPose(kpts);

  if (ok) {
    if (standHoldStart === null) {
      standHoldStart = now;
    }

    const held = now - standHoldStart;
    const remain = Math.max(0, STAND_HOLD_MS - held);

    setThreeStatus(`차렷 자세 유지 중... ${(remain / 1000).toFixed(1)}초`);
    logInfoThrottled(
      `차렷 자세 인식 중...\n` +
        `남은 시간: ${(remain / 1000).toFixed(1)}초\n` +
        `5초 유지하면 애니메이션 시작`,
      4
    );

    if (held >= STAND_HOLD_MS) {
      animReadyGateOpen = true;
      animationPaused = false;

      if (liveOn) {
        resetLiveTimer();
      }

      setThreeStatus("차렷 인식 완료 → 애니메이션 시작 (실시간 피드백 모드)");
      logInfo(
        "차렷 자세 5초 유지 성공\n" +
          "애니메이션을 시작합니다.\n" +
          "이제 YOLO가 빠르게 돌면서 카운트/피드백을 계산합니다."
      );
    }
  } else {
    standHoldStart = null;
    if (avatarScene && !animReadyGateOpen) {
      setThreeStatus("차렷 자세를 5초 유지하세요");
    }
  }
}

/** =========================
 * 스쿼트 분석
 * ========================= */
function computeSquatMetrics(kpts) {
  if (!kpts || kpts.length < 17) return null;

  const ls = kpts[5];
  const rs = kpts[6];
  const lh = kpts[11];
  const rh = kpts[12];
  const lk = kpts[13];
  const rk = kpts[14];
  const la = kpts[15];
  const ra = kpts[16];

  if (
    !isConfident(ls) ||
    !isConfident(rs) ||
    !isConfident(lh) ||
    !isConfident(rh) ||
    !isConfident(lk) ||
    !isConfident(rk) ||
    !isConfident(la) ||
    !isConfident(ra)
  ) {
    return null;
  }

  const hipCenter = { x: (lh.x + rh.x) * 0.5, y: (lh.y + rh.y) * 0.5 };
  const kneeCenter = { x: (lk.x + rk.x) * 0.5, y: (lk.y + rk.y) * 0.5 };
  const shoulderCenter = { x: (ls.x + rs.x) * 0.5, y: (ls.y + rs.y) * 0.5 };

  const thighLenL = dist2D(lh, lk);
  const thighLenR = dist2D(rh, rk);
  const thighLen = (thighLenL + thighLenR) * 0.5;

  const shinLenL = dist2D(lk, la);
  const shinLenR = dist2D(rk, ra);
  const shinLen = (shinLenL + shinLenR) * 0.5;

  const bodyScale = Math.max(40, (thighLen + shinLen) * 0.5);

  const hipToKneeY = kneeCenter.y - hipCenter.y;
  const depth = clamp01(1 - hipToKneeY / Math.max(1, thighLen));

  const torsoLean =
    Math.abs(shoulderCenter.x - hipCenter.x) / Math.max(1, bodyScale);

  const kneeDist = dist2D(lk, rk);
  const ankleDist = dist2D(la, ra);
  const kneeValgus = ankleDist > 1 ? kneeDist / ankleDist : 1;

  return {
    depth,
    torsoLean,
    kneeValgus,
  };
}

function updateSquatCounterAndFeedback(kpts) {
  const m = computeSquatMetrics(kpts);
  if (!m) {
    lastFeedback = "관절 인식 부족";
    return;
  }

  lastDepth = m.depth;
  lastTorsoLean = m.torsoLean;
  lastKneeValgus = m.kneeValgus;

  if (squatState === "UP" && m.depth >= 0.62) {
    squatState = "DOWN";
  } else if (squatState === "DOWN" && m.depth <= 0.32) {
    squatState = "UP";
    squatCount += 1;
  }

  const feedbacks = [];

  if (m.depth < 0.45) {
    feedbacks.push("깊이 부족");
  } else {
    feedbacks.push("깊이 양호");
  }

  if (m.torsoLean > 0.38) {
    feedbacks.push("허리/상체 숙임 큼");
  } else {
    feedbacks.push("상체 각도 양호");
  }

  if (m.kneeValgus < 0.75) {
    feedbacks.push("무릎이 안쪽으로 모임");
  } else {
    feedbacks.push("무릎 정렬 양호");
  }

  lastFeedback = feedbacks.join(" / ");
}

/** =========================
 * Webcam
 * ========================= */
async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
      audio: false,
    });

    video.srcObject = stream;
    await new Promise((resolve) => {
      video.onloadedmetadata = resolve;
    });

    const drawFrame = () => {
      const vw = video.videoWidth || 1;
      const vh = video.videoHeight || 1;
      const cw = canvas.width;
      const ch = canvas.height;

      const videoAspect = vw / vh;
      const canvasAspect = cw / ch;

      let sx = 0;
      let sy = 0;
      let sw = vw;
      let sh = vh;

      if (videoAspect > canvasAspect) {
        sw = vh * canvasAspect;
        sx = (vw - sw) / 2;
      } else {
        sh = vw / canvasAspect;
        sy = (vh - sh) / 2;
      }

      ctx.drawImage(video, sx, sy, sw, sh, 0, 0, cw, ch);
    };

    if (webcamTimer) {
      clearInterval(webcamTimer);
      webcamTimer = null;
    }

    drawFrame();
    webcamTimer = setInterval(drawFrame, WEBCAM_DRAW_INTERVAL);

    setThreeStatus("웹캠 시작됨");
    logInfo("웹캠 시작 완료");
  } catch (err) {
    console.error(err);
    setThreeStatus("Webcam error");
    logInfo("Webcam error: " + err.message);
  }
}

function stopWebcam() {
  if (webcamTimer) {
    clearInterval(webcamTimer);
    webcamTimer = null;
  }

  setLive(false);

  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }

  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  resetOverlayState();
  resetStableKpts();

  standHoldStart = null;
  animReadyGateOpen = false;
  animationPaused = true;

  squatCount = 0;
  squatState = "UP";
  lastDepth = 0;
  lastTorsoLean = 0;
  lastKneeValgus = 0;
  lastFeedback = "대기 중";

  setThreeStatus("웹캠 정지");
  logInfo("웹캠 정지");
}

/** =========================
 * Model (Worker)
 * ========================= */
async function loadModel() {
  try {
    initWorkerIfNeeded();
    workerReady = false;

    yoloWorker.postMessage({
      type: "init",
      payload: {
        modelUrl: "/models/yolo_pose.onnx",
      },
    });

    setThreeStatus("Loading ONNX in worker...");
    logInfo("Loading ONNX in WebWorker...");
  } catch (e) {
    console.error(e);
    setThreeStatus("Model load error");
    logInfo("Model load error: " + e.message);
  }
}

async function inferOnce() {
  if (!workerReady) {
    logInfo("Load ONNX 먼저.");
    return;
  }
  if (!stream) {
    logInfo("Webcam 먼저.");
    return;
  }
  if (workerBusy) return;

  try {
    workerBusy = true;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    yoloWorker.postMessage({
      type: "infer",
      payload: {
        imageData,
        width: canvas.width,
        height: canvas.height,
      },
    });
  } catch (e) {
    workerBusy = false;
    console.error(e);
    setThreeStatus("Infer error");
    logInfo("Infer error: " + e.message);
  }
}

/** =========================
 * Live 제어
 * ========================= */
function getInferIntervalMs() {
  return animReadyGateOpen ? INFER_INTERVAL_POST_GATE : INFER_INTERVAL_PRE_GATE;
}

async function liveLoop() {
  if (!liveOn) return;

  if (!workerBusy) {
    await inferOnce();
  }

  if (!liveOn) return;
  liveTimer = setTimeout(liveLoop, getInferIntervalMs());
}

function resetLiveTimer() {
  if (!liveOn) return;

  if (liveTimer) {
    clearTimeout(liveTimer);
    liveTimer = null;
  }

  liveTimer = setTimeout(liveLoop, getInferIntervalMs());
}

function setLive(on) {
  liveOn = on;

  if (liveTimer) {
    clearTimeout(liveTimer);
    liveTimer = null;
  }

  if (!liveOn) return;
  resetLiveTimer();
}

/** =========================
 * Three.js Scene (Gym + Animated Avatar)
 * ========================= */
const THREE_WRAP = document.getElementById("threeWrap");

const GYM_URL = "/models/Untitled_gym.glb";
const ANIM_AVATAR_URL = "/models/Untitled_squat.glb";

let renderer = null;
let scene = null;
let camera = null;
let controls = null;

let gymRoot = null;
let avatarScene = null;
let mixer = null;

const clock = new THREE.Clock();
let lastRenderTime = 0;
const RENDER_FPS = 15;
const RENDER_INTERVAL = 1000 / RENDER_FPS;

const AVATAR_POS = new THREE.Vector3(0, 0, 0.8);
const LOOK_TARGET_OFFSET = new THREE.Vector3(0, 1.15, 0);
const FRONT_CAM_OFFSET = new THREE.Vector3(0, 0.35, 2.75);

function initThreeIfNeeded() {
  if (renderer) return;

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(1);
  renderer.setSize(THREE_WRAP.clientWidth, THREE_WRAP.clientHeight);
  THREE_WRAP.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf7f7f7);

  camera = new THREE.PerspectiveCamera(
    50,
    THREE_WRAP.clientWidth / THREE_WRAP.clientHeight,
    0.1,
    2000
  );
  camera.position.set(0, 1.5, 3.2);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enabled = false;
  controls.enableDamping = false;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x333344, 1.0);
  hemi.position.set(0, 2, 0);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(2, 4, 2);
  scene.add(dir);

  window.addEventListener("resize", () => {
    if (!renderer) return;
    renderer.setSize(THREE_WRAP.clientWidth, THREE_WRAP.clientHeight);
    camera.aspect = THREE_WRAP.clientWidth / THREE_WRAP.clientHeight;
    camera.updateProjectionMatrix();
  });

  animate();
}

function getAvatarLookTarget() {
  return AVATAR_POS.clone().add(LOOK_TARGET_OFFSET);
}

function animate(now = 0) {
  rafId = requestAnimationFrame(animate);

  const delta = clock.getDelta();

  if (mixer && !animationPaused) {
    mixer.update(delta);
  }

  if (autoRotateEl && autoRotateEl.checked && camera && avatarScene) {
    const target = getAvatarLookTarget();
    const t = now * 0.0004;
    const radius = 2.75;
    camera.position.x = target.x + Math.sin(t) * radius;
    camera.position.z = target.z + Math.cos(t) * radius;
    camera.position.y = target.y + 0.35;
    camera.lookAt(target);
  }

  if (now - lastRenderTime < RENDER_INTERVAL) return;
  lastRenderTime = now;

  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

async function loadGym(loader) {
  return new Promise((resolve, reject) => {
    loader.load(
      GYM_URL,
      (gltf) => {
        if (gymRoot) scene.remove(gymRoot);
        gymRoot = gltf.scene;

        gymRoot.traverse((obj) => {
          if (obj.isMesh && obj.material) {
            obj.material.side = THREE.DoubleSide;
          }
        });

        scene.add(gymRoot);
        resolve();
      },
      undefined,
      reject
    );
  });
}

async function loadAnimatedAvatar(loader) {
  return new Promise((resolve, reject) => {
    loader.load(
      ANIM_AVATAR_URL,
      (gltf) => {
        if (avatarScene) scene.remove(avatarScene);

        avatarScene = gltf.scene;
        avatarScene.position.copy(AVATAR_POS);
        scene.add(avatarScene);

        mixer = null;
        animationPaused = true;
        animReadyGateOpen = false;
        standHoldStart = null;
        resetStableKpts();
        resetOverlayState();

        squatCount = 0;
        squatState = "UP";
        lastDepth = 0;
        lastTorsoLean = 0;
        lastKneeValgus = 0;
        lastFeedback = "대기 중";

        if (gltf.animations && gltf.animations.length > 0) {
          mixer = new THREE.AnimationMixer(avatarScene);
          const action = mixer.clipAction(gltf.animations[0]);
          action.setLoop(THREE.LoopRepeat);
          action.timeScale = 0.3;
          action.play();
        }

        const target = getAvatarLookTarget();

        camera.position.set(
          target.x + FRONT_CAM_OFFSET.x,
          target.y + FRONT_CAM_OFFSET.y,
          target.z + FRONT_CAM_OFFSET.z
        );
        camera.lookAt(target);

        setThreeStatus("차렷 자세를 5초 유지하면 애니메이션 시작");
        logInfo("장면 로드 완료\n차렷 자세를 5초 유지하세요");

        resolve();
      },
      undefined,
      reject
    );
  });
}

async function loadScene() {
  try {
    initThreeIfNeeded();
    setThreeStatus("Loading gym + animation...");

    const loader = new GLTFLoader();
    await loadGym(loader);
    await loadAnimatedAvatar(loader);

    setThreeStatus("장면 로드 완료 - 차렷 자세를 5초 유지하세요");
  } catch (e) {
    console.error(e);
    setThreeStatus("Load Scene error");
    logInfo("Scene load error: " + e.message);
  }
}

/** =========================
 * Buttons
 * ========================= */
document.getElementById("start").addEventListener("click", startWebcam);
document.getElementById("stop").addEventListener("click", stopWebcam);

document.getElementById("clear").addEventListener("click", () => {
  resetOverlayState();
  resetStableKpts();

  standHoldStart = null;
  animReadyGateOpen = false;
  animationPaused = true;

  squatCount = 0;
  squatState = "UP";
  lastDepth = 0;
  lastTorsoLean = 0;
  lastKneeValgus = 0;
  lastFeedback = "대기 중";

  logInfo("(cleared)");
  setThreeStatus("초기화됨 - 차렷 자세를 5초 유지하세요");
});

document.getElementById("loadModel").addEventListener("click", loadModel);
document.getElementById("infer").addEventListener("click", inferOnce);

document.getElementById("live").addEventListener("click", async () => {
  if (!workerReady) {
    logInfo("Load ONNX 먼저.");
    return;
  }
  if (!stream) {
    logInfo("Webcam 먼저.");
    return;
  }

  setLive(!liveOn);

  if (liveOn) {
    if (animReadyGateOpen) {
      setThreeStatus("분석 모드 LIVE ON");
    } else {
      setThreeStatus("LIVE ON - 차렷 자세를 5초 유지하세요");
    }
  } else {
    if (animReadyGateOpen) {
      setThreeStatus("애니메이션 재생 중 (분석 LIVE OFF)");
    } else {
      setThreeStatus("LIVE OFF");
    }
  }
});

document.getElementById("loadScene").addEventListener("click", loadScene);

document.getElementById("toggleAnim").addEventListener("click", () => {
  if (!animReadyGateOpen) {
    setThreeStatus("먼저 차렷 자세를 5초 유지하세요");
    return;
  }

  animationPaused = !animationPaused;
  setThreeStatus(animationPaused ? "Animation paused" : "Animation playing");
});

document.getElementById("resetAnim").addEventListener("click", () => {
  if (mixer) mixer.setTime(0);
  animationPaused = true;
  animReadyGateOpen = false;
  standHoldStart = null;

  resetStableKpts();
  resetOverlayState();

  squatCount = 0;
  squatState = "UP";
  lastDepth = 0;
  lastTorsoLean = 0;
  lastKneeValgus = 0;
  lastFeedback = "대기 중";

  setLive(false);

  setThreeStatus("리셋 완료 - 차렷 자세를 5초 유지하세요");
  logInfo("애니메이션 리셋됨\n차렷 자세 5초 유지 후 다시 시작");
});

/** =========================
 * 첫 안내
 * ========================= */
setThreeStatus(
  "Ready (Start Webcam → Load ONNX → Live ON → Load Scene → 차렷 5초)"
);

animateOverlay();