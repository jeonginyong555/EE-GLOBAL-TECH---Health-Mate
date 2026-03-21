/* eslint-disable no-console */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/** =========================
 * UI
 * ========================= */
const INITIAL_VIEW_W = 420;
const INITIAL_VIEW_H = 560;
const VIDEO_DRAW_INTERVAL = 33; // ~30fps

const INPUT_VIDEO_URL = "/squat.mp4";
const INPUT_POSE_JSON_URL = "/pose_squat.json";

const app = document.querySelector("#app");

app.innerHTML = `
  <div style="font-family:sans-serif; padding:12px;">
    <h2 style="margin:0 0 8px;">Health-Mate Preview (JSON Progress Control + Gym)</h2>

    <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start;">
      <!-- LEFT -->
      <div style="min-width:420px;">
        <div style="margin:10px 0 6px; font-weight:600;">Canvas (Video + Stickman)</div>
        <div id="canvasStage" style="position:relative; width:${INITIAL_VIEW_W}px; height:${INITIAL_VIEW_H}px; border:1px solid #ddd; background:#111; overflow:hidden;">
          <canvas id="canvas" width="${INITIAL_VIEW_W}" height="${INITIAL_VIEW_H}" style="position:absolute; left:0; top:0;"></canvas>
          <canvas id="overlay" width="${INITIAL_VIEW_W}" height="${INITIAL_VIEW_H}" style="position:absolute; left:0; top:0; pointer-events:none;"></canvas>
        </div>

        <div style="margin-top:6px; font-size:12px; color:#666;">
          <span style="display:inline-block;width:10px;height:10px;background:#0f0;margin-right:6px;"></span>pose lines
          &nbsp;&nbsp;
          <span style="display:inline-block;width:10px;height:10px;background:#f00;margin-right:6px;"></span>pose joints
        </div>

        <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
          <button id="start">Start Video</button>
          <button id="stop">Stop Video</button>
          <button id="loadPose">Load Pose JSON</button>
          <button id="sync">Sync 1 Frame</button>
          <button id="live">Live ON/OFF</button>
          <button id="clear">Clear</button>
        </div>

        <div style="margin-top:8px; font-size:12px; color:#666; line-height:1.45;">
          Video: <code>${INPUT_VIDEO_URL}</code><br/>
          Pose JSON: <code>${INPUT_POSE_JSON_URL}</code><br/>
          Left: <code>video + stickman from json</code><br/>
          Right: <code>gym + animated avatar progress control</code>
        </div>
      </div>

      <!-- RIGHT -->
      <div style="min-width:640px; flex:1;">
        <div style="margin:8px 0 6px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <button id="loadScene">Load Scene</button>
          <button id="toggleAnim">Anim ON/OFF</button>
          <button id="resetAnim">Anim Reset</button>
          <button id="testPose">Test 50%</button>
          <label style="display:flex; gap:6px; align-items:center;">
            <input id="autoRotate" type="checkbox" />
            Auto Rotate Camera
          </label>
          <span id="threeStatus" style="color:#555;">Ready</span>
        </div>

        <div id="threeWrap" style="width:920px; max-width:100%; height:520px; border:1px solid #ddd;"></div>

        <div style="margin-top:10px; font-weight:600;">Info</div>
        <pre id="info" style="background:#0f0f10; color:#9ef; padding:10px; border-radius:8px; max-width:920px; overflow:auto; height:260px;">(no data)</pre>

        <div style="margin-top:8px; color:#888; font-size:12px; line-height:1.45;">
          Gym GLB: <code>/models/Untitled_gym.glb</code><br/>
          Avatar GLB: <code>/models/Untitled_squat.glb</code><br/>
          Control: <code>pose_squat.json depth → animation progress</code><br/>
          Note: <code>action.time scrub mode + clip range limit</code>
        </div>
      </div>
    </div>
  </div>
`;

/** hidden input video */
const video = document.createElement("video");
video.autoplay = false;
video.muted = true;
video.loop = true;
video.playsInline = true;
video.preload = "auto";
video.style.display = "none";
document.body.appendChild(video);

/** DOM refs */
const canvasStage = document.getElementById("canvasStage");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const infoEl = document.getElementById("info");
const threeStatusEl = document.getElementById("threeStatus");
const autoRotateEl = document.getElementById("autoRotate");

/** =========================
 * 로그 유틸
 * ========================= */
let infoTick = 0;
function logInfo(msg) {
  infoEl.textContent = String(msg);
}
function logInfoThrottled(msg, every = 2) {
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
let rafId = null;
let videoTimer = null;
let liveOn = false;
let liveTimer = null;
let videoReady = false;
let poseReady = false;

let poseData = null;
let poseFPS = 30;

let lastPoseKpts = null;
let displayKpts = null;
let overlayAnimId = null;

/** =========================
 * 진행도 제어 상태
 * ========================= */
let animationPaused = false;
let squatDepthRaw = 0;
let squatProgressRaw = 0;
let squatProgressSmooth = 0;
let squatProgressRender = 0;

const SYNC_INTERVAL_MS = 33;
const DEPTH_UP = 0.18;
const DEPTH_DOWN = 0.52;
const PROGRESS_EMA = 0.35;
const PROGRESS_SNAP_LOW = 0.02;
const PROGRESS_SNAP_HIGH = 0.95;

// 한 클립 안에 여러 스쿼트가 들어있는 경우 앞 일부만 사용
const CLIP_START_NORM = 0.0;
const CLIP_END_NORM = 0.33;

/** =========================
 * 스쿼트 카운트/피드백 상태
 * ========================= */
let squatCount = 0;
let squatState = "UP";
let lastDepth = 0;
let lastKneeValgus = 1.0;
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
 * Three.js Scene
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
let squatClip = null;
let squatAction = null;

let lastMixerTime = -999;

let lastRenderTime = 0;
const RENDER_FPS = 20;
const RENDER_INTERVAL = 1000 / RENDER_FPS;

const AVATAR_POS = new THREE.Vector3(0, 0, 0.8);
const LOOK_TARGET_OFFSET = new THREE.Vector3(0, 1.05, 0);
const FRONT_CAM_OFFSET = new THREE.Vector3(0, 0.2, 2.5);

/** =========================
 * 유틸
 * ========================= */
function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}
function dist2(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.hypot(dx, dy);
}
function angleDeg2(a, b, c) {
  const abx = a[0] - b[0];
  const aby = a[1] - b[1];
  const cbx = c[0] - b[0];
  const cby = c[1] - b[1];

  const dot = abx * cbx + aby * cby;
  const m1 = Math.hypot(abx, aby);
  const m2 = Math.hypot(cbx, cby);
  if (m1 < 1e-6 || m2 < 1e-6) return 180;

  const cos = Math.max(-1, Math.min(1, dot / (m1 * m2)));
  return (Math.acos(cos) * 180) / Math.PI;
}
function kpOk(p, th = 0.05) {
  return !!p && (p[2] ?? 1) >= th;
}
function mapDepthToProgress(depth) {
  const p = (depth - DEPTH_UP) / Math.max(1e-6, DEPTH_DOWN - DEPTH_UP);
  return clamp01(p);
}
function postProcessProgress(p) {
  let out = p;
  if (out < PROGRESS_SNAP_LOW) out = 0;
  if (out > PROGRESS_SNAP_HIGH) out = 1;
  return out;
}
function getClipRange() {
  if (!squatClip) {
    return { clipStart: 0, clipEnd: 0, usableDuration: 0 };
  }
  const clipDuration = Math.max(0.0001, squatClip.duration);
  const clipStart = clipDuration * CLIP_START_NORM;
  const clipEnd = clipDuration * CLIP_END_NORM;
  const usableDuration = Math.max(0.0001, clipEnd - clipStart);
  return { clipStart, clipEnd, usableDuration };
}
function resetAvatarPoseToStart() {
  if (squatAction && mixer && squatClip) {
    const { clipStart } = getClipRange();
    squatAction.time = clipStart;
    mixer.update(0);
    avatarScene?.updateMatrixWorld(true);
    lastMixerTime = clipStart;
  }
}

/** =========================
 * JSON Pose
 * ========================= */
async function loadPoseJson() {
  try {
    const res = await fetch(INPUT_POSE_JSON_URL);
    if (!res.ok) {
      throw new Error(`pose json load failed: ${res.status}`);
    }

    poseData = await res.json();
    poseFPS = poseData.fps || 30;
    poseReady = Array.isArray(poseData.frames);

    setThreeStatus("Pose JSON loaded");
    logInfo(`pose_squat.json loaded\nfps=${poseFPS}\nframes=${poseData.frames?.length ?? 0}`);
  } catch (e) {
    console.error(e);
    poseReady = false;
    setThreeStatus("Pose JSON load error");
    logInfo("Pose JSON load error: " + e.message);
  }
}

function getPoseFrameAtTimeSec(timeSec) {
  if (!poseReady || !poseData?.frames?.length) return null;

  const frames = poseData.frames;
  const idx = Math.min(
    frames.length - 1,
    Math.max(0, Math.floor(timeSec * poseFPS))
  );

  return frames[idx] || null;
}

function extractKeypointsFromFrame(frame) {
  if (!frame || !frame.valid || !frame.keypoints) return null;
  return frame.keypoints;
}

/** =========================
 * 2D Drawing
 * ========================= */
function clearOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawThresholdByIndex(idx) {
  if ([13, 14, 15, 16].includes(idx)) return 0.05;
  if ([11, 12].includes(idx)) return 0.05;
  return 0.05;
}

function drawSkeleton2D(kpts, lineColor, dotColor, lineWidth = 1.5, dotRadius = 3) {
  octx.lineWidth = lineWidth;
  octx.strokeStyle = lineColor;
  octx.fillStyle = dotColor;

  const w = canvas.width;
  const h = canvas.height;

  for (const [a, b] of COCO_EDGES) {
    const pa = kpts[a];
    const pb = kpts[b];
    if (!pa || !pb) continue;
    if ((pa[2] ?? 1) < drawThresholdByIndex(a) || (pb[2] ?? 1) < drawThresholdByIndex(b)) continue;

    octx.beginPath();
    octx.moveTo(pa[0] * w, pa[1] * h);
    octx.lineTo(pb[0] * w, pb[1] * h);
    octx.stroke();
  }

  for (let i = 0; i < kpts.length; i++) {
    const p = kpts[i];
    if (!p || (p[2] ?? 1) < drawThresholdByIndex(i)) continue;

    octx.beginPath();
    octx.arc(p[0] * w, p[1] * h, dotRadius, 0, Math.PI * 2);
    octx.fill();
  }
}

/** =========================
 * Overlay smoothing
 * ========================= */
function smoothDisplayKpts(targetKpts) {
  if (!targetKpts) return null;

  if (!displayKpts) {
    displayKpts = targetKpts.map((p) => [...p]);
    return displayKpts;
  }

  for (let i = 0; i < targetKpts.length; i++) {
    const src = displayKpts[i];
    const dst = targetKpts[i];
    if (!src || !dst) continue;

    const alpha = [13, 14, 15, 16].includes(i) ? 0.75 : 0.9;

    src[0] = lerp(src[0], dst[0], alpha);
    src[1] = lerp(src[1], dst[1], alpha);
    src[2] = lerp(src[2] ?? 1, dst[2] ?? 1, alpha);
  }

  return displayKpts;
}

function animateOverlay() {
  overlayAnimId = requestAnimationFrame(animateOverlay);

  if (!lastPoseKpts) {
    clearOverlay();
    return;
  }

  const smoothed = smoothDisplayKpts(lastPoseKpts);
  if (!smoothed) return;

  clearOverlay();
  drawSkeleton2D(smoothed, "lime", "red", 1.5, 3);
}

function resetOverlayState() {
  lastPoseKpts = null;
  displayKpts = null;
  clearOverlay();
}

/** =========================
 * 스쿼트 분석 (정면/측면 혼합 대응)
 * ========================= */
function computeSquatMetrics(k) {
  if (!k || k.length < 17) return null;

  const ls = k[5];
  const rs = k[6];
  const lh = k[11];
  const rh = k[12];
  const lk = k[13];
  const rk = k[14];
  const la = k[15];
  const ra = k[16];

  const leftOk = kpOk(lh) && kpOk(lk) && kpOk(la);
  const rightOk = kpOk(rh) && kpOk(rk) && kpOk(ra);

  if (!leftOk && !rightOk) return null;

  let sideHip, sideKnee, sideAnkle;
  if (leftOk && rightOk) {
    const leftScore = (lh[2] ?? 1) + (lk[2] ?? 1) + (la[2] ?? 1);
    const rightScore = (rh[2] ?? 1) + (rk[2] ?? 1) + (ra[2] ?? 1);
    if (leftScore >= rightScore) {
      sideHip = lh;
      sideKnee = lk;
      sideAnkle = la;
    } else {
      sideHip = rh;
      sideKnee = rk;
      sideAnkle = ra;
    }
  } else if (leftOk) {
    sideHip = lh;
    sideKnee = lk;
    sideAnkle = la;
  } else {
    sideHip = rh;
    sideKnee = rk;
    sideAnkle = ra;
  }

  const sideThighLen = dist2(sideHip, sideKnee);
  const sideShinLen = dist2(sideKnee, sideAnkle);
  const legLen = Math.max(1e-6, sideThighLen + sideShinLen);

  const sideKneeAngle = angleDeg2(sideHip, sideKnee, sideAnkle);
  const sideKneeBend = clamp01((175 - sideKneeAngle) / 85);

  const sideVerticalDepth = clamp01(
    (sideHip[1] - sideKnee[1] + sideThighLen * 0.15) / Math.max(1e-6, sideThighLen * 0.9)
  );

  let frontDepth = sideKneeBend;
  if (leftOk && rightOk) {
    const kneeCenter = [(lk[0] + rk[0]) * 0.5, (lk[1] + rk[1]) * 0.5];
    const hipCenter = [(lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5];
    const avgThigh = (dist2(lh, lk) + dist2(rh, rk)) * 0.5;
    const hipToKneeY = kneeCenter[1] - hipCenter[1];
    frontDepth = clamp01(1 - hipToKneeY / Math.max(1e-6, avgThigh));
  }

  const frontness =
    leftOk && rightOk
      ? clamp01((Math.min(lk[2] ?? 1, rk[2] ?? 1) - 0.3) / 0.5)
      : 0;

  const sideDepth = clamp01(sideVerticalDepth * 0.45 + sideKneeBend * 0.55);
  const depth = clamp01(frontDepth * frontness + sideDepth * (1 - frontness));

  let torsoLean = 0;
  if (kpOk(ls) && kpOk(rs)) {
    const shoulderCenter = [(ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5];
    torsoLean = Math.abs(shoulderCenter[0] - sideHip[0]) / Math.max(1e-6, legLen * 0.45);
  }

  return {
    depth,
    torsoLean,
    kneeValgus: 1.0,
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

  if (squatState === "UP" && m.depth >= 0.58) {
    squatState = "DOWN";
  } else if (squatState === "DOWN" && m.depth <= 0.22) {
    squatState = "UP";
    squatCount += 1;
  }

  const feedbacks = [];
  feedbacks.push(m.depth < 0.50 ? "깊이 부족" : "깊이 양호");
  feedbacks.push(m.torsoLean > 0.85 ? "상체 숙임 큼" : "상체 각도 양호");
  lastFeedback = feedbacks.join(" / ");
}

/** =========================
 * Video
 * ========================= */
function resizeVideoStage(videoWidth, videoHeight) {
  const MAX_W = 520;
  const MAX_H = 720;

  const scale = Math.min(MAX_W / videoWidth, MAX_H / videoHeight);
  const drawW = Math.max(1, Math.round(videoWidth * scale));
  const drawH = Math.max(1, Math.round(videoHeight * scale));

  canvasStage.style.width = `${drawW}px`;
  canvasStage.style.height = `${drawH}px`;

  canvas.width = drawW;
  canvas.height = drawH;
  canvas.style.width = `${drawW}px`;
  canvas.style.height = `${drawH}px`;

  overlay.width = drawW;
  overlay.height = drawH;
  overlay.style.width = `${drawW}px`;
  overlay.style.height = `${drawH}px`;
}

function drawVideoFrameToCanvas() {
  if (!videoReady) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

async function startInputVideo() {
  try {
    stopInputVideo(false);

    video.srcObject = null;
    video.src = INPUT_VIDEO_URL;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;

    await new Promise((resolve, reject) => {
      const onLoaded = () => {
        cleanup();
        resolve();
      };
      const onError = () => {
        cleanup();
        reject(new Error("영상 로드 실패"));
      };
      const cleanup = () => {
        video.removeEventListener("loadedmetadata", onLoaded);
        video.removeEventListener("error", onError);
      };

      video.addEventListener("loadedmetadata", onLoaded);
      video.addEventListener("error", onError);
      video.load();
    });

    video.currentTime = 0;
    resizeVideoStage(video.videoWidth || 360, video.videoHeight || 640);
    await video.play();
    videoReady = true;

    if (videoTimer) {
      clearInterval(videoTimer);
      videoTimer = null;
    }

    drawVideoFrameToCanvas();
    videoTimer = setInterval(drawVideoFrameToCanvas, VIDEO_DRAW_INTERVAL);

    setThreeStatus("입력 영상 시작됨");
    logInfo(`입력 영상 시작 완료\n${INPUT_VIDEO_URL}`);
  } catch (err) {
    console.error(err);
    videoReady = false;
    setThreeStatus("Video error");
    logInfo("Video error: " + err.message);
  }
}

function stopInputVideo(resetTime = true) {
  if (videoTimer) {
    clearInterval(videoTimer);
    videoTimer = null;
  }

  setLive(false);

  if (!video.paused) video.pause();

  if (resetTime) {
    try {
      video.currentTime = 0;
    } catch {
      // ignore
    }
  }

  videoReady = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  resetOverlayState();

  squatCount = 0;
  squatState = "UP";
  lastDepth = 0;
  lastTorsoLean = 0;
  lastKneeValgus = 1.0;
  lastFeedback = "대기 중";

  squatDepthRaw = 0;
  squatProgressRaw = 0;
  squatProgressSmooth = 0;
  squatProgressRender = 0;

  resetAvatarPoseToStart();

  setThreeStatus("입력 영상 정지");
  logInfo("입력 영상 정지");
}

/** =========================
 * JSON Sync
 * ========================= */
function syncPoseFromVideoTime() {
  if (!videoReady || !poseReady) return;

  const frame = getPoseFrameAtTimeSec(video.currentTime);
  const kpts = extractKeypointsFromFrame(frame);

  if (!kpts) {
    lastFeedback = "pose frame 없음";
    return;
  }

  lastPoseKpts = kpts;
  updateSquatCounterAndFeedback(lastPoseKpts);

  squatDepthRaw = lastDepth;
  squatProgressRaw = mapDepthToProgress(squatDepthRaw);

  logInfoThrottled(
    `mode=json progress control\n` +
      `videoTime=${video.currentTime.toFixed(2)}s\n` +
      `poseFPS=${poseFPS}\n\n` +
      `squatCount=${squatCount}\n` +
      `depth=${lastDepth.toFixed(3)}\n` +
      `progressRaw=${squatProgressRaw.toFixed(3)}\n` +
      `progressSmooth=${squatProgressSmooth.toFixed(3)}\n` +
      `torsoLean=${lastTorsoLean.toFixed(3)}\n` +
      `feedback=${lastFeedback}\n\n` +
      `clip=${squatClip ? squatClip.name || "(no name)" : "none"}\n` +
      `clipDuration=${squatClip ? squatClip.duration.toFixed(3) : "0"}\n` +
      `range=${CLIP_START_NORM} ~ ${CLIP_END_NORM}\n` +
      `actionTime=${squatAction ? squatAction.time.toFixed(3) : "0"}`,
    2
  );

  setThreeStatus(
    `분석 중 / count ${squatCount} / raw ${squatProgressRaw.toFixed(2)} / apply ${squatProgressSmooth.toFixed(2)} / ${lastFeedback}`
  );
}

function liveLoop() {
  if (!liveOn) return;

  syncPoseFromVideoTime();

  if (!liveOn) return;
  liveTimer = setTimeout(liveLoop, SYNC_INTERVAL_MS);
}

function resetLiveTimer() {
  if (!liveOn) return;

  if (liveTimer) {
    clearTimeout(liveTimer);
    liveTimer = null;
  }

  liveTimer = setTimeout(liveLoop, SYNC_INTERVAL_MS);
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
 * Three.js
 * ========================= */
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
    45,
    THREE_WRAP.clientWidth / THREE_WRAP.clientHeight,
    0.1,
    2000
  );
  camera.position.set(0, 1.3, 2.6);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enabled = false;
  controls.enableDamping = false;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444466, 1.15);
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

function updateProgressControlledAnimation() {
  if (!mixer || !squatClip || !squatAction || animationPaused) return;

  const prev = squatProgressSmooth;
  const next = lerp(prev, squatProgressRaw, PROGRESS_EMA);

  squatProgressSmooth = postProcessProgress(next);
  squatProgressRender = squatProgressSmooth;

  const { clipStart, usableDuration } = getClipRange();
  const targetTime = clipStart + usableDuration * squatProgressRender;

  if (Math.abs(targetTime - lastMixerTime) > 0.001) {
    squatAction.time = targetTime;
    mixer.update(0);
    avatarScene?.updateMatrixWorld(true);
    lastMixerTime = targetTime;
  }
}

function animate(now = 0) {
  rafId = requestAnimationFrame(animate);

  if (autoRotateEl && autoRotateEl.checked && camera && avatarScene) {
    const target = getAvatarLookTarget();
    const t = now * 0.0004;
    const radius = 2.5;
    camera.position.x = target.x + Math.sin(t) * radius;
    camera.position.z = target.z + Math.cos(t) * radius;
    camera.position.y = target.y + 0.2;
    camera.lookAt(target);
  }

  updateProgressControlledAnimation();

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
        squatClip = null;
        squatAction = null;
        lastMixerTime = -999;

        squatCount = 0;
        squatState = "UP";
        lastDepth = 0;
        lastTorsoLean = 0;
        lastKneeValgus = 1.0;
        lastFeedback = "대기 중";

        squatDepthRaw = 0;
        squatProgressRaw = 0;
        squatProgressSmooth = 0;
        squatProgressRender = 0;

        const clips = gltf.animations || [];
        const clipLines = clips.map(
          (a, i) => `[${i}] ${a.name || "(no name)"} / duration=${a.duration.toFixed(3)} / tracks=${a.tracks.length}`
        );

        if (clips.length > 0) {
          mixer = new THREE.AnimationMixer(avatarScene);

          const byName = clips.find((a) =>
            (a.name || "").toLowerCase().includes("squat")
          );
          const byDuration = [...clips].sort((a, b) => b.duration - a.duration)[0];

          squatClip = byName || byDuration;
          squatAction = mixer.clipAction(squatClip);

          squatAction.enabled = true;
          squatAction.setLoop(THREE.LoopOnce, 1);
          squatAction.clampWhenFinished = true;
          squatAction.play();

          // 핵심: 자동 재생 말고 time만 직접 제어
          squatAction.paused = true;

          const { clipStart } = getClipRange();
          squatAction.time = clipStart;

          mixer.update(0);
          avatarScene.updateMatrixWorld(true);

          lastMixerTime = clipStart;
        }

        const target = getAvatarLookTarget();
        camera.position.set(
          target.x + FRONT_CAM_OFFSET.x,
          target.y + FRONT_CAM_OFFSET.y,
          target.z + FRONT_CAM_OFFSET.z
        );
        camera.lookAt(target);

        animationPaused = false;
        setThreeStatus("장면 로드 완료 - 진행도 제어 준비 완료");

        logInfo(
          "장면 로드 완료\n" +
          `animation clips: ${clips.length}\n` +
          (clipLines.length ? clipLines.join("\n") + "\n\n" : "") +
          `selected clip: ${squatClip ? squatClip.name || "(no name)" : "none"}\n` +
          `duration=${squatClip ? squatClip.duration.toFixed(3) : "0"}\n` +
          `range=${CLIP_START_NORM} ~ ${CLIP_END_NORM}`
        );

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
    setThreeStatus("Loading gym + avatar...");

    const loader = new GLTFLoader();
    await loadGym(loader);
    await loadAnimatedAvatar(loader);

    setThreeStatus("장면 로드 완료 - Video + Pose JSON + Live ON");
  } catch (e) {
    console.error(e);
    setThreeStatus("Load Scene error");
    logInfo("Scene load error: " + e.message);
  }
}

/** =========================
 * Buttons
 * ========================= */
document.getElementById("start").addEventListener("click", startInputVideo);
document.getElementById("stop").addEventListener("click", () => stopInputVideo(true));

document.getElementById("loadPose").addEventListener("click", loadPoseJson);

document.getElementById("sync").addEventListener("click", () => {
  if (!poseReady) {
    logInfo("Load Pose JSON 먼저.");
    return;
  }
  if (!videoReady) {
    logInfo("Start Video 먼저.");
    return;
  }
  syncPoseFromVideoTime();
});

document.getElementById("clear").addEventListener("click", () => {
  resetOverlayState();

  if (videoReady) {
    drawVideoFrameToCanvas();
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  squatCount = 0;
  squatState = "UP";
  lastDepth = 0;
  lastTorsoLean = 0;
  lastKneeValgus = 1.0;
  lastFeedback = "대기 중";

  squatDepthRaw = 0;
  squatProgressRaw = 0;
  squatProgressSmooth = 0;
  squatProgressRender = 0;

  resetAvatarPoseToStart();

  logInfo("(cleared)");
  setThreeStatus("초기화됨");
});

document.getElementById("live").addEventListener("click", () => {
  if (!poseReady) {
    logInfo("Load Pose JSON 먼저.");
    return;
  }
  if (!videoReady) {
    logInfo("Start Video 먼저.");
    return;
  }

  setLive(!liveOn);
  setThreeStatus(liveOn ? "JSON 분석 LIVE ON" : "JSON 분석 LIVE OFF");
});

document.getElementById("loadScene").addEventListener("click", loadScene);

document.getElementById("toggleAnim").addEventListener("click", () => {
  if (!mixer || !squatClip) {
    setThreeStatus("먼저 Scene을 로드하세요");
    return;
  }

  animationPaused = !animationPaused;
  setThreeStatus(animationPaused ? "Animation progress paused" : "Animation progress playing");
});

document.getElementById("resetAnim").addEventListener("click", () => {
  animationPaused = false;

  resetOverlayState();

  if (videoReady) {
    drawVideoFrameToCanvas();
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  squatCount = 0;
  squatState = "UP";
  lastDepth = 0;
  lastTorsoLean = 0;
  lastKneeValgus = 1.0;
  lastFeedback = "대기 중";

  squatDepthRaw = 0;
  squatProgressRaw = 0;
  squatProgressSmooth = 0;
  squatProgressRender = 0;

  resetAvatarPoseToStart();

  setLive(false);

  setThreeStatus("애니메이션 진행도 리셋 완료");
  logInfo("애니메이션 진행도 리셋됨");
});

document.getElementById("testPose").addEventListener("click", () => {
  if (!squatAction || !mixer || !squatClip) {
    logInfo("먼저 Scene을 로드하세요");
    return;
  }

  const { clipStart, usableDuration } = getClipRange();
  const t = clipStart + usableDuration * 0.5;

  squatAction.time = t;
  mixer.update(0);
  avatarScene?.updateMatrixWorld(true);
  lastMixerTime = t;

  setThreeStatus(`Test pose applied / t=${t.toFixed(2)}`);
  logInfo(
    `Test pose applied\nclip=${squatClip.name || "(no name)"}\nduration=${squatClip.duration.toFixed(3)}\nrange=${CLIP_START_NORM} ~ ${CLIP_END_NORM}\ntime=${t.toFixed(3)}`
  );
});

/** =========================
 * 영상 draw loop
 * ========================= */
function videoDrawLoop() {
  if (videoReady) {
    drawVideoFrameToCanvas();
  }
}
setInterval(videoDrawLoop, VIDEO_DRAW_INTERVAL);

/** =========================
 * 첫 안내
 * ========================= */
setThreeStatus("Ready (Start Video → Load Pose JSON → Load Scene → Live ON)");
animateOverlay();