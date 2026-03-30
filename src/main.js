/* eslint-disable no-console */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
} from "@mediapipe/tasks-vision";

/* =========================
 * PATH
 * ========================= */
const GUIDE_VIDEO_URL = "/squat.mp4";
const POSE_JSON_URL = "/pose_squat.json";

const MP_VISION_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const POSE_MODEL_URL = "/models/pose_landmarker_lite.task";
const HAND_MODEL_URL = "/models/hand_landmarker.task";

const GYM_URL = "/models/Untitled_gym.glb";
const AVATAR_URL = "/models/Untitled_squat.glb";

/* =========================
 * DOM
 * ========================= */
const dom = {
  body: document.body,

  modeScreen: document.getElementById("mode-screen"),
  appScreen: document.getElementById("app-screen"),
  themeToggle: document.getElementById("themeToggle"),
  backBtn: document.getElementById("backBtn"),
  modeCards: document.querySelectorAll(".mode-card"),

  viewerStage: document.getElementById("viewerStage"),
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  overlay: document.getElementById("overlay"),
  guideVideo: document.getElementById("guideVideo"),
  threeWrap: document.getElementById("threeWrap"),

  modeChip: document.getElementById("modeChip"),
  statusChip: document.getElementById("statusChip"),
  helpChip: document.getElementById("helpChip"),
  gestureChip: document.getElementById("gestureChip"),
  counterChip: document.getElementById("counterChip"),
  feedbackToastWrap: document.getElementById("feedbackToastWrap"),

  manualToggleBtn: document.getElementById("manualToggleBtn"),
  restartBtn: document.getElementById("restartBtn"),
  cameraRotateToggle: document.getElementById("cameraRotateToggle"),
};

const ctx = dom.canvas.getContext("2d", { willReadFrequently: true });
const octx = dom.overlay.getContext("2d", { willReadFrequently: true });

/* =========================
 * CONFIG
 * ========================= */
const DRAW_INTERVAL = 33;
const POSE_INTERVAL = 33;
const HAND_INTERVAL = 80;
const RENDER_FPS = 30;
const RENDER_INTERVAL = 1000 / RENDER_FPS;

const FIST_TOGGLE_HOLD_FRAMES = 8;
const FIST_TOGGLE_COOLDOWN_MS = 1500;
const CAMERA_ROTATE_STEP = 0.03;

const CLIP_START_NORM = 0.0;
const STANDARD_CLIP_END_NORM = 0.42;
const AVATAR_CLIP_END_NORM = 0.33;

const JSON_TIME_OFFSET_SEC = 0.0;

const FEEDBACK_COLORS = {
  line: "#21e17b",
  joint: "#ff3b30",
  warn: "#ffb020",
  bad: "#ff5a67",
  missing: "#39a0ff",
  hand: "#3cf",
};

const MP_POSE_EDGES = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [27, 29],
  [29, 31],
  [28, 30],
  [30, 32],
];

const JSON_POSE_EDGES = [
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const AVATAR_POS = new THREE.Vector3(0, 0, 0.8);
const LOOK_TARGET_OFFSET = new THREE.Vector3(0, 0.9, 0);
const FRONT_CAM_OFFSET = new THREE.Vector3(0, 0.55, 3.4);

/* =========================
 * STATE
 * ========================= */
const state = {
  currentMode: null,
  themeLight: false,
  stopped: true,

  webcamStream: null,
  poseData: null,
  poseFPS: 30,
  jsonDepthRange: {
    min: 0,
    max: 1,
  },

  poseLandmarker: null,
  handLandmarker: null,

  poseBusy: false,
  handBusy: false,

  drawTimer: null,
  poseTimer: null,
  handTimer: null,
  jsonTimer: null,
  jsonVideoFrameHandle: null,

  lastPoseKpts: null,
  smoothPoseKpts: null,
  poseVelocity: null,

  poseLock: {
    active: false,
    stableFrames: 0,
    baseCenter: null,
  },

  lastHandResult: null,
  lastHandedness: null,

  feedbackDisplay: "webcam",
  avatarDisplay: "avatar",
  standardDisplay: "avatarMain",

  fistHoldCount: 0,
  fistCooldown: false,

  cameraRotateEnabled: true,
  handRotateState: "NEUTRAL",
  handCurrentYaw: 0,

  gestureHandLabel: null,
  gestureHandLockFrames: 0,

  squatProgressRaw: 0,
  squatProgressSmooth: 0,
  squatCount: 0,
  squatState: "UP",
  lastRenderedCount: 0,
  counterPopTimer: null,

  shallowHoldFrames: 0,
  shallowHoldActive: false,

  isInReadyPose: true,

  feedbackText: "대기",
  partFeedback: {
    upper: "normal",
    leftLeg: "normal",
    rightLeg: "normal",
    knee: "normal",
  },

  feedbackToastCooldowns: {},
  lastFeedbackSnapshot: "",
};

/* =========================
 * THREE STATE
 * ========================= */
let renderer = null;
let scene = null;
let camera = null;
let controls = null;
let gymRoot = null;
let avatarScene = null;
let mixer = null;
let squatClip = null;
let squatAction = null;
let renderRafId = 0;
let lastRenderTime = 0;

/* =========================
 * UTILS
 * ========================= */
function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function clamp01(v) {
  return clamp(v, 0, 1);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function safePlay(videoEl) {
  return videoEl.play().catch(() => {});
}

function clearTimers() {
  if (state.drawTimer) clearInterval(state.drawTimer);
  if (state.poseTimer) clearInterval(state.poseTimer);
  if (state.handTimer) clearInterval(state.handTimer);
  if (state.jsonTimer) clearInterval(state.jsonTimer);

  state.drawTimer = null;
  state.poseTimer = null;
  state.handTimer = null;
  state.jsonTimer = null;
}

function clearOverlay() {
  octx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
}

function clearCanvas() {
  ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
}

function resizeStageToVideo(videoWidth, videoHeight) {
  const w = Math.max(1, videoWidth);
  const h = Math.max(1, videoHeight);
  dom.canvas.width = w;
  dom.canvas.height = h;
  dom.overlay.width = w;
  dom.overlay.height = h;
}

function setStatus(text) {
  dom.statusChip.textContent = text;
}

function setHelp(text) {
  dom.helpChip.textContent = text;
}

function setFeedbackText(text = "대기") {
  state.feedbackText = text;
  dom.statusChip.textContent = text;
}

function showModeScreen() {
  dom.modeScreen.classList.remove("hidden");
  dom.appScreen.classList.add("hidden");
}

function showAppScreen() {
  dom.modeScreen.classList.add("hidden");
  dom.appScreen.classList.remove("hidden");
}

function resetViewVisibility() {
  dom.video.style.display = "none";
  dom.canvas.style.display = "none";
  dom.overlay.style.display = "none";
  dom.guideVideo.style.display = "none";
  dom.threeWrap.style.display = "none";

  dom.guideVideo.classList.remove("stage-main", "stage-pip");
  dom.threeWrap.classList.remove("stage-main", "stage-pip");

  dom.guideVideo.style.inset = "";
  dom.guideVideo.style.width = "";
  dom.guideVideo.style.height = "";
  dom.threeWrap.style.inset = "";
  dom.threeWrap.style.width = "";
  dom.threeWrap.style.height = "";
}

function setTheme() {
  state.themeLight = !state.themeLight;
  dom.body.classList.toggle("light-mode", state.themeLight);
  dom.themeToggle.textContent = state.themeLight ? "☀️ LIGHT MODE" : "🌙 DARK MODE";

  if (scene) {
    scene.background = new THREE.Color(state.themeLight ? 0xf4f7f6 : 0x0a0a0c);
  }
}

function resizeThreeRenderer() {
  if (!renderer || !camera) return;

  const stageW = dom.viewerStage?.clientWidth || 1280;
  const stageH = dom.viewerStage?.clientHeight || 720;
  const wrapW = dom.threeWrap?.clientWidth || stageW;
  const wrapH = dom.threeWrap?.clientHeight || stageH;

  const w = Math.max(1, wrapW || stageW);
  const h = Math.max(1, wrapH || stageH);

  renderer.setSize(w, h, false);
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.style.display = "block";

  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
}

function resetPartFeedback() {
  state.partFeedback = {
    upper: "normal",
    leftLeg: "normal",
    rightLeg: "normal",
    knee: "normal",
  };
}

function showCounter(show) {
  if (!dom.counterChip) return;
  dom.counterChip.classList.toggle("hidden", !show);
}

function popCounter() {
  if (!dom.counterChip) return;

  dom.counterChip.classList.add("pop");

  if (state.counterPopTimer) {
    clearTimeout(state.counterPopTimer);
  }

  state.counterPopTimer = setTimeout(() => {
    dom.counterChip?.classList.remove("pop");
    state.counterPopTimer = null;
  }, 180);
}

function updateCounterChip(force = false) {
  if (!dom.counterChip) return;

  const nextText = `${state.squatCount}`;
  const prevText = dom.counterChip.textContent;

  if (force || prevText !== nextText) {
    dom.counterChip.textContent = nextText;
  }

  if (force) {
    state.lastRenderedCount = state.squatCount;
    return;
  }

  if (state.squatCount !== state.lastRenderedCount) {
    state.lastRenderedCount = state.squatCount;
    popCounter();
  }
}

function clearFeedbackToasts() {
  if (!dom.feedbackToastWrap) return;
  dom.feedbackToastWrap.innerHTML = "";
}

function showFeedbackToast({
  level = "warn",
  title = "피드백",
  text = "",
  duration = 1800,
}) {
  if (!dom.feedbackToastWrap) return;
  if (state.currentMode !== "feedback") return;

  const toast = document.createElement("div");
  toast.className = `feedback-toast ${level}`;

  const icon = document.createElement("div");
  icon.className = "feedback-toast-icon";
  icon.textContent =
    level === "bad" ? "!" :
    level === "warn" ? "!" :
    level === "missing" ? "?" : "✓";

  const body = document.createElement("div");
  body.className = "feedback-toast-body";

  const titleEl = document.createElement("div");
  titleEl.className = "feedback-toast-title";
  titleEl.textContent = title;

  const textEl = document.createElement("div");
  textEl.className = "feedback-toast-text";
  textEl.textContent = text;

  body.appendChild(titleEl);
  body.appendChild(textEl);

  toast.appendChild(icon);
  toast.appendChild(body);

  dom.feedbackToastWrap.prepend(toast);

  while (dom.feedbackToastWrap.children.length > 3) {
    dom.feedbackToastWrap.removeChild(dom.feedbackToastWrap.lastChild);
  }

  setTimeout(() => {
    toast.classList.add("hide");
    setTimeout(() => toast.remove(), 200);
  }, duration);
}

function tryToastFeedback(key, config, cooldownMs = 1400) {
  const now = performance.now();
  const last = state.feedbackToastCooldowns[key] || 0;

  if (now - last < cooldownMs) return;

  state.feedbackToastCooldowns[key] = now;
  showFeedbackToast(config);
}

function emitDetailedFeedbackToasts() {
  if (state.currentMode !== "feedback") return;
  if (state.feedbackDisplay !== "webcam") return;

  const pf = state.partFeedback;

  if (
    pf.upper === "missing" &&
    pf.leftLeg === "missing" &&
    pf.rightLeg === "missing" &&
    pf.knee === "missing"
  ) {
    tryToastFeedback("missing-all", {
      level: "missing",
      title: "관절 인식 부족",
      text: "몸이 화면 안에 충분히 들어오도록 서 주세요.",
      duration: 1500,
    }, 1800);
    return;
  }

  if (state.isInReadyPose) return;

  if (state.shallowHoldActive || state.feedbackText === "깊이 부족") {
    tryToastFeedback("depth-bad", {
      level: "bad",
      title: "깊이 부족",
      text: "엉덩이를 조금 더 아래로 내려서 끝까지 앉아 주세요.",
      duration: 1700,
    }, 1600);
  } else if (state.feedbackText === "조금 더 앉아주세요") {
    tryToastFeedback("depth-warn", {
      level: "warn",
      title: "조금만 더",
      text: "현재 자세는 거의 맞습니다. 조금만 더 깊게 앉아 주세요.",
      duration: 1500,
    }, 1500);
  }

  if (pf.upper === "bad") {
    tryToastFeedback("upper-bad", {
      level: "bad",
      title: "상체 기울어짐",
      text: "허리를 세우고 가슴을 너무 숙이지 않게 유지해 주세요.",
      duration: 1700,
    }, 1500);
  } else if (pf.upper === "warn") {
    tryToastFeedback("upper-warn", {
      level: "warn",
      title: "상체 주의",
      text: "상체가 살짝 앞으로 기울었습니다. 조금만 더 세워 주세요.",
      duration: 1500,
    }, 1400);
  }

  if (pf.knee === "bad") {
    tryToastFeedback("knee-bad", {
      level: "bad",
      title: "무릎 정렬 불안정",
      text: "무릎 방향을 좌우로 흔들지 말고 발끝 방향과 맞춰 주세요.",
      duration: 1700,
    }, 1500);
  } else if (pf.knee === "warn") {
    tryToastFeedback("knee-warn", {
      level: "warn",
      title: "무릎 주의",
      text: "무릎이 좌우로 살짝 흔들립니다. 정면으로 유지해 주세요.",
      duration: 1500,
    }, 1400);
  }

  const badLegs = pf.leftLeg === "bad" || pf.rightLeg === "bad";
  const warnLegs = pf.leftLeg === "warn" || pf.rightLeg === "warn";

  if (badLegs) {
    tryToastFeedback("legs-bad", {
      level: "bad",
      title: "하체 균형 불안정",
      text: "한쪽 다리에 치우치지 말고 좌우 밸런스를 맞춰 내려가 주세요.",
      duration: 1700,
    }, 1500);
  } else if (warnLegs) {
    tryToastFeedback("legs-warn", {
      level: "warn",
      title: "하체 균형 주의",
      text: "좌우 높이나 중심이 약간 다릅니다. 균형을 맞춰 주세요.",
      duration: 1500,
    }, 1400);
  }

  const snapshot = JSON.stringify({
    upper: pf.upper,
    leftLeg: pf.leftLeg,
    rightLeg: pf.rightLeg,
    knee: pf.knee,
    shallow: state.shallowHoldActive,
    text: state.feedbackText,
  });

  const allNormal =
    pf.upper === "normal" &&
    pf.leftLeg === "normal" &&
    pf.rightLeg === "normal" &&
    pf.knee === "normal" &&
    !state.shallowHoldActive &&
    !state.isInReadyPose;

  if (allNormal && snapshot !== state.lastFeedbackSnapshot) {
    tryToastFeedback("good", {
      level: "good",
      title: "좋아요",
      text: "현재 자세가 안정적입니다. 그대로 유지해 주세요.",
      duration: 1100,
    }, 2200);
  }

  state.lastFeedbackSnapshot = snapshot;
}

function pointState(p, th = 0.05) {
  if (!p) return "missing";
  if ((p[2] ?? 0) < th) return "missing";
  return "ok";
}

function kpOk(p, th = 0.05) {
  return !!p && (p[2] ?? 0) >= th;
}

function edgeHasMissing(pa, pb, th = 0.05) {
  return pointState(pa, th) === "missing" || pointState(pb, th) === "missing";
}

function getPartColor(partState, isJoint = false) {
  if (partState === "missing") return FEEDBACK_COLORS.missing;
  if (partState === "warn") return FEEDBACK_COLORS.warn;
  if (partState === "bad") return FEEDBACK_COLORS.bad;
  return isJoint ? FEEDBACK_COLORS.joint : FEEDBACK_COLORS.line;
}

function normalizeJsonKpts(frame) {
  if (!frame?.valid || !frame?.keypoints) return null;

  return frame.keypoints.map((p) => {
    if (!p) return null;
    return [
      clamp01(p[0]),
      clamp01(p[1]),
      Number.isFinite(p[2]) ? p[2] : 1,
    ];
  });
}

function getPoseCenter(kpts) {
  if (!kpts) return null;

  const lh = kpts[23];
  const rh = kpts[24];
  const ls = kpts[11];
  const rs = kpts[12];

  if (kpOk(lh) && kpOk(rh)) {
    return [
      (lh[0] + rh[0]) * 0.5,
      (lh[1] + rh[1]) * 0.5,
    ];
  }

  if (kpOk(ls) && kpOk(rs)) {
    return [
      (ls[0] + rs[0]) * 0.5,
      (ls[1] + rs[1]) * 0.5,
    ];
  }

  return null;
}

function getPoseScale(kpts) {
  if (!kpts) return 0.2;

  const ls = kpts[11];
  const rs = kpts[12];
  const lh = kpts[23];
  const rh = kpts[24];

  const shoulderCenter = avgPoint(ls, rs);
  const hipCenter = avgPoint(lh, rh);

  if (shoulderCenter && hipCenter) {
    return Math.max(
      0.12,
      Math.hypot(
        shoulderCenter[0] - hipCenter[0],
        shoulderCenter[1] - hipCenter[1]
      )
    );
  }

  return 0.2;
}

function poseHasEnoughCore(kpts) {
  if (!kpts) return false;
  const core = [11, 12, 23, 24, 25, 26];
  return core.every((i) => kpOk(kpts[i], 0.08));
}

/* 기마자세/얕은 홀드 감지 */
function updateShallowHoldState(kpts) {
  state.shallowHoldActive = false;

  if (!kpts) {
    state.shallowHoldFrames = 0;
    return;
  }

  const lh = kpts[23];
  const rh = kpts[24];
  const lk = kpts[25];
  const rk = kpts[26];
  const la = kpts[27];
  const ra = kpts[28];

  if (![lh, rh, lk, rk, la, ra].every((p) => kpOk(p, 0.08))) {
    state.shallowHoldFrames = 0;
    return;
  }

  const hipY = (lh[1] + rh[1]) * 0.5;
  const kneeY = (lk[1] + rk[1]) * 0.5;

  const depth = clamp01(1 - ((kneeY - hipY - 0.04) / 0.24));

  const leftThigh = angle2D(lh, lk);
  const rightThigh = angle2D(rh, rk);
  const leftShin = angle2D(lk, la);
  const rightShin = angle2D(rk, ra);

  const leftKneeAngle = angleDiffDeg(leftThigh, leftShin);
  const rightKneeAngle = angleDiffDeg(rightThigh, rightShin);
  const kneeAngleAvg = ((leftKneeAngle ?? 180) + (rightKneeAngle ?? 180)) * 0.5;

  const prevDepth = state.squatProgressSmooth ?? depth;
  const depthDelta = Math.abs(depth - prevDepth);

  const kneesBent = kneeAngleAvg < 155;
  const shallowDepth = depth < 0.52;
  const nearlyStill = depthDelta < 0.02;

  if (kneesBent && shallowDepth && nearlyStill) {
    state.shallowHoldFrames += 1;
  } else {
    state.shallowHoldFrames = 0;
  }

  if (state.shallowHoldFrames >= 8) {
    state.shallowHoldActive = true;
  }
}

function updateReadyPoseState(kpts) {
  if (!kpts) {
    state.isInReadyPose = true;
    return;
  }

  const lh = kpts[23];
  const rh = kpts[24];
  const lk = kpts[25];
  const rk = kpts[26];

  if (![lh, rh, lk, rk].every((p) => kpOk(p, 0.08))) {
    state.isInReadyPose = true;
    return;
  }

  const hipY = (lh[1] + rh[1]) * 0.5;
  const kneeY = (lk[1] + rk[1]) * 0.5;

  const depth = clamp01(1 - ((kneeY - hipY - 0.04) / 0.24));

  if (state.isInReadyPose) {
    if (depth > 0.45) {
      state.isInReadyPose = false;
    }
    return;
  }

  if (depth < 0.30) {
    state.isInReadyPose = true;
  }
}

/* =========================
 * VIDEO / WEBCAM / JSON
 * ========================= */
async function startWebcam() {
  stopWebcam();

  state.webcamStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
    audio: false,
  });

  dom.video.src = "";
  dom.video.srcObject = state.webcamStream;
  dom.video.muted = true;
  dom.video.playsInline = true;
  dom.video.loop = false;
  dom.video.autoplay = true;

  await new Promise((resolve) => {
    dom.video.onloadedmetadata = () => resolve();
  });

  resizeStageToVideo(dom.video.videoWidth || 640, dom.video.videoHeight || 480);
  await safePlay(dom.video);
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach((t) => t.stop());
    state.webcamStream = null;
  }
  dom.video.srcObject = null;
}

async function startGuideVideo() {
  dom.guideVideo.src = GUIDE_VIDEO_URL;
  dom.guideVideo.muted = true;
  dom.guideVideo.loop = true;
  dom.guideVideo.playsInline = true;
  dom.guideVideo.preload = "auto";

  await new Promise((resolve, reject) => {
    const ok = () => {
      cleanup();
      resolve();
    };
    const fail = () => {
      cleanup();
      reject(new Error("guide video load failed"));
    };
    const cleanup = () => {
      dom.guideVideo.removeEventListener("loadedmetadata", ok);
      dom.guideVideo.removeEventListener("error", fail);
    };
    dom.guideVideo.addEventListener("loadedmetadata", ok);
    dom.guideVideo.addEventListener("error", fail);
    dom.guideVideo.load();
  });

  dom.guideVideo.currentTime = 0;

  await safePlay(dom.guideVideo);
}

function stopGuideVideo() {
  try {
    dom.guideVideo.pause();
  } catch {}
  dom.guideVideo.removeAttribute("src");
  dom.guideVideo.load();
}

async function loadPoseJson() {
  if (state.poseData) return;
  const res = await fetch(POSE_JSON_URL);
  if (!res.ok) throw new Error(`pose json load failed: ${res.status}`);
  state.poseData = await res.json();
  state.poseFPS = state.poseData.fps || 30;

  const values = [];
  for (const frame of state.poseData.frames || []) {
    if (!frame?.valid || !frame?.keypoints) continue;

    const k = frame.keypoints;
    const lh = k[11];
    const rh = k[12];
    const lk = k[13];
    const rk = k[14];
    if (!lh || !rh || !lk || !rk) continue;

    const hipY = (lh[1] + rh[1]) * 0.5;
    const kneeY = (lk[1] + rk[1]) * 0.5;
    const diff = kneeY - hipY;
    if (Number.isFinite(diff)) values.push(diff);
  }

  if (values.length > 1) {
    state.jsonDepthRange.min = Math.min(...values);
    state.jsonDepthRange.max = Math.max(...values);
  } else {
    state.jsonDepthRange.min = 0;
    state.jsonDepthRange.max = 1;
  }
}

function getPoseFrameAtTimeSec(timeSec) {
  if (!state.poseData?.frames?.length) return null;
  const frames = state.poseData.frames;
  const idx = Math.min(frames.length - 1, Math.max(0, Math.floor(timeSec * state.poseFPS)));
  return frames[idx] || null;
}

function computeDepthFromFrame(frame) {
  if (!frame?.valid || !frame?.keypoints) return 0;

  const k = frame.keypoints;
  const lh = k[11];
  const rh = k[12];
  const lk = k[13];
  const rk = k[14];
  if (!lh || !rh || !lk || !rk) return 0;

  const hipY = (lh[1] + rh[1]) * 0.5;
  const kneeY = (lk[1] + rk[1]) * 0.5;
  const diff = kneeY - hipY;

  const minV = state.jsonDepthRange.min;
  const maxV = state.jsonDepthRange.max;
  const span = Math.max(0.0001, maxV - minV);

  const norm = 1 - (diff - minV) / span;
  return clamp01(norm);
}

/* =========================
 * MEDIAPIPE TASKS
 * ========================= */
async function ensureVisionResolver() {
  if (ensureVisionResolver._vision) return ensureVisionResolver._vision;
  ensureVisionResolver._vision = await FilesetResolver.forVisionTasks(MP_VISION_WASM);
  return ensureVisionResolver._vision;
}

async function loadPoseModel() {
  if (state.poseLandmarker) return;

  const vision = await ensureVisionResolver();

  state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: POSE_MODEL_URL },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.45,
    minPosePresenceConfidence: 0.45,
    minTrackingConfidence: 0.65,
    outputSegmentationMasks: false,
  });
}

async function loadHandsModel() {
  if (state.handLandmarker) return;

  const vision = await ensureVisionResolver();

  state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL_URL },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.60,
    minHandPresenceConfidence: 0.60,
    minTrackingConfidence: 0.60,
  });
}

function mpPoseToKpts(landmarks) {
  if (!landmarks || landmarks.length < 33) return null;
  return landmarks.map((p) => [
    clamp01(p.x),
    clamp01(p.y),
    p.visibility ?? 1,
  ]);
}

async function inferPoseFrame() {
  if (
    state.stopped ||
    !state.poseLandmarker ||
    !dom.video.videoWidth ||
    dom.video.readyState < 3 ||
    state.poseBusy
  ) return;

  state.poseBusy = true;

  try {
    const result = state.poseLandmarker.detectForVideo(dom.video, performance.now());
    const lm = result?.landmarks?.[0] || null;

    if (!lm) {
      state.lastPoseKpts = null;

      if (state.smoothPoseKpts) {
        for (const p of state.smoothPoseKpts) {
          if (!p) continue;
          p[2] = lerp(p[2] ?? 1, 0.0, 0.35);
        }
      }

      state.poseLock.active = false;
      state.poseLock.stableFrames = 0;
      state.poseLock.baseCenter = null;
      return;
    }

    const kpts = mpPoseToKpts(lm);
    state.lastPoseKpts = kpts;
    updateSmoothPose(kpts);
  } catch (err) {
    console.error("POSE ERROR:", err);
  } finally {
    state.poseBusy = false;
  }
}

/* =========================
 * POSE SMOOTH
 * ========================= */
function cloneKpts(kpts) {
  return kpts ? kpts.map((p) => [...p]) : null;
}

function updateSmoothPose(newPose) {
  if (!newPose || !poseHasEnoughCore(newPose)) return;

  const newCenter = getPoseCenter(newPose);
  const newScale = getPoseScale(newPose);

  if (!state.smoothPoseKpts) {
    state.smoothPoseKpts = cloneKpts(newPose);
    state.poseVelocity = newPose.map(() => [0, 0]);

    state.poseLock.active = false;
    state.poseLock.stableFrames = 1;
    state.poseLock.baseCenter = newCenter ? [...newCenter] : null;
    return;
  }

  if (!state.poseVelocity) {
    state.poseVelocity = newPose.map(() => [0, 0]);
  }

  const prevCenter = getPoseCenter(state.smoothPoseKpts);
  const centerJump =
    newCenter && prevCenter
      ? Math.hypot(newCenter[0] - prevCenter[0], newCenter[1] - prevCenter[1])
      : 0;

  if (!state.poseLock.active) {
    if (centerJump < newScale * 0.22) {
      state.poseLock.stableFrames += 1;
    } else {
      state.poseLock.stableFrames = 0;
    }

    if (state.poseLock.stableFrames >= 6) {
      state.poseLock.active = true;
      state.poseLock.baseCenter = newCenter ? [...newCenter] : null;
    }
  }

  const hardJump = centerJump > newScale * 0.55;

  for (let i = 0; i < newPose.length; i++) {
    const cur = newPose[i];
    const prev = state.smoothPoseKpts[i];
    if (!prev) continue;

    if (!kpOk(cur, 0.08)) {
      prev[2] = lerp(prev[2] ?? 1, 0.35, 0.18);
      continue;
    }

    const dx = cur[0] - prev[0];
    const dy = cur[1] - prev[1];
    const move = Math.hypot(dx, dy);

    if (move > newScale * 0.95) {
      prev[2] = lerp(prev[2] ?? 1, cur[2] ?? 1, 0.12);
      continue;
    }

    const isLower = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32].includes(i);
    const isCore = [11, 12, 23, 24].includes(i);

    let alpha = isLower ? 0.62 : 0.52;

    if (state.poseLock.active) {
      alpha = isLower ? 0.50 : 0.42;
    }

    if (hardJump) {
      alpha *= 0.72;
    }

    if (move > newScale * 0.65) {
      alpha *= 0.55;
    }

    if (isCore) {
      alpha *= 0.92;
    }

    alpha = clamp(alpha, 0.18, 0.72);

    state.poseVelocity[i][0] = dx;
    state.poseVelocity[i][1] = dy;

    prev[0] = lerp(prev[0], cur[0], alpha);
    prev[1] = lerp(prev[1], cur[1], alpha);
    prev[2] = lerp(prev[2] ?? 1, cur[2] ?? 1, 0.28);
  }
}

function radToDeg(rad) {
  return (rad * 180) / Math.PI;
}

function angle2D(a, b) {
  if (!a || !b) return null;
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  return radToDeg(Math.atan2(dy, dx));
}

function avgPoint(a, b) {
  if (!a || !b) return null;
  return [
    (a[0] + b[0]) * 0.5,
    (a[1] + b[1]) * 0.5,
    Math.min(a[2] ?? 1, b[2] ?? 1),
  ];
}

function angleDiffDeg(a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  let d = Math.abs(a - b) % 360;
  if (d > 180) d = 360 - d;
  return d;
}

function getWebcamPoseAngles(kpts) {
  if (!kpts) return null;

  const ls = kpts[11];
  const rs = kpts[12];
  const lh = kpts[23];
  const rh = kpts[24];
  const lk = kpts[25];
  const rk = kpts[26];
  const la = kpts[27];
  const ra = kpts[28];

  const torsoTop = avgPoint(ls, rs);
  const torsoBottom = avgPoint(lh, rh);
  const leftHip = lh;
  const rightHip = rh;
  const leftKnee = lk;
  const rightKnee = rk;
  const leftAnkle = la;
  const rightAnkle = ra;

  if (
    !torsoTop || !torsoBottom ||
    !leftHip || !rightHip || !leftKnee || !rightKnee || !leftAnkle || !rightAnkle
  ) {
    return null;
  }

  return {
    upper: angle2D(torsoBottom, torsoTop),
    leftLeg: angle2D(leftHip, leftKnee),
    rightLeg: angle2D(rightHip, rightKnee),
    leftShin: angle2D(leftKnee, leftAnkle),
    rightShin: angle2D(rightKnee, rightAnkle),
  };
}

function detectSquatPhase(kpts) {
  if (!kpts) return "unknown";

  const lh = kpts[23];
  const rh = kpts[24];
  const lk = kpts[25];
  const rk = kpts[26];

  if (![lh, rh, lk, rk].every((p) => kpOk(p, 0.08))) return "unknown";

  const hipY = (lh[1] + rh[1]) * 0.5;
  const kneeY = (lk[1] + rk[1]) * 0.5;

  const depth = clamp01(1 - ((kneeY - hipY - 0.02) / 0.30));

  if (depth < 0.42) return "stand";
  if (depth > 0.72) return "squat";
  return "mid";
}

/* =========================
 * FEEDBACK EVAL
 * ========================= */
function evaluatePartFeedbackLoose(webcamKpts) {
  resetPartFeedback();

  if (!webcamKpts) {
    state.partFeedback.upper = "missing";
    state.partFeedback.leftLeg = "missing";
    state.partFeedback.rightLeg = "missing";
    state.partFeedback.knee = "missing";
    setFeedbackText("관절 인식 부족");
    return;
  }

  const coreIdx = [11, 12, 23, 24, 25, 26];
  const ankleVisible = kpOk(webcamKpts[27], 0.05) && kpOk(webcamKpts[28], 0.05);
  const coreMissing = coreIdx.some((i) => !kpOk(webcamKpts[i], 0.08));

  if (coreMissing) {
    state.partFeedback.upper = "missing";
    state.partFeedback.leftLeg = "missing";
    state.partFeedback.rightLeg = "missing";
    state.partFeedback.knee = "missing";
    setFeedbackText("관절 인식 부족");
    return;
  }

  const angles = getWebcamPoseAngles(webcamKpts);
  const phase = detectSquatPhase(webcamKpts);

  if (!angles || phase === "unknown") {
    state.partFeedback.upper = "missing";
    state.partFeedback.leftLeg = "missing";
    state.partFeedback.rightLeg = "missing";
    state.partFeedback.knee = "missing";
    setFeedbackText("관절 인식 부족");
    return;
  }

  if (state.isInReadyPose) {
    state.partFeedback.upper = "normal";
    state.partFeedback.leftLeg = "normal";
    state.partFeedback.rightLeg = "normal";
    state.partFeedback.knee = "normal";
    setFeedbackText("준비중");
    return;
  }

  const ls = webcamKpts[11];
  const rs = webcamKpts[12];
  const lh = webcamKpts[23];
  const rh = webcamKpts[24];
  const lk = webcamKpts[25];
  const rk = webcamKpts[26];
  const la = webcamKpts[27];
  const ra = webcamKpts[28];

  const hipCenterX = ((lh?.[0] ?? 0) + (rh?.[0] ?? 0)) * 0.5;
  const kneeCenterX = ((lk?.[0] ?? 0) + (rk?.[0] ?? 0)) * 0.5;
  const shoulderCenterX = ((ls?.[0] ?? 0) + (rs?.[0] ?? 0)) * 0.5;

  const hipYAvg = ((lh?.[1] ?? 0) + (rh?.[1] ?? 0)) * 0.5;
  const kneeYAvg = ((lk?.[1] ?? 0) + (rk?.[1] ?? 0)) * 0.5;
  const ankleYAvg =
    ((la?.[1] ?? kneeYAvg + 0.18) + (ra?.[1] ?? kneeYAvg + 0.18)) * 0.5;

  const squatDepth = clamp01(1 - ((kneeYAvg - hipYAvg - 0.04) / 0.24));

  const hipWidth = Math.abs((lh?.[0] ?? 0) - (rh?.[0] ?? 0));
  const bodyScale = Math.max(0.08, Math.abs(kneeYAvg - hipYAvg) + hipWidth);

  const kneeYOffset = Math.abs((lk?.[1] ?? 0) - (rk?.[1] ?? 0));
  const hipYOffset = Math.abs((lh?.[1] ?? 0) - (rh?.[1] ?? 0));
  const torsoLeanX = Math.abs(shoulderCenterX - hipCenterX);
  const kneeCenterOffsetX = Math.abs(kneeCenterX - hipCenterX);

  const leftLegLen =
    Math.hypot((lk?.[0] ?? 0) - (lh?.[0] ?? 0), (lk?.[1] ?? 0) - (lh?.[1] ?? 0)) +
    (ankleVisible
      ? Math.hypot((la?.[0] ?? 0) - (lk?.[0] ?? 0), (la?.[1] ?? 0) - (lk?.[1] ?? 0))
      : 0.18);

  const rightLegLen =
    Math.hypot((rk?.[0] ?? 0) - (rh?.[0] ?? 0), (rk?.[1] ?? 0) - (rh?.[1] ?? 0)) +
    (ankleVisible
      ? Math.hypot((ra?.[0] ?? 0) - (rk?.[0] ?? 0), (ra?.[1] ?? 0) - (rk?.[1] ?? 0))
      : 0.18);

  const legLenAvg = Math.max(0.001, (leftLegLen + rightLegLen) * 0.5);
  const legLenDiffRatio = Math.abs(leftLegLen - rightLegLen) / legLenAvg;

  const leftKneeAngle = angleDiffDeg(angles.leftLeg, angles.leftShin);
  const rightKneeAngle = angleDiffDeg(angles.rightLeg, angles.rightShin);
  const kneeDiffLR = Math.abs((leftKneeAngle ?? 0) - (rightKneeAngle ?? 0));

  let upperState = "normal";
  let leftLegState = "normal";
  let rightLegState = "normal";
  let kneeState = "normal";

  const tryingSquat = squatDepth > 0.45;

  const depthTooShallowBad =
    tryingSquat && squatDepth < 0.52;

  const depthTooShallowWarn =
    tryingSquat && squatDepth >= 0.52 && squatDepth < 0.68;

  if (depthTooShallowWarn) {
    leftLegState = "warn";
    rightLegState = "warn";
    kneeState = "warn";
  }

  if (depthTooShallowBad) {
    leftLegState = "bad";
    rightLegState = "bad";
    kneeState = "bad";
  }

  if (torsoLeanX / bodyScale > 0.60) upperState = "warn";
  if (torsoLeanX / bodyScale > 0.82) upperState = "bad";

  if (kneeYOffset > 0.10 || hipYOffset > 0.08) {
    if (leftLegState !== "bad") leftLegState = "warn";
    if (rightLegState !== "bad") rightLegState = "warn";
  }
  if (kneeYOffset > 0.18 || hipYOffset > 0.14) {
    leftLegState = "bad";
    rightLegState = "bad";
  }

  const oneLegLike =
    kneeYOffset > 0.18 ||
    legLenDiffRatio > 0.28 ||
    (ankleVisible && Math.abs((la?.[1] ?? 0) - (ra?.[1] ?? 0)) > 0.20);

  if (oneLegLike) {
    leftLegState = "bad";
    rightLegState = "bad";
    kneeState = "bad";
  }

  if (kneeDiffLR > 20 || kneeCenterOffsetX / bodyScale > 0.32) {
    if (kneeState !== "bad") kneeState = "warn";
  }
  if (kneeDiffLR > 35 || kneeCenterOffsetX / bodyScale > 0.48) {
    kneeState = "bad";
  }

  if (phase === "squat") {
    const tooCompressed = Math.abs(ankleYAvg - hipYAvg) < 0.12;
    const kneesClearlyAboveHip =
      (lk?.[1] ?? 0) < (lh?.[1] ?? 0) - 0.10 ||
      (rk?.[1] ?? 0) < (rh?.[1] ?? 0) - 0.10;

    if (tooCompressed || kneesClearlyAboveHip) {
      if (leftLegState === "normal") leftLegState = "warn";
      if (rightLegState === "normal") rightLegState = "warn";
      if (kneeState === "normal") kneeState = "warn";
    }
  }

  state.partFeedback.upper = upperState;
  state.partFeedback.leftLeg = leftLegState;
  state.partFeedback.rightLeg = rightLegState;
  state.partFeedback.knee = kneeState;

  const hasBad =
    upperState === "bad" ||
    leftLegState === "bad" ||
    rightLegState === "bad" ||
    kneeState === "bad";

  const hasWarn =
    upperState === "warn" ||
    leftLegState === "warn" ||
    rightLegState === "warn" ||
    kneeState === "warn";

  if (depthTooShallowBad) {
    setFeedbackText("깊이 부족");
  } else if (depthTooShallowWarn) {
    setFeedbackText("조금 더 앉아주세요");
  } else if (hasBad) {
    setFeedbackText("자세 불안정");
  } else if (hasWarn) {
    setFeedbackText("자세 주의");
  } else {
    setFeedbackText("자세 양호");
  }
}

/* =========================
 * DRAW
 * ========================= */
function drawWebcamFrame() {
  if (!dom.video || dom.video.readyState < 2) return;

  const w = dom.canvas.width;
  const h = dom.canvas.height;
  if (!w || !h) return;

  clearCanvas();
  ctx.drawImage(dom.video, 0, 0, w, h);
}

function drawGuideVideoFrame() {
  if (!dom.guideVideo || dom.guideVideo.readyState < 2) return;

  const w = dom.canvas.width;
  const h = dom.canvas.height;
  if (!w || !h) return;

  clearCanvas();
  ctx.drawImage(dom.guideVideo, 0, 0, w, h);
}

function handLmOk(p) {
  return !!p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z);
}

function drawHandOverlay(w, h) {
  const picked = pickGestureHand(state.lastHandResult);
  const lm = picked?.lm;
  if (!lm) return;

  octx.lineWidth = 1.5;
  octx.strokeStyle = FEEDBACK_COLORS.hand;
  octx.fillStyle = FEEDBACK_COLORS.hand;

  for (const [a, b] of HAND_EDGES) {
    const pa = lm[a];
    const pb = lm[b];
    if (!handLmOk(pa) || !handLmOk(pb)) continue;
    octx.beginPath();
    octx.moveTo(pa.x * w, pa.y * h);
    octx.lineTo(pb.x * w, pb.y * h);
    octx.stroke();
  }

  for (const p of lm) {
    if (!handLmOk(p)) continue;
    octx.beginPath();
    octx.arc(p.x * w, p.y * h, 2.5, 0, Math.PI * 2);
    octx.fill();
  }
}

function getMpEdgePart(a, b) {
  const pair = [a, b].sort((x, y) => x - y).join("-");

  const upperPairs = new Set([
    "11-12", "11-13", "13-15", "12-14", "14-16", "11-23", "12-24", "23-24",
  ]);

  const leftLegPairs = new Set([
    "23-25", "25-27", "27-29", "29-31",
  ]);

  const rightLegPairs = new Set([
    "24-26", "26-28", "28-30", "30-32",
  ]);

  if (upperPairs.has(pair)) return "upper";
  if (leftLegPairs.has(pair)) return "leftLeg";
  if (rightLegPairs.has(pair)) return "rightLeg";
  return "upper";
}

function getJsonEdgePart(a, b) {
  const pair = [a, b].sort((x, y) => x - y).join("-");

  const upperPairs = new Set([
    "5-6", "5-7", "7-9", "6-8", "8-10", "5-11", "6-12", "11-12",
  ]);

  const leftLegPairs = new Set([
    "11-13", "13-15",
  ]);

  const rightLegPairs = new Set([
    "12-14", "14-16",
  ]);

  if (upperPairs.has(pair)) return "upper";
  if (leftLegPairs.has(pair)) return "leftLeg";
  if (rightLegPairs.has(pair)) return "rightLeg";
  return "upper";
}

function getJointPart(index) {
  if ([11, 12, 23, 24, 5, 6].includes(index)) return "upper";
  if ([25, 27, 29, 31, 13, 15].includes(index)) return "leftLeg";
  if ([26, 28, 30, 32, 14, 16].includes(index)) return "rightLeg";
  return "upper";
}

function getJointOverrideState(index) {
  if ([25, 26, 13, 14].includes(index)) {
    if (state.partFeedback.knee === "bad") return "bad";
    if (state.partFeedback.knee === "warn") return "warn";
    if (state.partFeedback.knee === "missing") return "missing";
  }

  const part = getJointPart(index);
  return state.partFeedback[part] || "normal";
}

function drawPoseOverlayByPart(kpts, edges, edgePartResolver) {
  clearOverlay();
  if (!kpts) return;

  const w = dom.overlay.width;
  const h = dom.overlay.height;
  octx.shadowBlur = 0;

  for (const [a, b] of edges) {
    const pa = kpts[a];
    const pb = kpts[b];
    const part = edgePartResolver(a, b);

    let color = FEEDBACK_COLORS.line;
    if (edgeHasMissing(pa, pb)) {
      color = FEEDBACK_COLORS.missing;
    } else {
      const partState = state.partFeedback[part] || "normal";
      color = getPartColor(partState, false);
    }

    if (!pa || !pb) continue;

    octx.beginPath();
    octx.lineWidth = 3;
    octx.strokeStyle = color;
    octx.moveTo(pa[0] * w, pa[1] * h);
    octx.lineTo(pb[0] * w, pb[1] * h);
    octx.stroke();
  }

  for (let i = 0; i < kpts.length; i++) {
    const p = kpts[i];
    if (!p) continue;

    let color = FEEDBACK_COLORS.joint;

    if (pointState(p) === "missing") {
      color = FEEDBACK_COLORS.missing;
    } else {
      const jointState = getJointOverrideState(i);
      color = getPartColor(jointState, true);
    }

    octx.beginPath();
    octx.fillStyle = color;
    octx.arc(p[0] * w, p[1] * h, 4, 0, Math.PI * 2);
    octx.fill();
  }
}

function drawSkeleton2D(kpts) {
  if (!kpts) {
    clearOverlay();
    return;
  }

  drawPoseOverlayByPart(kpts, MP_POSE_EDGES, getMpEdgePart);

  if (
    state.lastHandResult?.landmarks?.length &&
    ((state.currentMode === "feedback" && state.feedbackDisplay === "webcam") ||
      (state.currentMode === "avatar" && state.avatarDisplay === "webcam"))
  ) {
    drawHandOverlay(dom.overlay.width, dom.overlay.height);
  }
}

function drawJsonSkeletonFromGuideVideo() {
  if (!dom.guideVideo || dom.guideVideo.readyState < 2) {
    clearOverlay();
    return;
  }

  const t = Math.max(0, (dom.guideVideo.currentTime || 0) + JSON_TIME_OFFSET_SEC);
  const frame = getPoseFrameAtTimeSec(t);
  const kpts = normalizeJsonKpts(frame);

  if (!kpts) {
    clearOverlay();
    return;
  }

  drawPoseOverlayByPart(kpts, JSON_POSE_EDGES, getJsonEdgePart);
}

function updateSquatProgressFromPose(kpts) {
  if (!kpts) return;

  const lh = kpts[23] ?? kpts[11];
  const rh = kpts[24] ?? kpts[12];
  const lk = kpts[25] ?? kpts[13];
  const rk = kpts[26] ?? kpts[14];

  if (![lh, rh, lk, rk].every((p) => kpOk(p))) return;

  const hipY = (lh[1] + rh[1]) * 0.5;
  const kneeY = (lk[1] + rk[1]) * 0.5;

  const rawDepth = clamp01(1 - ((kneeY - hipY - 0.05) / 0.20));

  state.squatProgressRaw = rawDepth;

  const alpha = state.currentMode === "avatar" ? 0.9 : 0.35;
  state.squatProgressSmooth = lerp(state.squatProgressSmooth, rawDepth, alpha);

  if (state.squatState === "UP" && state.squatProgressSmooth > 0.55) {
    state.squatState = "DOWN";
  } else if (state.squatState === "DOWN" && state.squatProgressSmooth < 0.20) {
    state.squatState = "UP";
    state.squatCount += 1;
    updateCounterChip();
  }
}

/* =========================
 * HANDS
 * ========================= */
function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function fingerExtended(lm, tipIdx, pipIdx) {
  if (!handLmOk(lm[tipIdx]) || !handLmOk(lm[pipIdx])) return false;
  return lm[tipIdx].y < lm[pipIdx].y;
}

function isFistLandmarks(lm) {
  if (!lm || lm.length < 21) return false;

  const wrist = lm[0];
  const indexTip = lm[8];
  const middleTip = lm[12];
  const ringTip = lm[16];
  const pinkyTip = lm[20];
  const indexMcp = lm[5];
  const pinkyMcp = lm[17];

  if (!wrist || !indexTip || !middleTip || !ringTip || !pinkyTip || !indexMcp || !pinkyMcp) {
    return false;
  }

  const palmWidth = dist2(indexMcp, pinkyMcp);
  if (palmWidth < 0.03) return false;

  const d1 = dist2(indexTip, wrist);
  const d2 = dist2(middleTip, wrist);
  const d3 = dist2(ringTip, wrist);
  const d4 = dist2(pinkyTip, wrist);

  return (
    d1 < palmWidth * 1.25 &&
    d2 < palmWidth * 1.25 &&
    d3 < palmWidth * 1.25 &&
    d4 < palmWidth * 1.25
  );
}

function isKnifeHand(lm) {
  if (!lm || lm.length < 21) return false;

  const indexExt = fingerExtended(lm, 8, 6);
  const middleExt = fingerExtended(lm, 12, 10);
  const ringExt = fingerExtended(lm, 16, 14);
  const pinkyExt = fingerExtended(lm, 20, 18);

  if (!(indexExt && middleExt && ringExt && pinkyExt)) return false;

  const spread =
    Math.abs(lm[8].x - lm[12].x) +
    Math.abs(lm[12].x - lm[16].x) +
    Math.abs(lm[16].x - lm[20].x);

  return spread < 0.22;
}

function isHandCentered(lm) {
  const wrist = lm?.[0];
  if (!wrist) return false;
  return wrist.x > 0.22 && wrist.x < 0.78 && wrist.y > 0.12 && wrist.y < 0.88;
}

function decayGestureHandLock() {
  if (state.gestureHandLockFrames > 0) {
    state.gestureHandLockFrames -= 1;
  }

  if (state.gestureHandLockFrames <= 0) {
    state.gestureHandLabel = null;
    state.gestureHandLockFrames = 0;
  }
}

function pickGestureHand(result) {
  const hands = result?.landmarks || [];
  const handednesses = result?.handednesses || [];

  if (!hands.length) return null;

  const candidates = hands.map((lm, i) => {
    const wrist = lm?.[0];
    const label = handednesses?.[i]?.[0]?.categoryName || `Hand${i}`;
    return {
      lm,
      label,
      wristY: wrist?.y ?? 999,
    };
  });

  if (state.gestureHandLabel) {
    const locked = candidates.find((c) => c.label === state.gestureHandLabel);
    if (locked) {
      state.gestureHandLockFrames = Math.min(state.gestureHandLockFrames + 1, 12);
      return locked;
    }
  }

  candidates.sort((a, b) => a.wristY - b.wristY);
  const picked = candidates[0];

  if (picked) {
    state.gestureHandLabel = picked.label;
    state.gestureHandLockFrames = 8;
  }

  return picked;
}

function getPalmFacingState(lm, handednessLabel) {
  if (!lm || lm.length < 21) return "UNKNOWN";

  const wrist = lm[0];
  const indexMcp = lm[5];
  const pinkyMcp = lm[17];

  if (!handLmOk(wrist) || !handLmOk(indexMcp) || !handLmOk(pinkyMcp)) {
    return "UNKNOWN";
  }

  const vx1 = indexMcp.x - wrist.x;
  const vy1 = indexMcp.y - wrist.y;
  const vx2 = pinkyMcp.x - wrist.x;
  const vy2 = pinkyMcp.y - wrist.y;

  const cross = vx1 * vy2 - vy1 * vx2;

  if (handednessLabel === "Right") {
    return cross < 0 ? "PALM" : "BACK";
  }
  if (handednessLabel === "Left") {
    return cross > 0 ? "PALM" : "BACK";
  }
  return "UNKNOWN";
}

function getKnifeRotateState(lm, handednessLabel) {
  if (!isKnifeHand(lm)) return "NEUTRAL";

  const facing = getPalmFacingState(lm, handednessLabel);
  if (facing === "PALM") return "RIGHT";
  if (facing === "BACK") return "LEFT";
  return "NEUTRAL";
}

function toggleCurrentModeDisplay() {
  if (state.currentMode === "feedback") {
    state.feedbackDisplay = state.feedbackDisplay === "webcam" ? "guide" : "webcam";
    clearFeedbackToasts();
    applyFeedbackDisplayMode();
  } else if (state.currentMode === "avatar") {
    state.avatarDisplay = state.avatarDisplay === "avatar" ? "webcam" : "avatar";
    applyAvatarDisplayMode();
  }
}

function handleFistToggle(lm) {
  const fist = isFistLandmarks(lm) && isHandCentered(lm);

  if (!fist) {
    state.fistHoldCount = 0;
    return false;
  }

  state.fistHoldCount += 1;
  dom.gestureChip.textContent = `GESTURE: FIST ${state.fistHoldCount}/${FIST_TOGGLE_HOLD_FRAMES}`;

  if (state.fistHoldCount >= FIST_TOGGLE_HOLD_FRAMES && !state.fistCooldown) {
    state.fistCooldown = true;
    state.fistHoldCount = 0;
    toggleCurrentModeDisplay();

    setTimeout(() => {
      state.fistCooldown = false;
    }, FIST_TOGGLE_COOLDOWN_MS);

    return true;
  }

  return false;
}

async function inferHands() {
  if (
    state.stopped ||
    !state.handLandmarker ||
    !dom.video.videoWidth ||
    state.handBusy ||
    (state.currentMode !== "feedback" && state.currentMode !== "avatar")
  ) return;

  state.handBusy = true;

  try {
    state.lastHandResult = state.handLandmarker.detectForVideo(dom.video, performance.now());

    const picked = pickGestureHand(state.lastHandResult);
    const lm = picked?.lm || null;
    const handedness = picked?.label || null;
    state.lastHandedness = handedness;

    if (!lm) {
      decayGestureHandLock();
      state.handRotateState = "NEUTRAL";
      state.fistHoldCount = 0;
      dom.gestureChip.textContent = "GESTURE: -";
      return;
    }

    const toggled = handleFistToggle(lm);
    if (toggled) return;

    if (state.currentMode === "avatar" && state.avatarDisplay === "avatar") {
      state.handRotateState = getKnifeRotateState(lm, handedness);
      dom.gestureChip.textContent = `GESTURE: ${state.handRotateState} (${handedness})`;
    } else {
      state.handRotateState = "NEUTRAL";
      dom.gestureChip.textContent = `GESTURE: HAND (${handedness})`;
    }
  } catch (err) {
    console.warn("HAND ERROR:", err);
  } finally {
    state.handBusy = false;
  }
}

/* =========================
 * THREE
 * ========================= */
function getAvatarLookTarget() {
  return AVATAR_POS.clone().add(LOOK_TARGET_OFFSET);
}

function getClipRange() {
  if (!squatClip) return { clipStart: 0, usableDuration: 0 };
  const clipDuration = Math.max(0.0001, squatClip.duration);
  const clipEndNorm =
    state.currentMode === "standard" ? STANDARD_CLIP_END_NORM : AVATAR_CLIP_END_NORM;

  const clipStart = clipDuration * CLIP_START_NORM;
  const clipEnd = clipDuration * clipEndNorm;
  const usableDuration = Math.max(0.0001, clipEnd - clipStart);

  return { clipStart, usableDuration };
}

function initThree() {
  if (renderer) return;

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
    alpha: false,
  });

  renderer.setPixelRatio(window.devicePixelRatio || 1);
  dom.threeWrap.innerHTML = "";
  dom.threeWrap.appendChild(renderer.domElement);

  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.style.display = "block";

  scene = new THREE.Scene();
  scene.background = new THREE.Color(state.themeLight ? 0xf4f7f6 : 0x0a0a0c);

  camera = new THREE.PerspectiveCamera(52, 1, 0.1, 2000);
  camera.position.set(0, 1.3, 2.6);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444466, 1.15);
  hemi.position.set(0, 2, 0);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(2, 4, 2);
  scene.add(dir);

  resizeThreeRenderer();
}

async function loadGym(loader) {
  if (gymRoot) return;

  await new Promise((resolve, reject) => {
    loader.load(
      GYM_URL,
      (gltf) => {
        gymRoot = gltf.scene;
        gymRoot.traverse((obj) => {
          if (obj.isMesh && obj.material) obj.material.side = THREE.DoubleSide;
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
  if (avatarScene) return;

  try {
    await new Promise((resolve, reject) => {
      loader.load(
        AVATAR_URL,
        (gltf) => {
          avatarScene = gltf.scene;
          avatarScene.position.copy(AVATAR_POS);
          scene.add(avatarScene);

          const clips = gltf.animations || [];
          if (clips.length > 0) {
            mixer = new THREE.AnimationMixer(avatarScene);
            squatClip = clips[0];
            squatAction = mixer.clipAction(squatClip);
            squatAction.enabled = true;
            squatAction.setLoop(THREE.LoopOnce, 1);
            squatAction.clampWhenFinished = true;
            squatAction.play();
            squatAction.paused = true;

            const { clipStart } = getClipRange();
            squatAction.time = clipStart;
            mixer.update(0);
            avatarScene.updateMatrixWorld(true);
          }

          resolve();
        },
        undefined,
        reject
      );
    });
  } catch (err) {
    console.warn("AVATAR LOAD FAIL:", err);
    const fallback = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, 1.6, 0.4),
      new THREE.MeshStandardMaterial({ color: 0x1fe4ff })
    );
    fallback.position.copy(AVATAR_POS);
    avatarScene = fallback;
    scene.add(avatarScene);
  }

  const target = getAvatarLookTarget();
  camera.position.set(
    target.x + FRONT_CAM_OFFSET.x,
    target.y + FRONT_CAM_OFFSET.y,
    target.z + FRONT_CAM_OFFSET.z
  );
  camera.lookAt(target);
}

function updateProgressControlledAnimation() {
  if (!mixer || !squatClip || !squatAction) return;

  const { clipStart, usableDuration } = getClipRange();

  const animProgress = state.squatProgressSmooth;
  const targetTime = clipStart + usableDuration * clamp01(animProgress);

  squatAction.time = targetTime;
  mixer.update(0);
  avatarScene?.updateMatrixWorld(true);
}

function applyHandCameraControl() {
  if (
    !camera ||
    !avatarScene ||
    state.currentMode !== "avatar" ||
    state.avatarDisplay !== "avatar"
  ) return;

  if (!state.cameraRotateEnabled) return;

  if (state.handRotateState === "LEFT") {
    state.handCurrentYaw -= CAMERA_ROTATE_STEP;
  } else if (state.handRotateState === "RIGHT") {
    state.handCurrentYaw += CAMERA_ROTATE_STEP;
  } else {
    return;
  }

  state.handCurrentYaw = clamp(state.handCurrentYaw, -Math.PI * 0.95, Math.PI * 0.95);

  const target = getAvatarLookTarget();
  const radius = Math.sqrt(
    FRONT_CAM_OFFSET.x * FRONT_CAM_OFFSET.x +
    FRONT_CAM_OFFSET.z * FRONT_CAM_OFFSET.z
  );

  const x = target.x + Math.sin(state.handCurrentYaw) * radius;
  const z = target.z + Math.cos(state.handCurrentYaw) * radius;
  const y = target.y + FRONT_CAM_OFFSET.y;

  camera.position.set(x, y, z);
  camera.lookAt(target);

  if (controls) {
    controls.target.copy(target);
  }
}

function animateThree(now = 0) {
  if (state.stopped) return;

  renderRafId = requestAnimationFrame(animateThree);

  applyHandCameraControl();

  if (controls && controls.enabled) {
    controls.update();
  }

  updateProgressControlledAnimation();

  if (now - lastRenderTime < RENDER_INTERVAL) return;
  lastRenderTime = now;

  if (renderer && scene && camera && dom.threeWrap.style.display !== "none") {
    renderer.render(scene, camera);
  }
}

/* =========================
 * DISPLAY APPLY
 * ========================= */
function applyStandardDisplayMode() {
  if (state.currentMode !== "standard") return;

  dom.video.style.display = "none";
  dom.canvas.style.display = "none";
  dom.overlay.style.display = "none";

  dom.guideVideo.style.display = "block";
  dom.threeWrap.style.display = "block";

  dom.guideVideo.classList.remove("stage-main", "stage-pip");
  dom.threeWrap.classList.remove("stage-main", "stage-pip");

  showCounter(false);

  if (state.standardDisplay === "avatarMain") {
    dom.threeWrap.classList.add("stage-main");
    dom.guideVideo.classList.add("stage-pip");
    dom.modeChip.textContent = "정석 모드 · 아바타 메인";
    setHelp("작은 비디오 더블클릭 시 비디오/아바타 위치 전환");
  } else {
    dom.guideVideo.classList.add("stage-main");
    dom.threeWrap.classList.add("stage-pip");
    dom.modeChip.textContent = "정석 모드 · 비디오 메인";
    setHelp("작은 아바타 더블클릭 시 비디오/아바타 위치 전환");
  }

  requestAnimationFrame(() => {
    resizeThreeRenderer();
  });
}

function toggleStandardDisplay() {
  if (state.currentMode !== "standard") return;
  state.standardDisplay =
    state.standardDisplay === "avatarMain" ? "videoMain" : "avatarMain";
  applyStandardDisplayMode();
}

function applyFeedbackDisplayMode() {
  if (state.currentMode !== "feedback") return;

  const showWebcam = state.feedbackDisplay === "webcam";

  dom.video.style.display = "none";
  dom.canvas.style.display = "block";
  dom.overlay.style.display = "block";
  dom.guideVideo.style.display = "none";
  dom.threeWrap.style.display = "none";

  showCounter(false);

  dom.modeChip.textContent = showWebcam ? "피드백 모드 · 웹캠" : "피드백 모드 · 비디오";
  setHelp("주먹 인식 시 webcam / video 전환");
}

function applyAvatarDisplayMode() {
  if (state.currentMode !== "avatar") return;

  const showWebcam = state.avatarDisplay === "webcam";

  dom.video.style.display = "none";
  dom.canvas.style.display = showWebcam ? "block" : "none";
  dom.overlay.style.display = showWebcam ? "block" : "none";
  dom.guideVideo.style.display = "none";
  dom.threeWrap.style.display = showWebcam ? "none" : "block";

  dom.threeWrap.classList.remove("stage-main", "stage-pip");
  if (!showWebcam) {
    dom.threeWrap.classList.add("stage-main");
  }

  showCounter(!showWebcam);

  dom.modeChip.textContent = showWebcam ? "아바타 모드 · 웹캠" : "아바타 모드 · 아바타";
  setHelp(
    showWebcam
      ? "주먹 인식 시 avatar 전환"
      : "주먹 인식 시 webcam 전환 / 손날 회전으로 카메라 회전"
  );

  if (!showWebcam) {
    clearOverlay();
    requestAnimationFrame(() => {
      resizeThreeRenderer();
    });
  }
}

/* =========================
 * DRAW LOOPS
 * ========================= */
function startFeedbackDrawLoop() {
  if (state.drawTimer) clearInterval(state.drawTimer);

  state.drawTimer = setInterval(() => {
    if (state.stopped) return;

    const webcamPose = state.smoothPoseKpts;

    if (webcamPose) {
      updateReadyPoseState(webcamPose);
      updateShallowHoldState(webcamPose);
      updateSquatProgressFromPose(webcamPose);

      if (poseHasEnoughCore(webcamPose)) {
        evaluatePartFeedbackLoose(webcamPose);
        emitDetailedFeedbackToasts();
      } else {
        resetPartFeedback();
        state.partFeedback.upper = "missing";
        state.partFeedback.leftLeg = "missing";
        state.partFeedback.rightLeg = "missing";
        state.partFeedback.knee = "missing";
        setFeedbackText("관절 인식 부족");
        emitDetailedFeedbackToasts();
      }
    } else {
      state.isInReadyPose = true;
      state.shallowHoldFrames = 0;
      state.shallowHoldActive = false;
      resetPartFeedback();
      state.partFeedback.upper = "missing";
      state.partFeedback.leftLeg = "missing";
      state.partFeedback.rightLeg = "missing";
      state.partFeedback.knee = "missing";
      setFeedbackText("관절 인식 부족");
      emitDetailedFeedbackToasts();
    }

    if (state.feedbackDisplay === "webcam") {
      drawWebcamFrame();

      if (webcamPose) {
        drawSkeleton2D(webcamPose);
      } else {
        clearOverlay();
      }
    } else {
      drawGuideVideoFrame();

      if (dom.guideVideo.readyState >= 2) {
        drawJsonSkeletonFromGuideVideo();
      } else {
        clearOverlay();
      }
    }
  }, DRAW_INTERVAL);
}

function startAvatarDrawLoop() {
  if (state.drawTimer) clearInterval(state.drawTimer);

  state.drawTimer = setInterval(() => {
    if (state.stopped) return;

    const webcamPose = state.smoothPoseKpts;

    if (webcamPose) {
      updateReadyPoseState(webcamPose);
      updateShallowHoldState(webcamPose);
      updateSquatProgressFromPose(webcamPose);

      if (poseHasEnoughCore(webcamPose)) {
        evaluatePartFeedbackLoose(webcamPose);
      } else {
        resetPartFeedback();
        state.partFeedback.upper = "missing";
        state.partFeedback.leftLeg = "missing";
        state.partFeedback.rightLeg = "missing";
        state.partFeedback.knee = "missing";
        setFeedbackText("관절 인식 부족");
      }
    } else {
      state.isInReadyPose = true;
      state.shallowHoldFrames = 0;
      state.shallowHoldActive = false;
      resetPartFeedback();
      state.partFeedback.upper = "missing";
      state.partFeedback.leftLeg = "missing";
      state.partFeedback.rightLeg = "missing";
      state.partFeedback.knee = "missing";
      setFeedbackText("관절 인식 부족");
    }

    if (state.avatarDisplay === "webcam") {
      drawWebcamFrame();

      if (webcamPose) {
        drawSkeleton2D(webcamPose);
      } else {
        clearOverlay();
      }
    } else {
      clearCanvas();
      clearOverlay();
    }
  }, DRAW_INTERVAL);
}

/* =========================
 * MODES
 * ========================= */
function syncStandardToVideoFrame() {
  if (state.stopped || state.currentMode !== "standard") return;

  const t = Math.max(0, (dom.guideVideo.currentTime || 0) + JSON_TIME_OFFSET_SEC);
  const frame = getPoseFrameAtTimeSec(t);
  const depth = computeDepthFromFrame(frame);

  state.squatProgressRaw = depth;
  state.squatProgressSmooth = depth;

  if (
    typeof dom.guideVideo.requestVideoFrameCallback === "function" &&
    !state.stopped &&
    state.currentMode === "standard"
  ) {
    state.jsonVideoFrameHandle = dom.guideVideo.requestVideoFrameCallback(syncStandardToVideoFrame);
  }
}

async function startStandardMode() {
  await destroyCurrentMode();

  clearFeedbackToasts();
  state.feedbackToastCooldowns = {};
  state.lastFeedbackSnapshot = "";

  state.currentMode = "standard";
  state.stopped = false;
  state.standardDisplay = "avatarMain";
  state.squatCount = 0;
  state.squatProgressSmooth = 0;
  state.squatState = "UP";
  state.handRotateState = "NEUTRAL";
  state.fistHoldCount = 0;
  state.gestureHandLabel = null;
  state.gestureHandLockFrames = 0;
  state.handCurrentYaw = 0;
  state.lastRenderedCount = 0;
  state.shallowHoldFrames = 0;
  state.shallowHoldActive = false;

  showAppScreen();
  resetViewVisibility();
  showCounter(false);
  updateCounterChip(true);

  dom.gestureChip.textContent = "GESTURE: -";
  setStatus("LOADING...");
  setHelp("아바타 + 비디오/JSON 보기 전용");

  dom.manualToggleBtn.style.display = "none";
  dom.cameraRotateToggle.checked = false;
  state.cameraRotateEnabled = false;

  await startGuideVideo();
  dom.guideVideo.currentTime = 0;
  await loadPoseJson();

  initThree();
  if (controls) {
    controls.enabled = true;
  }

  const loader = new GLTFLoader();
  await loadGym(loader);
  await loadAnimatedAvatar(loader);

  applyStandardDisplayMode();

  requestAnimationFrame(() => {
    resizeThreeRenderer();
  });

  if (typeof dom.guideVideo.requestVideoFrameCallback === "function") {
    state.jsonVideoFrameHandle = dom.guideVideo.requestVideoFrameCallback(syncStandardToVideoFrame);
  } else {
    state.jsonTimer = setInterval(() => {
      if (state.stopped) return;

      const t = Math.max(0, (dom.guideVideo.currentTime || 0) + JSON_TIME_OFFSET_SEC);
      const frame = getPoseFrameAtTimeSec(t);
      const depth = computeDepthFromFrame(frame);

      state.squatProgressRaw = depth;
      state.squatProgressSmooth = depth;
    }, 1000 / Math.max(1, state.poseFPS || 30));
  }

  animateThree();
  setStatus("READY");
}

async function startFeedbackMode() {
  await destroyCurrentMode();

  clearFeedbackToasts();
  state.feedbackToastCooldowns = {};
  state.lastFeedbackSnapshot = "";

  state.currentMode = "feedback";
  state.stopped = false;
  state.feedbackDisplay = "webcam";
  state.squatCount = 0;
  state.squatProgressSmooth = 0;
  state.squatState = "UP";
  state.lastPoseKpts = null;
  state.smoothPoseKpts = null;
  state.poseVelocity = null;
  state.poseLock.active = false;
  state.poseLock.stableFrames = 0;
  state.poseLock.baseCenter = null;
  state.handRotateState = "NEUTRAL";
  state.fistHoldCount = 0;
  state.gestureHandLabel = null;
  state.gestureHandLockFrames = 0;
  state.lastRenderedCount = 0;
  state.shallowHoldFrames = 0;
  state.shallowHoldActive = false;
  state.isInReadyPose = true;

  showAppScreen();
  resetViewVisibility();
  showCounter(false);
  updateCounterChip(true);

  dom.modeChip.textContent = "피드백 모드";
  dom.gestureChip.textContent = "GESTURE: -";
  setFeedbackText("LOADING...");

  dom.manualToggleBtn.style.display = "inline-flex";
  dom.cameraRotateToggle.checked = false;
  state.cameraRotateEnabled = false;

  await startWebcam();
  await startGuideVideo();
  await loadPoseJson();
  await loadPoseModel();
  await loadHandsModel();

  if (controls) {
    controls.enabled = true;
  }

  applyFeedbackDisplayMode();
  startFeedbackDrawLoop();

  state.poseTimer = setInterval(() => {
    inferPoseFrame();
  }, POSE_INTERVAL);

  state.handTimer = setInterval(() => {
    inferHands();
  }, HAND_INTERVAL);

  setStatus("READY");
}

async function startAvatarMode() {
  await destroyCurrentMode();

  clearFeedbackToasts();
  state.feedbackToastCooldowns = {};
  state.lastFeedbackSnapshot = "";

  state.currentMode = "avatar";
  state.stopped = false;
  state.avatarDisplay = "avatar";
  state.squatCount = 0;
  state.squatProgressSmooth = 0;
  state.squatState = "UP";
  state.lastPoseKpts = null;
  state.smoothPoseKpts = null;
  state.poseVelocity = null;
  state.poseLock.active = false;
  state.poseLock.stableFrames = 0;
  state.poseLock.baseCenter = null;
  state.handCurrentYaw = 0;
  state.handRotateState = "NEUTRAL";
  state.fistHoldCount = 0;
  state.gestureHandLabel = null;
  state.gestureHandLockFrames = 0;
  state.lastRenderedCount = 0;
  state.shallowHoldFrames = 0;
  state.shallowHoldActive = false;
  state.isInReadyPose = true;

  showAppScreen();
  resetViewVisibility();
  showCounter(true);
  updateCounterChip(true);

  dom.modeChip.textContent = "아바타 모드";
  dom.gestureChip.textContent = "GESTURE: -";
  setFeedbackText("LOADING...");

  dom.manualToggleBtn.style.display = "inline-flex";
  dom.cameraRotateToggle.checked = true;
  state.cameraRotateEnabled = true;

  await startWebcam();
  await startGuideVideo();
  dom.guideVideo.currentTime = 0;
  await loadPoseJson();
  await loadPoseModel();
  await loadHandsModel();

  initThree();
  if (controls) {
    controls.enabled = false;
  }

  const loader = new GLTFLoader();
  await loadGym(loader);
  await loadAnimatedAvatar(loader);

  applyAvatarDisplayMode();

  requestAnimationFrame(() => {
    resizeThreeRenderer();
  });

  startAvatarDrawLoop();

  state.poseTimer = setInterval(() => {
    inferPoseFrame();
  }, POSE_INTERVAL);

  state.handTimer = setInterval(() => {
    inferHands();
  }, HAND_INTERVAL);

  animateThree();
  setStatus("READY");
}

/* =========================
 * DESTROY / RESET
 * ========================= */
async function destroyCurrentMode() {
  state.stopped = true;
  clearTimers();

  if (renderRafId) {
    cancelAnimationFrame(renderRafId);
    renderRafId = 0;
  }

  if (
    state.jsonVideoFrameHandle != null &&
    typeof dom.guideVideo.cancelVideoFrameCallback === "function"
  ) {
    dom.guideVideo.cancelVideoFrameCallback(state.jsonVideoFrameHandle);
  }
  state.jsonVideoFrameHandle = null;

  stopWebcam();
  stopGuideVideo();

  try {
    dom.video.pause();
  } catch {}
  try {
    dom.guideVideo.pause();
  } catch {}

  dom.video.srcObject = null;
  dom.video.removeAttribute("src");
  dom.video.load?.();

  clearOverlay();
  clearCanvas();
  clearFeedbackToasts();

  state.lastPoseKpts = null;
  state.smoothPoseKpts = null;
  state.poseVelocity = null;
  state.poseLock.active = false;
  state.poseLock.stableFrames = 0;
  state.poseLock.baseCenter = null;
  state.lastHandResult = null;
  state.lastHandedness = null;
  state.poseBusy = false;
  state.handBusy = false;
  state.handRotateState = "NEUTRAL";
  state.fistHoldCount = 0;
  state.gestureHandLabel = null;
  state.gestureHandLockFrames = 0;
  state.lastRenderedCount = 0;
  state.shallowHoldFrames = 0;
  state.shallowHoldActive = false;
  state.isInReadyPose = true;
  state.feedbackToastCooldowns = {};
  state.lastFeedbackSnapshot = "";

  if (state.counterPopTimer) {
    clearTimeout(state.counterPopTimer);
    state.counterPopTimer = null;
  }

  showCounter(false);

  if (dom.counterChip) {
    dom.counterChip.textContent = "0";
    dom.counterChip.classList.remove("pop");
  }

  resetPartFeedback();
  setFeedbackText("READY");

  if (renderer) {
    renderer.dispose();
    dom.threeWrap.innerHTML = "";
  }

  renderer = null;
  scene = null;
  camera = null;
  controls = null;
  gymRoot = null;
  avatarScene = null;
  mixer = null;
  squatClip = null;
  squatAction = null;
}

/* =========================
 * EVENTS
 * ========================= */
dom.themeToggle.addEventListener("click", setTheme);

dom.backBtn.addEventListener("click", async () => {
  await destroyCurrentMode();
  state.currentMode = null;
  resetViewVisibility();
  setStatus("READY");
  setHelp("설정 대기 중");
  dom.gestureChip.textContent = "GESTURE: -";
  showModeScreen();
});

dom.modeCards.forEach((card) => {
  card.addEventListener("click", async () => {
    const mode = card.dataset.mode;
    if (mode === "standard") await startStandardMode();
    else if (mode === "feedback") await startFeedbackMode();
    else if (mode === "avatar") await startAvatarMode();
  });
});

dom.restartBtn.addEventListener("click", async () => {
  if (state.currentMode === "standard") await startStandardMode();
  else if (state.currentMode === "feedback") await startFeedbackMode();
  else if (state.currentMode === "avatar") await startAvatarMode();
});

dom.manualToggleBtn.addEventListener("click", () => {
  if (state.currentMode === "feedback") {
    state.feedbackDisplay = state.feedbackDisplay === "webcam" ? "guide" : "webcam";
    clearFeedbackToasts();
    applyFeedbackDisplayMode();
  } else if (state.currentMode === "avatar") {
    state.avatarDisplay = state.avatarDisplay === "avatar" ? "webcam" : "avatar";
    applyAvatarDisplayMode();
  }
});

dom.cameraRotateToggle.addEventListener("change", (e) => {
  state.cameraRotateEnabled = !!e.target.checked;
});

dom.guideVideo.addEventListener("dblclick", () => {
  if (state.currentMode === "standard") {
    toggleStandardDisplay();
  }
});

dom.threeWrap.addEventListener("dblclick", () => {
  if (state.currentMode === "standard") {
    toggleStandardDisplay();
  }
});

window.addEventListener("resize", () => {
  resizeThreeRenderer();
});

/* =========================
 * INIT UI
 * ========================= */
showModeScreen();
resetViewVisibility();
setStatus("READY");
setHelp("설정 대기 중");
dom.gestureChip.textContent = "GESTURE: -";
dom.manualToggleBtn.style.display = "none";
showCounter(false);
updateCounterChip(true);
clearFeedbackToasts();