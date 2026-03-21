/* eslint-disable no-console */
import * as THREE from "three";
import * as ort from "onnxruntime-web";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/** =========================
 * PATH
 * ========================= */
const ORT_DIST_BASE = "/ort/";
const INPUT_VIDEO_URL = "/squat.mp4";
const INPUT_POSE_JSON_URL = "/pose_squat.json";
const YOLO_MODEL_URL = "/models/yolo_pose.onnx";
const HAND_MODEL_URL = "/models/hand_landmarker.task";
const GYM_URL = "/models/Untitled_gym.glb";
const ANIM_AVATAR_URL = "/models/Untitled_squat.glb";

/** 조원 서버 기준 */
const API_BASE = "http://localhost:3000";
const STUDENT_ID = "2025730082";
const STUDENT_NAME = "Jeong In-yong";

ort.env.wasm.wasmPaths = ORT_DIST_BASE;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

/** =========================
 * DOM
 * ========================= */
const body = document.body;

const screens = {
  home: document.getElementById("home-screen"),
  mode: document.getElementById("mode-screen"),
  app: document.getElementById("app-screen"),
};

const themeToggleBtn = document.getElementById("themeToggle");
const themeIconEl = document.getElementById("theme-icon");
const themeTextEl = document.getElementById("theme-text");

const goModeBtn = document.getElementById("goModeBtn");
const showGuideBtn = document.getElementById("showGuideBtn");
const goHomeBtn = document.getElementById("goHomeBtn");
const backToModeBtn = document.getElementById("backToModeBtn");

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const canvasStage = document.getElementById("canvasStage");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const octx = overlay.getContext("2d");

const prepareBtn = document.getElementById("prepareBtn");
const runBtn = document.getElementById("runBtn");
const resetBtn = document.getElementById("resetBtn");
const resetBaselineBtn = document.getElementById("resetBaselineBtn");
const saveResultBtn = document.getElementById("saveResult");

const autoRotateEl = document.getElementById("autoRotate");
const handOverlayToggleEl = document.getElementById("handOverlayToggle");

const targetSetsInput = document.getElementById("targetSets");
const setRepTargetInput = document.getElementById("setRepTarget");
const restSecondsInput = document.getElementById("restSeconds");

const analysisStatusEl = document.getElementById("analysisStatus");
const saveStatusEl = document.getElementById("saveStatus");
const feedbackMainEl = document.getElementById("feedbackMain");
const feedbackDetailEl = document.getElementById("feedbackDetail");
const infoEl = document.getElementById("info");

const countValEl = document.getElementById("count-val");
const progressValEl = document.getElementById("progress-val");
const setValEl = document.getElementById("set-val");

const cardInputModeEl = document.getElementById("cardInputMode");
const cardModeEl = document.getElementById("cardMode");
const cardAvgDepthEl = document.getElementById("cardAvgDepth");
const cardSetEl = document.getElementById("cardSet");
const cardSetRepEl = document.getElementById("cardSetRep");
const cardRestEl = document.getElementById("cardRest");
const cardDepthLowEl = document.getElementById("cardDepthLow");
const cardTorsoWarnEl = document.getElementById("cardTorsoWarn");
const cardSpeedEl = document.getElementById("cardSpeed");
const cardRepTimeEl = document.getElementById("cardRepTime");

const modeCards = document.querySelectorAll(".mode-card");
const inputModeRadios = document.querySelectorAll('input[name="inputMode"]');
const workoutModeRadios = document.querySelectorAll('input[name="workoutMode"]');

const THREE_WRAP = document.getElementById("threeWrap");

/** =========================
 * STATE
 * ========================= */
let inputMode = "json";
let workoutMode = "normal";

let videoReady = false;
let poseReady = false;
let yoloReady = false;
let handReady = false;
let sceneReady = false;
let prepareBusy = false;

let poseData = null;
let poseFPS = 30;

let liveOn = false;
let liveTimer = null;
let videoTimer = null;
let webcamStream = null;

let lastPoseKpts = null;
let displayKpts = null;

let squatCount = 0;
let squatState = "UP";
let lastDepth = 0;
let lastTorsoLean = 0;
let lastFeedback = "대기 중";

let analyzedFrameCount = 0;
let depthAccum = 0;
let depthLowCount = 0;
let torsoWarningCount = 0;
let sessionSaved = false;
let sessionStartTime = 0;

let currentSet = 1;
let targetSets = 3;
let setRepTarget = 10;
let currentSetCount = 0;
let setResults = [];

let restSeconds = 30;
let restRemaining = 0;
let isResting = false;
let restTimer = null;
let wasLiveBeforeRest = false;

let repStartTime = 0;
let lastRepDuration = 0;
let speedFeedback = "대기";
let repTorsoWarnFlag = false;
let repMaxDepth = 0;

let baselineDepthOffset = 0;

/** =========================
 * YOLO
 * ========================= */
let yoloSession = null;
let yoloBusy = false;

const YOLO_INPUT_SIZE = 640;
const YOLO_SCORE_THRESHOLD = 0.25;
const YOLO_KPT_THRESHOLD = 0.20;
const YOLO_IOU_THRESHOLD = 0.45;

/** =========================
 * HANDS
 * ========================= */
let handLandmarker = null;
let handBusy = false;
let lastHandResult = null;

const HAND_MAX_NUM = 1;
const HAND_MIN_DETECTION_CONF = 0.55;
const HAND_MIN_PRESENCE_CONF = 0.55;
const HAND_MIN_TRACKING_CONF = 0.55;

/** =========================
 * pose stabilize
 * ========================= */
let lastStablePoseKpts = null;
let lastStablePoseTime = 0;
let lostPoseFrames = 0;

const POSE_HOLD_MS = 900;
const POSE_MAX_LOST_FRAMES = 18;
const POSE_MIN_VISIBLE_JOINTS = 8;
const POSE_CARRY_CONF_TH = 0.18;
const POSE_CENTER_JUMP_TH = 0.12;
const POSE_WIDTH_RATIO_MIN = 0.45;
const POSE_WIDTH_MIN_ABS = 0.035;

const UPPER_BLEND_ALPHA_GOOD = 0.18;
const UPPER_BLEND_ALPHA_WEAK = 0.10;
const LOWER_BLEND_ALPHA_GOOD = 0.58;
const LOWER_BLEND_ALPHA_WEAK = 0.42;

/** =========================
 * progress / sync
 * ========================= */
let squatProgressRaw = 0;
let squatProgressSmooth = 0;

const VIDEO_DRAW_INTERVAL = 33;
const SYNC_INTERVAL_MS = 33;
const PROGRESS_EMA = 0.35;

const DEPTH_UP = 0.18;
const DEPTH_DOWN = 0.52;
const DEPTH_LOW_THRESHOLD = 0.40;
const TORSO_WARN_THRESHOLD = 1.10;

/** =========================
 * Three.js
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
let lastMixerTime = -999;
let rafId = null;

const AVATAR_POS = new THREE.Vector3(0, 0, 0.8);
const LOOK_TARGET_OFFSET = new THREE.Vector3(0, 0.9, 0);
const FRONT_CAM_OFFSET = new THREE.Vector3(0, 0.55, 3.4);

let lastRenderTime = 0;
const RENDER_FPS = 20;
const RENDER_INTERVAL = 1000 / RENDER_FPS;

const CLIP_START_NORM = 0.0;
const CLIP_END_NORM = 0.33;

/** =========================
 * hidden infer canvas
 * ========================= */
const inferCanvas = document.createElement("canvas");
const inferCtx = inferCanvas.getContext("2d", { willReadFrequently: true });

/** =========================
 * skeleton
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

const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/** =========================
 * util
 * ========================= */
function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}
function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
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
function cloneKpts(kpts) {
  return kpts ? kpts.map((p) => [...p]) : null;
}
function getNowSec() {
  if (inputMode === "json" && videoReady) return video.currentTime || 0;
  return performance.now() / 1000;
}
function getInputModeLabel() {
  return inputMode === "json" ? "JSON" : "Webcam";
}
function getModeLabel() {
  return workoutMode === "normal" ? "Normal" : "Challenge";
}
function setAnalysisStatus(msg) {
  analysisStatusEl.textContent = msg;
}
function logInfo(msg) {
  infoEl.textContent = String(msg);
}
function countVisibleJoints(kpts, th = 0.05) {
  if (!kpts) return 0;
  let n = 0;
  for (const p of kpts) {
    if (p && (p[2] ?? 0) >= th) n += 1;
  }
  return n;
}
function getJointCenter(kpts, indices) {
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const idx of indices) {
    const p = kpts?.[idx];
    if (p && (p[2] ?? 0) > 0.05) {
      sx += p[0];
      sy += p[1];
      n += 1;
    }
  }
  if (!n) return null;
  return [sx / n, sy / n];
}
function getPoseCenter(kpts) {
  return getJointCenter(kpts, [5, 6, 11, 12, 13, 14]);
}
function getShoulderWidth(kpts) {
  const ls = kpts?.[5];
  const rs = kpts?.[6];
  if (!ls || !rs) return 0;
  if ((ls[2] ?? 0) < 0.05 || (rs[2] ?? 0) < 0.05) return 0;
  return Math.abs(rs[0] - ls[0]);
}
function getHipWidth(kpts) {
  const lh = kpts?.[11];
  const rh = kpts?.[12];
  if (!lh || !rh) return 0;
  if ((lh[2] ?? 0) < 0.05 || (rh[2] ?? 0) < 0.05) return 0;
  return Math.abs(rh[0] - lh[0]);
}
function mapDepthToProgress(depth) {
  const p = (depth - DEPTH_UP) / Math.max(1e-6, DEPTH_DOWN - DEPTH_UP);
  return clamp01(p);
}
function evaluateSpeed(durationSec) {
  if (durationSec < 0.8) return "너무 빠름";
  if (durationSec < 2.0) return "적정 속도";
  return "너무 느림";
}
function setRunButtonState() {
  runBtn.textContent = liveOn ? "종료" : "시작";
}
function isPreparedForCurrentMode() {
  if (inputMode === "json") return poseReady && sceneReady;
  return yoloReady && handReady && sceneReady;
}

/** =========================
 * theme / screen
 * ========================= */
function updateThemeButton() {
  const light = body.classList.contains("light-mode");
  themeIconEl.textContent = light ? "☀️" : "🌙";
  themeTextEl.textContent = light ? "LIGHT MODE" : "DARK MODE";
  if (scene) {
    scene.background = new THREE.Color(light ? 0xf4f7f6 : 0x0a0a0c);
  }
}
function showScreen(name) {
  Object.entries(screens).forEach(([key, el]) => {
    if (key === name) el.classList.remove("hidden");
    else el.classList.add("hidden");
  });
}
function goHome() {
  stopInput(true);
  showScreen("home");
}
function goMode() {
  stopInput(true);
  showScreen("mode");
}
function goAppWithMode(mode) {
  const radio = document.querySelector(`input[name="inputMode"][value="${mode}"]`);
  if (radio) radio.checked = true;
  readInputMode();
  showScreen("app");
  setAnalysisStatus(`${getInputModeLabel()} 모드 선택됨`);
  updateResultCards();
  resetHandCameraState(false);
}

/** =========================
 * form state
 * ========================= */
function readInputMode() {
  inputMode =
    document.querySelector('input[name="inputMode"]:checked')?.value || "json";
}
function readWorkoutInputs() {
  workoutMode =
    document.querySelector('input[name="workoutMode"]:checked')?.value || "normal";

  targetSets = Math.max(1, Number(targetSetsInput.value || 3));
  setRepTarget = Math.max(1, Number(setRepTargetInput.value || 10));
  restSeconds = Math.max(0, Number(restSecondsInput.value || 30));
}
function updateResultCards() {
  const avgDepth = analyzedFrameCount > 0 ? depthAccum / analyzedFrameCount : 0;

  cardInputModeEl.textContent = getInputModeLabel();
  cardModeEl.textContent = getModeLabel();
  cardAvgDepthEl.textContent = avgDepth.toFixed(3);
  cardSetEl.textContent = workoutMode === "normal" ? `${currentSet} / ${targetSets}` : "-";
  cardSetRepEl.textContent =
    workoutMode === "normal" ? `${currentSetCount} / ${setRepTarget}` : "-";
  cardRestEl.textContent = isResting ? `${restRemaining}s` : "-";
  cardDepthLowEl.textContent = String(depthLowCount);
  cardTorsoWarnEl.textContent = String(torsoWarningCount);
  cardSpeedEl.textContent = speedFeedback;
  cardRepTimeEl.textContent = `${lastRepDuration.toFixed(2)}s`;

  countValEl.textContent = String(squatCount);
  progressValEl.textContent = `${Math.round(squatProgressRaw * 100)}%`;
  setValEl.textContent =
    workoutMode === "normal" ? `${currentSet} / ${targetSets}` : "∞";

  updateFeedbackPanel();
}
function updateFeedbackPanel() {
  let mainText = "대기 중";
  let detailText = lastFeedback;
  let color = "var(--text-color)";

  if (isResting) {
    mainText = `휴식 중 (${restRemaining}s)`;
    detailText = "다음 세트를 위해 휴식 중입니다.";
    color = "var(--warn)";
  } else if (lastFeedback.includes("깊이 부족")) {
    mainText = "더 깊게 앉아주세요";
    detailText = "스쿼트 깊이가 기준보다 낮습니다.";
    color = "var(--bad)";
  } else if (lastFeedback.includes("상체 숙임")) {
    mainText = "상체를 조금 더 세워주세요";
    detailText = "상체 숙임이 크게 감지되었습니다.";
    color = "var(--warn)";
  } else if (squatState === "DOWN") {
    mainText = "좋아요, 올라오세요";
    detailText = "현재 하강 자세가 인식되었습니다.";
    color = "var(--good)";
  } else if (squatCount > 0) {
    mainText = "좋아요";
    detailText = `현재까지 ${squatCount}회 완료했습니다.`;
    color = "var(--good)";
  }

  feedbackMainEl.textContent = mainText;
  feedbackDetailEl.textContent = detailText || "분석 대기 중";
  feedbackMainEl.style.color = color;
}

/** =========================
 * overlay
 * ========================= */
function clearOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
}
function drawThresholdByIndex() {
  return 0.05;
}
function drawSkeleton2D(kpts, lineColor = "lime", dotColor = "red", lineWidth = 2, dotRadius = 3) {
  octx.lineWidth = lineWidth;
  octx.strokeStyle = lineColor;
  octx.fillStyle = dotColor;

  const w = overlay.width;
  const h = overlay.height;

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
function handLmOk(p) {
  return !!p && Number.isFinite(p.x) && Number.isFinite(p.y);
}
function drawHandOverlay(handResult) {
  if (!handOverlayToggleEl.checked) return;
  if (!handResult?.landmarks?.length) return;

  const lm = handResult.landmarks[0];
  const w = overlay.width;
  const h = overlay.height;

  octx.lineWidth = 1.5;
  octx.strokeStyle = "#3cf";
  octx.fillStyle = "#3cf";

  for (const [a, b] of HAND_EDGES) {
    const pa = lm[a];
    const pb = lm[b];
    if (!handLmOk(pa) || !handLmOk(pb)) continue;
    octx.beginPath();
    octx.moveTo(pa.x * w, pa.y * h);
    octx.lineTo(pb.x * w, pb.y * h);
    octx.stroke();
  }

  for (let i = 0; i < lm.length; i++) {
    const p = lm[i];
    if (!handLmOk(p)) continue;
    octx.beginPath();
    octx.arc(p.x * w, p.y * h, i === 0 ? 4 : 2.5, 0, Math.PI * 2);
    octx.fill();
  }
}
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

function normalizeAngleRad(a) {
  let x = a;
  while (x > Math.PI) x -= Math.PI * 2;
  while (x < -Math.PI) x += Math.PI * 2;
  return x;
}

function getHandRollRad(lm) {
  if (!lm || lm.length < 18) return 0;
  const indexMcp = lm[5];
  const pinkyMcp = lm[17];
  if (!handLmOk(indexMcp) || !handLmOk(pinkyMcp)) return 0;
  return Math.atan2(indexMcp.y - pinkyMcp.y, indexMcp.x - pinkyMcp.x);
}

function getFingerExtended(lm, tip, pip, mcp) {
  const a = lm[tip];
  const b = lm[pip];
  const c = lm[mcp];
  if (!handLmOk(a) || !handLmOk(b) || !handLmOk(c)) return false;
  return a.y < b.y && b.y < c.y;
}

function getThumbExtended(lm) {
  const tip = lm[4];
  const ip = lm[3];
  const mcp = lm[2];
  const wrist = lm[0];
  if (!handLmOk(tip) || !handLmOk(ip) || !handLmOk(mcp) || !handLmOk(wrist)) return false;

  const dxTip = Math.abs(tip.x - wrist.x);
  const dxIp = Math.abs(ip.x - wrist.x);
  const dxMcp = Math.abs(mcp.x - wrist.x);

  return dxTip > dxIp && dxIp > dxMcp * 0.85;
}

function isOpenPalmLandmarks(lm) {
  if (!lm || lm.length < 21) return false;

  const indexOpen = getFingerExtended(lm, 8, 6, 5);
  const middleOpen = getFingerExtended(lm, 12, 10, 9);
  const ringOpen = getFingerExtended(lm, 16, 14, 13);
  const pinkyOpen = getFingerExtended(lm, 20, 18, 17);
  const thumbOpen = getThumbExtended(lm);

  const openCount = [thumbOpen, indexOpen, middleOpen, ringOpen, pinkyOpen].filter(Boolean).length;
  return openCount >= 4;
}

function getHandFacingDirection(lm) {
  if (!lm || lm.length < 21) return "NEUTRAL";

  const wrist = lm[0];
  const indexMcp = lm[5];
  const pinkyMcp = lm[17];
  const middleTip = lm[12];
  const middleMcp = lm[9];

  if (
    !handLmOk(wrist) ||
    !handLmOk(indexMcp) ||
    !handLmOk(pinkyMcp) ||
    !handLmOk(middleTip) ||
    !handLmOk(middleMcp)
  ) {
    return "NEUTRAL";
  }

  // 손바닥 폭/방향
  const palmDx = pinkyMcp.x - indexMcp.x;
  const palmDy = pinkyMcp.y - indexMcp.y;

  // 손가락 방향
  const fingerDx = middleTip.x - middleMcp.x;
  const fingerDy = middleTip.y - middleMcp.y;

  // 2D cross product
  const cross = palmDx * fingerDy - palmDy * fingerDx;

  // 너무 애매하면 중립
  if (Math.abs(cross) < 0.008) {
    return "NEUTRAL";
  }

  // 웹캠 좌우반전/카메라 방향 기준
  // 현재 사용자 테스트 기준:
  // 손등 보이게 꺾음 -> LEFT
  // 손바닥 보이게 꺾음 -> RIGHT
  return cross > 0 ? "RIGHT" : "LEFT";
}

function updateHandRotateState() {
  const lm = lastHandResult?.landmarks?.[0] || null;

  if (!lm) {
    handRotateState = "NEUTRAL";
    lastOpenPalm = false;
    return;
  }

  lastOpenPalm = isOpenPalmLandmarks(lm);

  // 손이 충분히 보이고 펴져 있을 때만 회전 허용
  if (!lastOpenPalm) {
    handRotateState = "NEUTRAL";
    return;
  }

  handRotateState = getHandFacingDirection(lm);
}

function resetHandCameraState(resetYaw = false) {
  handRotateState = "NEUTRAL";
  lastOpenPalm = false;
  if (resetYaw) handCurrentYaw = 0;
}

function getBaseCameraPositionByYaw(yawRad) {
  const target = getAvatarLookTarget();
  const r = Math.sqrt(FRONT_CAM_OFFSET.x * FRONT_CAM_OFFSET.x + FRONT_CAM_OFFSET.z * FRONT_CAM_OFFSET.z);
  const x = target.x + Math.sin(yawRad) * r;
  const z = target.z + Math.cos(yawRad) * r;
  const y = target.y + FRONT_CAM_OFFSET.y;
  return { x, y, z, target };
}

function applyHandCameraControl() {
  if (!camera || !avatarScene) return;
  if (inputMode !== "webcam") return;
  if (!handReady || !videoReady) return;
  if (isResting) return;

  updateHandRotateState();

  if (handRotateState === "LEFT") {
    handCurrentYaw -= HAND_ROTATE_SPEED;
  } else if (handRotateState === "RIGHT") {
    handCurrentYaw += HAND_ROTATE_SPEED;
  }

  handCurrentYaw = clamp(handCurrentYaw, -HAND_YAW_LIMIT, HAND_YAW_LIMIT);

  const camPos = getBaseCameraPositionByYaw(handCurrentYaw);
  camera.position.set(camPos.x, camPos.y, camPos.z);
  camera.lookAt(camPos.target);
}

function renderOverlayFrame() {
  clearOverlay();

  if (lastPoseKpts) {
    const smoothed = smoothDisplayKpts(lastPoseKpts);
    if (smoothed) drawSkeleton2D(smoothed);
  }

  if (inputMode === "webcam" && handReady && lastHandResult?.landmarks?.length) {
    drawHandOverlay(lastHandResult);
  }
}

function resetOverlayState() {
  lastPoseKpts = null;
  displayKpts = null;
  lastHandResult = null;
  clearOverlay();

  lastStablePoseKpts = null;
  lastStablePoseTime = 0;
  lostPoseFrames = 0;
}

/** =========================
 * pose stabilize
 * ========================= */
function blendJoint(a, b, alpha) {
  if (!a && !b) return [0, 0, 0];
  if (!a) return [...b];
  if (!b) return [...a];
  return [
    lerp(a[0], b[0], alpha),
    lerp(a[1], b[1], alpha),
    Math.max(a[2] ?? 0, b[2] ?? 0),
  ];
}
function blendPoseUpperLower(prevKpts, currKpts, upperAlpha, lowerAlpha) {
  if (!prevKpts && !currKpts) return null;
  if (!prevKpts) return cloneKpts(currKpts);
  if (!currKpts) return cloneKpts(prevKpts);

  const out = [];
  const upperSet = new Set([0, 5, 6, 7, 8, 9, 10]);
  const lowerSet = new Set([11, 12, 13, 14, 15, 16]);

  for (let i = 0; i < Math.max(prevKpts.length, currKpts.length); i++) {
    const prev = prevKpts[i];
    const curr = currKpts[i];

    if (upperSet.has(i)) out.push(blendJoint(prev, curr, upperAlpha));
    else if (lowerSet.has(i)) out.push(blendJoint(prev, curr, lowerAlpha));
    else out.push(blendJoint(prev, curr, (upperAlpha + lowerAlpha) * 0.5));
  }
  return out;
}
function carryForwardMissingJoints(currKpts, prevKpts, confTh = 0.18) {
  if (!currKpts && !prevKpts) return null;
  if (!currKpts) return prevKpts ? prevKpts.map((p) => [...p]) : null;
  if (!prevKpts) return currKpts.map((p) => [...p]);

  return currKpts.map((p, i) => {
    const cur = p ? [...p] : null;
    const prev = prevKpts[i] ? [...prevKpts[i]] : null;

    if (!cur && prev) return prev;
    if (!cur) return [0, 0, 0];
    if ((cur[2] ?? 0) < confTh && prev) {
      return [prev[0], prev[1], Math.max(cur[2] ?? 0, prev[2] ?? 0.4)];
    }
    return cur;
  });
}
function normalizeLeftRightPairs(currKpts, prevKpts) {
  if (!currKpts) return null;
  const out = cloneKpts(currKpts);

  const LEFT_CHAIN = [5, 7, 9, 11, 13, 15];
  const RIGHT_CHAIN = [6, 8, 10, 12, 14, 16];

  function swapWholeSide(kpts) {
    for (let i = 0; i < LEFT_CHAIN.length; i++) {
      const li = LEFT_CHAIN[i];
      const ri = RIGHT_CHAIN[i];
      const tmp = kpts[li];
      kpts[li] = kpts[ri];
      kpts[ri] = tmp;
    }
  }

  const ls = out[5];
  const rs = out[6];
  const lh = out[11];
  const rh = out[12];

  const shoulderOk = ls && rs && (ls[2] ?? 0) >= 0.05 && (rs[2] ?? 0) >= 0.05;
  const hipOk = lh && rh && (lh[2] ?? 0) >= 0.05 && (rh[2] ?? 0) >= 0.05;

  if (!prevKpts) {
    if (shoulderOk && hipOk) {
      if (ls[0] > rs[0] || lh[0] > rh[0]) swapWholeSide(out);
    } else if (shoulderOk) {
      if (ls[0] > rs[0]) swapWholeSide(out);
    } else if (hipOk) {
      if (lh[0] > rh[0]) swapWholeSide(out);
    }
    return out;
  }

  let keepCost = 0;
  let swapCost = 0;

  for (let i = 0; i < LEFT_CHAIN.length; i++) {
    const li = LEFT_CHAIN[i];
    const ri = RIGHT_CHAIN[i];

    const curL = out[li];
    const curR = out[ri];
    const prevL = prevKpts[li];
    const prevR = prevKpts[ri];

    if (!curL || !curR || !prevL || !prevR) continue;

    keepCost +=
      Math.abs(curL[0] - prevL[0]) +
      Math.abs(curL[1] - prevL[1]) +
      Math.abs(curR[0] - prevR[0]) +
      Math.abs(curR[1] - prevR[1]);

    swapCost +=
      Math.abs(curR[0] - prevL[0]) +
      Math.abs(curR[1] - prevL[1]) +
      Math.abs(curL[0] - prevR[0]) +
      Math.abs(curL[1] - prevR[1]);
  }

  if (swapCost < keepCost) swapWholeSide(out);
  return out;
}
function fixTorsoCross(kpts, prevKpts = null) {
  if (!kpts) return null;
  const out = cloneKpts(kpts);

  const ls = out[5];
  const rs = out[6];
  const lh = out[11];
  const rh = out[12];

  const shoulderOk = ls && rs && (ls[2] ?? 0) >= 0.05 && (rs[2] ?? 0) >= 0.05;
  const hipOk = lh && rh && (lh[2] ?? 0) >= 0.05 && (rh[2] ?? 0) >= 0.05;
  if (!(shoulderOk && hipOk)) return out;

  const shoulderOrderOk = ls[0] <= rs[0];
  const hipOrderOk = lh[0] <= rh[0];
  if (shoulderOrderOk === hipOrderOk) return out;

  const LEFT_CHAIN = [5, 7, 9, 11, 13, 15];
  const RIGHT_CHAIN = [6, 8, 10, 12, 14, 16];
  const swapped = cloneKpts(out);

  for (let i = 0; i < LEFT_CHAIN.length; i++) {
    const li = LEFT_CHAIN[i];
    const ri = RIGHT_CHAIN[i];
    const tmp = swapped[li];
    swapped[li] = swapped[ri];
    swapped[ri] = tmp;
  }

  if (!prevKpts) return swapped;

  let keepCost = 0;
  let swapCost = 0;
  for (let i = 0; i < LEFT_CHAIN.length; i++) {
    const li = LEFT_CHAIN[i];
    const ri = RIGHT_CHAIN[i];
    const aL = out[li];
    const aR = out[ri];
    const bL = swapped[li];
    const bR = swapped[ri];
    const pL = prevKpts[li];
    const pR = prevKpts[ri];

    if (!aL || !aR || !bL || !bR || !pL || !pR) continue;

    keepCost +=
      Math.abs(aL[0] - pL[0]) + Math.abs(aL[1] - pL[1]) +
      Math.abs(aR[0] - pR[0]) + Math.abs(aR[1] - pR[1]);

    swapCost +=
      Math.abs(bL[0] - pL[0]) + Math.abs(bL[1] - pL[1]) +
      Math.abs(bR[0] - pR[0]) + Math.abs(bR[1] - pR[1]);
  }

  return swapCost < keepCost ? swapped : out;
}
function isPosePlausible(currKpts, prevKpts) {
  if (!currKpts) return false;

  const sw = getShoulderWidth(currKpts);
  const hw = getHipWidth(currKpts);

  if (sw > 0 && sw < POSE_WIDTH_MIN_ABS && hw > 0 && hw < POSE_WIDTH_MIN_ABS) {
    return false;
  }

  if (!prevKpts) return true;

  const prevCenter = getPoseCenter(prevKpts);
  const currCenter = getPoseCenter(currKpts);

  if (prevCenter && currCenter) {
    const jump = Math.hypot(currCenter[0] - prevCenter[0], currCenter[1] - prevCenter[1]);
    if (jump > POSE_CENTER_JUMP_TH) return false;
  }

  const prevSW = getShoulderWidth(prevKpts);
  const prevHW = getHipWidth(prevKpts);

  if (prevSW > 0.05 && sw > 0 && sw / prevSW < POSE_WIDTH_RATIO_MIN) return false;
  if (prevHW > 0.05 && hw > 0 && hw / prevHW < POSE_WIDTH_RATIO_MIN) return false;

  return true;
}
function stabilizeWebcamPose(currKpts, nowSec) {
  if (!currKpts) {
    lostPoseFrames += 1;

    const withinHoldTime = (nowSec - lastStablePoseTime) * 1000 <= POSE_HOLD_MS;
    const withinHoldFrames = lostPoseFrames <= POSE_MAX_LOST_FRAMES;

    if (lastStablePoseKpts && withinHoldTime && withinHoldFrames) {
      return cloneKpts(lastStablePoseKpts);
    }
    return null;
  }

  let normalized = normalizeLeftRightPairs(currKpts, lastStablePoseKpts);
  normalized = fixTorsoCross(normalized, lastStablePoseKpts);

  let repaired = carryForwardMissingJoints(normalized, lastStablePoseKpts, POSE_CARRY_CONF_TH);
  repaired = fixTorsoCross(repaired, lastStablePoseKpts);

  const visibleNow = countVisibleJoints(repaired, 0.05);
  const hasEnoughNow = visibleNow >= POSE_MIN_VISIBLE_JOINTS;
  const plausible = isPosePlausible(repaired, lastStablePoseKpts);

  if (hasEnoughNow && plausible) {
    const upperAlpha = visibleNow >= 12 ? UPPER_BLEND_ALPHA_GOOD : UPPER_BLEND_ALPHA_WEAK;
    const lowerAlpha = visibleNow >= 12 ? LOWER_BLEND_ALPHA_GOOD : LOWER_BLEND_ALPHA_WEAK;
    repaired = blendPoseUpperLower(lastStablePoseKpts, repaired, upperAlpha, lowerAlpha);

    lastStablePoseKpts = cloneKpts(repaired);
    lastStablePoseTime = nowSec;
    lostPoseFrames = 0;
    return repaired;
  }

  lostPoseFrames += 1;
  const withinHoldTime = (nowSec - lastStablePoseTime) * 1000 <= POSE_HOLD_MS;
  const withinHoldFrames = lostPoseFrames <= POSE_MAX_LOST_FRAMES;

  if (lastStablePoseKpts && withinHoldTime && withinHoldFrames) {
    return cloneKpts(lastStablePoseKpts);
  }

  return repaired;
}

/** =========================
 * input
 * ========================= */
function resizeStageToVideo(videoWidth, videoHeight) {
  const w = Math.max(1, videoWidth);
  const h = Math.max(1, videoHeight);

  canvas.width = w;
  canvas.height = h;
  overlay.width = w;
  overlay.height = h;

  canvas.style.width = "100%";
  canvas.style.height = "100%";
  overlay.style.width = "100%";
  overlay.style.height = "100%";
  video.style.width = "100%";
  video.style.height = "100%";
  canvasStage.style.width = "100%";
  canvasStage.style.height = "100%";
}
function drawVideoFrameToCanvas() {
  if (!videoReady || isResting) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}
async function startJsonVideoInput() {
  stopWebcamStream();

  video.srcObject = null;
  video.src = INPUT_VIDEO_URL;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = "auto";

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

  resizeStageToVideo(video.videoWidth || 720, video.videoHeight || 1280);
  video.currentTime = 0;
  await video.play();

  videoReady = true;
  setAnalysisStatus("JSON 영상 입력 준비 완료");
}
async function startWebcamInput() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("브라우저가 getUserMedia를 지원하지 않음");
  }

  stopWebcamStream();

  webcamStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
    audio: false,
  });

  video.src = "";
  video.srcObject = webcamStream;
  video.muted = true;
  video.playsInline = true;
  video.loop = false;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });

  resizeStageToVideo(video.videoWidth || 640, video.videoHeight || 480);
  await video.play();

  videoReady = true;
  setAnalysisStatus("웹캠 입력 준비 완료");
}
function stopWebcamStream() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
}
async function startInput() {
  readInputMode();
  readWorkoutInputs();
  resetHandCameraState();
  if (inputMode === "json") await startJsonVideoInput();
  else await startWebcamInput();

  resetSessionStats();

  if (videoTimer) clearInterval(videoTimer);
  videoTimer = setInterval(() => {
    drawVideoFrameToCanvas();
    renderOverlayFrame();
  }, VIDEO_DRAW_INTERVAL);

  drawVideoFrameToCanvas();
}
function stopInput(resetTime = true) {
  setLive(false);

  if (videoTimer) {
    clearInterval(videoTimer);
    videoTimer = null;
  }

  stopRestTimer();

  const shouldSave = !sessionSaved && (squatCount > 0 || analyzedFrameCount > 0);

  if (shouldSave) {
    saveSessionResult().catch(console.error);
  }

  try {
    if (!video.paused) video.pause();
  } catch {}

  stopWebcamStream();

  if (resetTime) {
    try {
      video.currentTime = 0;
    } catch {}
  }

  video.srcObject = null;
  videoReady = false;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resetOverlayState();
  resetAvatarPoseToStart();

  if (runBtn) setRunButtonState();
}

/** =========================
 * json pose
 * ========================= */
async function loadPoseJson() {
  const res = await fetch(INPUT_POSE_JSON_URL);
  if (!res.ok) throw new Error(`pose json load failed: ${res.status}`);

  poseData = await res.json();
  poseFPS = poseData.fps || 30;
  poseReady = Array.isArray(poseData.frames);

  logInfo(`pose json loaded\nfps=${poseFPS}\nframes=${poseData.frames?.length ?? 0}`);
}
function getPoseFrameAtTimeSec(timeSec) {
  if (!poseReady || !poseData?.frames?.length) return null;

  const frames = poseData.frames;
  const idx = Math.min(frames.length - 1, Math.max(0, Math.floor(timeSec * poseFPS)));
  return frames[idx] || null;
}
function extractKeypointsFromFrame(frame) {
  if (!frame || !frame.valid || !frame.keypoints) return null;
  return frame.keypoints;
}

/** =========================
 * yolo
 * ========================= */
async function loadYoloModel() {
  yoloSession = await ort.InferenceSession.create(YOLO_MODEL_URL, {
    executionProviders: ["wasm"],
  });
  yoloReady = true;

  logInfo(
    "YOLO model loaded\n" +
      `inputs=${JSON.stringify(yoloSession.inputNames)}\n` +
      `outputs=${JSON.stringify(yoloSession.outputNames)}`
  );
}
function iouXYXY(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const inter = interW * interH;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / Math.max(1e-6, areaA + areaB - inter);
}
function nmsBoxes(boxes, iouThreshold = 0.45) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep = [];

  while (sorted.length) {
    const cur = sorted.shift();
    keep.push(cur);

    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iouXYXY(cur, sorted[i]) > iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return keep;
}
function preprocessForYolo(srcVideo) {
  const srcW = srcVideo.videoWidth || canvas.width || 640;
  const srcH = srcVideo.videoHeight || canvas.height || 640;

  inferCanvas.width = YOLO_INPUT_SIZE;
  inferCanvas.height = YOLO_INPUT_SIZE;

  inferCtx.fillStyle = "black";
  inferCtx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);

  const scale = Math.min(YOLO_INPUT_SIZE / srcW, YOLO_INPUT_SIZE / srcH);
  const drawW = Math.round(srcW * scale);
  const drawH = Math.round(srcH * scale);
  const padX = Math.floor((YOLO_INPUT_SIZE - drawW) / 2);
  const padY = Math.floor((YOLO_INPUT_SIZE - drawH) / 2);

  inferCtx.drawImage(srcVideo, 0, 0, srcW, srcH, padX, padY, drawW, drawH);

  const imageData = inferCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  const { data } = imageData;

  const chw = new Float32Array(1 * 3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
  const area = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;

  for (let i = 0; i < area; i++) {
    const r = data[i * 4 + 0] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    chw[i] = r;
    chw[area + i] = g;
    chw[area * 2 + i] = b;
  }

  return {
    tensor: new ort.Tensor("float32", chw, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]),
    meta: { srcW, srcH, scale, padX, padY },
  };
}
function decodeYoloPoseOutput(outputTensor, meta) {
  const dims = outputTensor.dims;
  const arr = outputTensor.data;

  let rows = [];
  let featureSize = 0;
  let count = 0;

  if (dims.length !== 3) {
    throw new Error(`지원하지 않는 output dims: ${JSON.stringify(dims)}`);
  }

  if (dims[1] === 56) {
    featureSize = dims[1];
    count = dims[2];
    for (let i = 0; i < count; i++) {
      const row = new Float32Array(featureSize);
      for (let j = 0; j < featureSize; j++) {
        row[j] = arr[j * count + i];
      }
      rows.push(row);
    }
  } else if (dims[2] === 56) {
    count = dims[1];
    featureSize = dims[2];
    for (let i = 0; i < count; i++) {
      const start = i * featureSize;
      rows.push(arr.slice(start, start + featureSize));
    }
  } else {
    throw new Error(`56 feature 출력이 아님: ${JSON.stringify(dims)}`);
  }

  const candidates = [];

  for (const row of rows) {
    const cx = row[0];
    const cy = row[1];
    const w = row[2];
    const h = row[3];
    const score = row[4];
    if (score < YOLO_SCORE_THRESHOLD) continue;

    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;

    const kpts = [];
    for (let k = 0; k < 17; k++) {
      const base = 5 + k * 3;
      const kx = row[base + 0];
      const ky = row[base + 1];
      const ks = row[base + 2];
      kpts.push([kx, ky, ks]);
    }

    candidates.push({ x1, y1, x2, y2, score, kpts });
  }

  if (!candidates.length) return null;

  const kept = nmsBoxes(candidates, YOLO_IOU_THRESHOLD);
  if (!kept.length) return null;

  const { srcW, srcH, scale, padX, padY } = meta;

  const mappedCandidates = kept.map((cand) => {
    const mapped = cand.kpts.map(([x, y, s]) => {
      const rx = (x - padX) / Math.max(1e-6, scale);
      const ry = (y - padY) / Math.max(1e-6, scale);

      return [
        clamp01(rx / Math.max(1, srcW)),
        clamp01(ry / Math.max(1, srcH)),
        s >= YOLO_KPT_THRESHOLD ? s : 0,
      ];
    });

    return {
      score: cand.score,
      keypoints: mapped,
    };
  });

  return { candidates: mappedCandidates };
}
function selectBestCandidate(candidates, prevKpts) {
  if (!candidates?.length) return null;
  if (!prevKpts) return candidates[0];

  let best = null;
  let bestCost = Infinity;

  const prevCenter = getPoseCenter(prevKpts);
  const prevSW = getShoulderWidth(prevKpts);
  const prevHW = getHipWidth(prevKpts);

  for (const cand of candidates) {
    const kpts = normalizeLeftRightPairs(cand.keypoints, prevKpts);
    const currCenter = getPoseCenter(kpts);

    let centerCost = 0;
    if (prevCenter && currCenter) {
      centerCost = Math.hypot(currCenter[0] - prevCenter[0], currCenter[1] - prevCenter[1]);
    }

    const sw = getShoulderWidth(kpts);
    const hw = getHipWidth(kpts);
    const widthCost = Math.abs(sw - prevSW) * 1.5 + Math.abs(hw - prevHW) * 1.5;
    const visiblePenalty = Math.max(0, 10 - countVisibleJoints(kpts, 0.05)) * 0.03;
    const scoreBonus = (1 - cand.score) * 0.08;
    const totalCost = centerCost * 2.5 + widthCost + visiblePenalty + scoreBonus;

    if (totalCost < bestCost) {
      bestCost = totalCost;
      best = { score: cand.score, keypoints: kpts };
    }
  }

  return best;
}
async function inferYoloFromCurrentFrame() {
  if (!yoloReady || !yoloSession || !videoReady) return null;
  if (yoloBusy) return null;

  yoloBusy = true;
  try {
    const { tensor, meta } = preprocessForYolo(video);
    const inputName = yoloSession.inputNames[0];
    const outputs = await yoloSession.run({ [inputName]: tensor });
    const outputName = yoloSession.outputNames[0];
    const outputTensor = outputs[outputName];
    if (!outputTensor) throw new Error("YOLO output 없음");

    const decoded = decodeYoloPoseOutput(outputTensor, meta);
    if (!decoded?.candidates?.length) return null;

    return selectBestCandidate(decoded.candidates, lastStablePoseKpts);
  } finally {
    yoloBusy = false;
  }
}

/** =========================
 * hands
 * ========================= */
async function loadHandsModel() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
    },
    runningMode: "VIDEO",
    numHands: HAND_MAX_NUM,
    minHandDetectionConfidence: HAND_MIN_DETECTION_CONF,
    minHandPresenceConfidence: HAND_MIN_PRESENCE_CONF,
    minTrackingConfidence: HAND_MIN_TRACKING_CONF,
  });

  handReady = true;
  logInfo("Hands model loaded");
}
async function inferHandsFromCurrentFrame() {
  if (!handReady || !handLandmarker || !videoReady || inputMode !== "webcam") return null;
  if (handBusy) return lastHandResult;

  handBusy = true;
  try {
    const ts = performance.now();
    const result = handLandmarker.detectForVideo(video, ts);
    lastHandResult = result || null;
    lastOpenPalm = !!(result?.landmarks?.[0] && isOpenPalmLandmarks(result.landmarks[0]));
    return lastHandResult;
  } finally {
    handBusy = false;
  }
}

/** =========================
 * hand camera control
 * ========================= */
let handCurrentYaw = 0;
let handRotateState = "NEUTRAL"; // LEFT / RIGHT / NEUTRAL
let lastOpenPalm = false;

const HAND_YAW_LIMIT = Math.PI * 0.95;
const HAND_ROTATE_SPEED = 0.03; // 값 키우면 더 빨라짐

/** =========================
 * squat analysis
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
  let depth = clamp01(frontDepth * frontness + sideDepth * (1 - frontness));
  depth = clamp01(depth - baselineDepthOffset);

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
function finishCurrentSet() {
  const avgDepth = analyzedFrameCount > 0 ? depthAccum / analyzedFrameCount : 0;

  setResults.push({
    setNumber: currentSet,
    reps: currentSetCount,
    avgDepth,
    depthLowCount,
    torsoWarningCount,
  });

  if (currentSet >= targetSets) {
    stopRestTimer();
    setAnalysisStatus("모든 세트 완료");
    setLive(false);
    updateResultCards();
    return;
  }

  currentSet += 1;
  currentSetCount = 0;
  startRestTimer();
  updateResultCards();
}
function updateSquatCounterAndFeedback(kpts) {
  const m = computeSquatMetrics(kpts);
  if (!m) {
    lastFeedback = "관절 인식 부족";
    return;
  }

  lastDepth = m.depth;
  lastTorsoLean = m.torsoLean;

  analyzedFrameCount += 1;
  depthAccum += m.depth;

  const isTorsoWarnNow = m.torsoLean > TORSO_WARN_THRESHOLD;

  if (squatState === "UP" && m.depth >= 0.58) {
    squatState = "DOWN";
    repTorsoWarnFlag = false;
    repMaxDepth = m.depth;
    repStartTime = getNowSec();
  }

  if (squatState === "DOWN") {
    repMaxDepth = Math.max(repMaxDepth, m.depth);
    if (isTorsoWarnNow) repTorsoWarnFlag = true;
  }

  const currentDepthLowDisplay =
    squatState === "DOWN" ? repMaxDepth < DEPTH_LOW_THRESHOLD : false;

  if (squatState === "DOWN" && m.depth <= 0.22) {
    squatState = "UP";
    squatCount += 1;

    if (workoutMode === "normal") {
      currentSetCount += 1;
    }

    if (repMaxDepth < DEPTH_LOW_THRESHOLD) depthLowCount += 1;
    if (repTorsoWarnFlag) torsoWarningCount += 1;

    lastRepDuration = Math.max(0, getNowSec() - repStartTime);
    speedFeedback = evaluateSpeed(lastRepDuration);

    if (workoutMode === "normal" && currentSetCount >= setRepTarget) {
      finishCurrentSet();
    }

    repTorsoWarnFlag = false;
    repMaxDepth = 0;
  }

  const feedbacks = [];
  if (isResting) {
    feedbacks.push(`휴식 중 ${restRemaining}초`);
  } else if (squatState === "DOWN") {
    feedbacks.push(currentDepthLowDisplay ? "깊이 부족" : "깊이 양호");
    feedbacks.push(isTorsoWarnNow ? "상체 숙임 큼" : "상체 각도 양호");
  } else {
    feedbacks.push("준비 자세");
    feedbacks.push(isTorsoWarnNow ? "상체 약간 숙임" : "상체 각도 양호");
  }

  if (speedFeedback && speedFeedback !== "대기") feedbacks.push(speedFeedback);
  lastFeedback = feedbacks.join(" / ");

  squatProgressRaw = mapDepthToProgress(lastDepth);
  squatProgressSmooth = lerp(squatProgressSmooth, squatProgressRaw, PROGRESS_EMA);

  updateResultCards();
}

/** =========================
 * rest
 * ========================= */
function stopRestTimer() {
  if (restTimer) {
    clearInterval(restTimer);
    restTimer = null;
  }
  isResting = false;
  restRemaining = 0;
}
function pauseInputForRest() {
  wasLiveBeforeRest = liveOn;
  setLive(false);

  if (inputMode === "json" && videoReady) {
    try {
      video.pause();
    } catch {}
  }

  setAnalysisStatus(`휴식 중... ${restRemaining}초`);
}
async function resumeInputAfterRest() {
  if (inputMode === "json" && videoReady) {
    try {
      await video.play();
    } catch {}
  }

  if (wasLiveBeforeRest) setLive(true);
  wasLiveBeforeRest = false;
  setAnalysisStatus(`${currentSet}세트 시작`);
}
function startRestTimer() {
  stopRestTimer();
  isResting = true;
  restRemaining = restSeconds;

  pauseInputForRest();
  updateResultCards();

  restTimer = setInterval(async () => {
    restRemaining -= 1;
    updateResultCards();

    if (restRemaining > 0) {
      setAnalysisStatus(`휴식 중... ${restRemaining}초`);
      return;
    }

    stopRestTimer();
    await resumeInputAfterRest();
    updateResultCards();
  }, 1000);
}

/** =========================
 * session reset
 * ========================= */
function resetSessionStats() {
  squatCount = 0;
  squatState = "UP";
  lastDepth = 0;
  lastTorsoLean = 0;
  lastFeedback = "대기 중";

  analyzedFrameCount = 0;
  depthAccum = 0;
  depthLowCount = 0;
  torsoWarningCount = 0;
  sessionSaved = false;
  sessionStartTime = getNowSec();

  currentSet = 1;
  currentSetCount = 0;
  setResults = [];

  repTorsoWarnFlag = false;
  repMaxDepth = 0;

  repStartTime = 0;
  lastRepDuration = 0;
  speedFeedback = "대기";

  squatProgressRaw = 0;
  squatProgressSmooth = 0;
  stopRestTimer();

  updateResultCards();
  saveStatusEl.textContent = "저장 대기";
}

/** =========================
 * server save
 * ========================= */
async function saveSessionResult() {
  try {
    saveStatusEl.textContent = "저장 중...";

    const payload = {
      studentId: STUDENT_ID,
      name: STUDENT_NAME,
      count: squatCount,
      avgDepth: analyzedFrameCount > 0 ? depthAccum / analyzedFrameCount : 0,
      depthLowCount,
      torsoWarningCount,
      mode: workoutMode,
      inputMode,
      setResults,
      kst_time: new Intl.DateTimeFormat("ko-KR", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
        timeZone: "Asia/Seoul",
      })
        .format(new Date())
        .replace(/\. /g, "-")
        .replace(".", ""),
    };

    const res = await fetch(`${API_BASE}/api/workout`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`save failed: ${res.status}`);

    sessionSaved = true;
    saveStatusEl.textContent = "저장 완료";
    setAnalysisStatus("DB 저장 완료");
  } catch (err) {
    console.error(err);
    saveStatusEl.textContent = "저장 실패";
    logInfo(`Save error: ${err.message}`);
  }
}

/** =========================
 * sync
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

  setAnalysisStatus(`JSON 분석 중 / count ${squatCount} / ${lastFeedback}`);
  logInfo(
    `inputMode=${inputMode}
mode=${workoutMode}
videoTime=${video.currentTime.toFixed(2)}s
poseFPS=${poseFPS}

squatCount=${squatCount}
currentSet=${currentSet}
currentSetCount=${currentSetCount}
isResting=${isResting}
restRemaining=${restRemaining}

depth=${lastDepth.toFixed(3)}
progressRaw=${squatProgressRaw.toFixed(3)}
torsoLean=${lastTorsoLean.toFixed(3)}
speed=${speedFeedback}
repTime=${lastRepDuration.toFixed(2)}s
feedback=${lastFeedback}`
  );
}
async function inferPoseFromWebcam() {
  if (!videoReady || !yoloReady) return;

  `handOpen=${lastOpenPalm}\n` +
 `handRotateState=${handRotateState}\n` +
 `handYaw=${handCurrentYaw.toFixed(3)}\n`
  await inferHandsFromCurrentFrame();

  const result = await inferYoloFromCurrentFrame();
  const nowSec = getNowSec();

  if (!result?.keypoints) {
    lostPoseFrames += 1;

    const withinHoldTime = (nowSec - lastStablePoseTime) * 1000 <= POSE_HOLD_MS;
    const withinHoldFrames = lostPoseFrames <= POSE_MAX_LOST_FRAMES;

    if (lastStablePoseKpts && withinHoldTime && withinHoldFrames) {
      lastPoseKpts = cloneKpts(lastStablePoseKpts);
      updateSquatCounterAndFeedback(lastPoseKpts);
      lastFeedback = "이전 포즈 유지 중";
    } else {
      lastFeedback = "YOLO 인식 없음";
    }
    return;
  }

  const stabilizedKpts = stabilizeWebcamPose(result.keypoints, nowSec);

  if (!stabilizedKpts || countVisibleJoints(stabilizedKpts, 0.05) < 6) {
    lastFeedback = "YOLO 관절 부족";
    return;
  }

  lastPoseKpts = fixTorsoCross(stabilizedKpts, lastStablePoseKpts);
  updateSquatCounterAndFeedback(lastPoseKpts);

  setAnalysisStatus(`YOLO 분석 중 / count ${squatCount} / ${lastFeedback}`);
  logInfo(
    `inputMode=${inputMode}
mode=${workoutMode}
yoloScore=${result.score?.toFixed?.(3) ?? "n/a"}
visible=${countVisibleJoints(stabilizedKpts, 0.05)}

squatCount=${squatCount}
currentSet=${currentSet}
currentSetCount=${currentSetCount}
isResting=${isResting}
restRemaining=${restRemaining}

depth=${lastDepth.toFixed(3)}
progressRaw=${squatProgressRaw.toFixed(3)}
torsoLean=${lastTorsoLean.toFixed(3)}
speed=${speedFeedback}
repTime=${lastRepDuration.toFixed(2)}s
feedback=${lastFeedback}`
  );
}
async function syncOrInferOneFrame() {
  if (isResting) return;

  if (inputMode === "json") syncPoseFromVideoTime();
  else await inferPoseFromWebcam();
}
async function liveLoop() {
  if (!liveOn) return;
  await syncOrInferOneFrame();
  if (!liveOn) return;
  liveTimer = setTimeout(liveLoop, SYNC_INTERVAL_MS);
}
function setLive(on) {
  liveOn = on;

  if (liveTimer) {
    clearTimeout(liveTimer);
    liveTimer = null;
  }

  if (liveOn) {
    liveTimer = setTimeout(liveLoop, SYNC_INTERVAL_MS);
  }
  setRunButtonState();
}

/** =========================
 * three scene
 * ========================= */
function getAvatarLookTarget() {
  return AVATAR_POS.clone().add(LOOK_TARGET_OFFSET);
}
function getClipRange() {
  if (!squatClip) return { clipStart: 0, usableDuration: 0 };

  const clipDuration = Math.max(0.0001, squatClip.duration);
  const clipStart = clipDuration * CLIP_START_NORM;
  const clipEnd = clipDuration * CLIP_END_NORM;
  const usableDuration = Math.max(0.0001, clipEnd - clipStart);

  return { clipStart, usableDuration };
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
  scene.background = new THREE.Color(body.classList.contains("light-mode") ? 0xf4f7f6 : 0x0a0a0c);

  camera = new THREE.PerspectiveCamera(
  52,
  THREE_WRAP.clientWidth / THREE_WRAP.clientHeight,
  0.1,
  2000
);
  camera.position.set(0, 1.3, 2.6);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enabled = true;
  controls.enableDamping = true;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444466, 1.15);
  hemi.position.set(0, 2, 0);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(2, 4, 2);
  scene.add(dir);

  window.addEventListener("resize", () => {
    if (!renderer) return;
    const w = THREE_WRAP.clientWidth;
    const h = THREE_WRAP.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });

  animate();
}
async function loadGym(loader) {
  return new Promise((resolve, reject) => {
    loader.load(
      GYM_URL,
      (gltf) => {
        if (gymRoot) scene.remove(gymRoot);
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

        const clips = gltf.animations || [];
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

        sceneReady = true;
        resolve();
      },
      undefined,
      reject
    );
  });
}
async function loadScene() {
  initThreeIfNeeded();
  const loader = new GLTFLoader();
  await loadGym(loader);
  await loadAnimatedAvatar(loader);
  sceneReady = true;
}
function updateProgressControlledAnimation() {
  if (!mixer || !squatClip || !squatAction) return;

  const { clipStart, usableDuration } = getClipRange();
  const targetTime = clipStart + usableDuration * squatProgressSmooth;

  if (Math.abs(targetTime - lastMixerTime) > 0.001) {
    squatAction.time = targetTime;
    mixer.update(0);
    avatarScene?.updateMatrixWorld(true);
    lastMixerTime = targetTime;
  }
}
function animate(now = 0) {
  rafId = requestAnimationFrame(animate);

  if (controls) controls.update();

  const handCtrlEffective =
  inputMode === "webcam" &&
  videoReady &&
  handReady &&
  handRotateState !== "NEUTRAL";

  if (!handCtrlEffective && autoRotateEl.checked && camera && avatarScene) {
    const target = getAvatarLookTarget();
    const t = now * 0.0004;
    const radius = 3.4;
    camera.position.x = target.x + Math.sin(t) * radius;
    camera.position.z = target.z + Math.cos(t) * radius;
    camera.position.y = target.y + 0.45;
    camera.lookAt(target);
  }

  applyHandCameraControl();
  updateProgressControlledAnimation();

  if (now - lastRenderTime < RENDER_INTERVAL) return;
  lastRenderTime = now;

  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

/** =========================
 * app control
 * ========================= */
async function prepareAll() {
  if (prepareBusy) return;

  try {
    prepareBusy = true;
    readInputMode();
    readWorkoutInputs();

    setAnalysisStatus("준비 중...");
    logInfo("준비 시작...");

    if (inputMode === "json") {
      if (!poseReady) await loadPoseJson();
    } else {
      if (!yoloReady) await loadYoloModel();
      if (!handReady) await loadHandsModel();
    }

    if (!sceneReady) {
      await loadScene();
    }

    setAnalysisStatus(
      inputMode === "json"
        ? `준비 완료 / Pose:OK / Scene:OK`
        : `준비 완료 / YOLO:OK / Hands:OK / Scene:OK`
    );
    logInfo("준비 완료");
  } catch (err) {
    console.error(err);
    setAnalysisStatus("준비 실패");
    logInfo(`준비 실패: ${err.message}`);
  } finally {
    prepareBusy = false;
  }
}
async function toggleRun() {
  try {
    readInputMode();
    readWorkoutInputs();

    if (liveOn || videoReady) {
      stopInput(true);
      setAnalysisStatus("운동 종료");
      return;
    }

    if (!isPreparedForCurrentMode()) {
      await prepareAll();
      if (!isPreparedForCurrentMode()) {
        logInfo("준비가 완료되지 않음.");
        return;
      }
    }

    await startInput();

    if (inputMode === "json" && !poseReady) {
      logInfo("Pose JSON 준비 필요");
      return;
    }
    if (inputMode === "webcam" && (!yoloReady || !handReady)) {
      logInfo("YOLO / Hands 준비 필요");
      return;
    }

    setLive(true);
    setAnalysisStatus("분석 LIVE ON");
  } catch (err) {
    console.error(err);
    setAnalysisStatus("시작 실패");
    logInfo(`시작 실패: ${err.message}`);
  }
}
function resetAll() {
  setLive(false);
  stopRestTimer();
  resetOverlayState();

  resetHandCameraState(true);
  handCurrentYaw = 0;
  handTargetYaw = 0;

  if (videoReady && !isResting) {
    drawVideoFrameToCanvas();
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  resetSessionStats();
  resetAvatarPoseToStart();
  baselineDepthOffset = 0;

  setAnalysisStatus("초기화 완료");
  logInfo("초기화 완료");
}
function resetBaseline() {
  baselineDepthOffset = clamp(lastDepth * 0.4, 0, 0.2);
  setAnalysisStatus("기준 리셋 완료");
  logInfo(`baselineDepthOffset=${baselineDepthOffset.toFixed(3)}`);
}

/** =========================
 * events
 * ========================= */
themeToggleBtn.addEventListener("click", () => {
  body.classList.toggle("light-mode");
  updateThemeButton();
});

goModeBtn.addEventListener("click", () => showScreen("mode"));
showGuideBtn.addEventListener("click", () => {
  alert("운동 시스템 보기:\n영상 입력 → 자세 분석 → 3D 아바타 → 실시간 피드백 → DB 저장");
});
goHomeBtn.addEventListener("click", goHome);
backToModeBtn.addEventListener("click", goMode);

modeCards.forEach((card) => {
  card.addEventListener("click", () => {
    const mode = card.dataset.mode || "json";
    goAppWithMode(mode);
  });
});

prepareBtn.addEventListener("click", prepareAll);
runBtn.addEventListener("click", toggleRun);
resetBtn.addEventListener("click", resetAll);
resetBaselineBtn.addEventListener("click", resetBaseline);
saveResultBtn.addEventListener("click", saveSessionResult);

inputModeRadios.forEach((el) => {
  el.addEventListener("change", () => {
    readInputMode();
    updateResultCards();
    setAnalysisStatus(`입력 모드 변경: ${getInputModeLabel()}`);
  });
});

workoutModeRadios.forEach((el) => {
  el.addEventListener("change", () => {
    readWorkoutInputs();
    updateResultCards();
  });
});

[targetSetsInput, setRepTargetInput, restSecondsInput].forEach((el) => {
  el.addEventListener("input", () => {
    readWorkoutInputs();
    updateResultCards();
  });
});

/** =========================
 * boot
 * ========================= */
video.autoplay = false;
video.muted = true;
video.loop = true;
video.playsInline = true;

readInputMode();
readWorkoutInputs();
updateThemeButton();
updateResultCards();
setRunButtonState();
setAnalysisStatus("Ready (모드 선택 → 준비하기 → 시작)");
logInfo("(no data)");
showScreen("home");