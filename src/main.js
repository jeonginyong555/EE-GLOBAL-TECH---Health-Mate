/* eslint-disable no-console */
import * as THREE from "three";
import * as ort from "onnxruntime-web";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import MediapipeAvatarManager from "./MediapipeAvatarManager.js";
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

const YOLO_MODEL_URL = "/models/yolo_pose.onnx";
const ORT_DIST_BASE = "/ort/";

ort.env.wasm.wasmPaths = ORT_DIST_BASE;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const GYM_LIST = {
  gym1: {
    name: "gym1",
    url: "/models/Untitled_gym.glb",
    thumbnail: "/images/Untitled_gym.png",
  },
  gym2: {
    name: "gym2",
    url: "/models/Untitled_gym2.glb",
    thumbnail: "/images/Untitled_gym2.png",
  },
  gym3: {
    name: "gym3",
    url: "/models/Untitled_gym3.glb",
    thumbnail: "/images/Untitled_gym3.png",
  },
  gym4: {
    name: "gym4",
    url: "/models/Untitled_gym4.glb",
    thumbnail: "/images/Untitled_gym4.png",
  },
  cyber: {
  name: "Cyber Fitness",
  type: "image",
  url: "/images/3d.png",
  thumbnail: "/images/3d.png",
},
ptzone: {
  name: "PT Zone",
  type: "image",
  url: "/images/astrogen.png",
  thumbnail: "/images/astrogen.png",
},
darkroom: {
  name: "Dark Room",
  type: "image",
  url: "/images/org.png",
  thumbnail: "/images/org.png",
},
brickgym: {
  name: "Brick Gym",
  type: "image",
  url: "/images/teto.png",
  thumbnail: "/images/teto.png",
},
};

const AVATAR_LIST = [
  {
    id: "default",
    name: "기본 RPM",
    url: "/models/Untitled_squat.glb",
    thumbnail: "/images/avatar_default.png",
  },
  {
    id: "vroid1",
    name: "VRoid 1",
    url: "/models/Untitled_vroid_squat_1.glb",
    thumbnail: "/images/avatar_vroid1.png",
  },
  {
    id: "vroid2",
    name: "VRoid 2",
    url: "/models/Untitled_vroid_squat_2.glb",
    thumbnail: "/images/avatar_vroid2.png",
  },
  {
    id: "vroid3",
    name: "VRoid 3",
    url: "/models/Untitled_vroid_squat_3.glb",
    thumbnail: "/images/avatar_vroid3.png",
  },
  {
    id: "vroid4",
    name: "VRoid 4",
    url: "/models/Untitled_vroid_squat_4.glb",
    thumbnail: "/images/avatar_vroid4.png",
  },
];

function updateAvatarPreview() {
  const selected = getSelectedAvatarInfo();

  if (dom.avatarPreviewImage) {
    dom.avatarPreviewImage.src = selected.thumbnail;
  }

  if (dom.avatarPreviewName) {
    dom.avatarPreviewName.textContent = selected.name;
  }
}

function updateGymPreview() {
  const selected = getSelectedGymInfo();

  if (dom.gymPreviewImage) {
    dom.gymPreviewImage.src = selected.thumbnail;
  }

  if (dom.gymPreviewName) {
    dom.gymPreviewName.textContent = selected.name;
  }
}

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
  avatarSelect: document.getElementById("avatarSelect"),

  avatarPreviewImage: document.getElementById("avatarPreviewImage"),
  avatarPreviewName: document.getElementById("avatarPreviewName"),

  modeRecordsBtn: document.getElementById("modeRecordsBtn"),
  openRecordingsBtn: document.getElementById("openRecordingsBtn"),
  modeAnalyticsBtn: document.getElementById("modeAnalyticsBtn"),

  viewerStage: document.getElementById("viewerStage"),
  video: document.getElementById("video"),
  webcamStage: document.getElementById("webcamStage"),
  canvas: document.getElementById("canvas"),
  overlay: document.getElementById("overlay"),

  guideStage: document.getElementById("guideStage"),
  guideVideo: document.getElementById("guideVideo"),
  guideCanvas: document.getElementById("guideCanvas"),

  threeWrap: document.getElementById("threeWrap"),

  modeChip: document.getElementById("modeChip"),
  statusChip: document.getElementById("statusChip"),
  helpChip: document.getElementById("helpChip"),
  gestureChip: document.getElementById("gestureChip"),
  counterChip: document.getElementById("counterChip"),
  feedbackToastWrap: document.getElementById("feedbackToastWrap"),

  centerAlert: document.getElementById("centerAlert"),
  centerAlertTitle: document.getElementById("centerAlertTitle"),
  centerAlertText: document.getElementById("centerAlertText"),

  manualToggleBtn: document.getElementById("manualToggleBtn"),
  restartBtn: document.getElementById("restartBtn"),
  recordAvatarBtn: document.getElementById("recordAvatarBtn"),
  cameraRotateToggle: document.getElementById("cameraRotateToggle"),

  pipSettingsBtn: document.getElementById("pipSettingsBtn"),
  pipSettingsPanel: document.getElementById("pipSettingsPanel"),
  pipToggleLabel: document.getElementById("pipToggleLabel"),
  pipVisibleToggle: document.getElementById("pipVisibleToggle"),

  feedbackWebcamOverlayToggleRow: document.getElementById("feedbackWebcamOverlayToggleRow"),
  feedbackGuideOverlayToggleRow: document.getElementById("feedbackGuideOverlayToggleRow"),

  avatarFistToggleRow: document.getElementById("avatarFistToggleRow"),
  avatarKnifeToggleRow: document.getElementById("avatarKnifeToggleRow"),
  avatarOverlayToggleRow: document.getElementById("avatarOverlayToggleRow"),
  avatarMarkerToggleRow: document.getElementById("avatarMarkerToggleRow"),

  feedbackWebcamOverlayToggle: document.getElementById("feedbackWebcamOverlayToggle"),
  feedbackGuideOverlayToggle: document.getElementById("feedbackGuideOverlayToggle"),

  avatarFistToggle: document.getElementById("avatarFistToggle"),
  avatarKnifeToggle: document.getElementById("avatarKnifeToggle"),
  avatarOverlayToggle: document.getElementById("avatarOverlayToggle"),
  avatarMarkerToggle: document.getElementById("avatarMarkerToggle"),

  saveRecordBtn: document.getElementById("saveRecordBtn"),
  openRecordsBtn: document.getElementById("openRecordsBtn"),
  openAnalyticsBtn: document.getElementById("openAnalyticsBtn"),
  studentIdInput: document.getElementById("studentIdInput"),
  userNameInput: document.getElementById("userNameInput"),
  workoutTypeSelect: document.getElementById("workoutTypeSelect"),
  normalWorkoutSettings: document.getElementById("normalWorkoutSettings"),
  challengeWorkoutSettings: document.getElementById("challengeWorkoutSettings"),
  setCountInput: document.getElementById("setCountInput"),
  repCountInput: document.getElementById("repCountInput"),
  restSecondsInput: document.getElementById("restSecondsInput"),
  gymSelect: document.getElementById("gymSelect"),
  gymPreviewImage: document.getElementById("gymPreviewImage"),
  gymPreviewName: document.getElementById("gymPreviewName"),
  retargetAvatarSelect: document.getElementById("retargetAvatarSelect"),
  retargetVideoSelect: document.getElementById("retargetVideoSelect"),
};

const ctx = dom.canvas.getContext("2d", { willReadFrequently: true });
const octx = dom.overlay.getContext("2d", { willReadFrequently: true });
const gctx = dom.guideCanvas?.getContext("2d", { willReadFrequently: true });
const yoloCanvas = document.createElement("canvas");
const yoloCtx = yoloCanvas.getContext("2d", { willReadFrequently: true });

/* =========================
 * CONFIG
 * ========================= */
const DRAW_INTERVAL = 33;
const POSE_INTERVAL = 33;
const HAND_INTERVAL = 80;
const RENDER_FPS = 30;
const RENDER_INTERVAL = 1000 / RENDER_FPS;

const FIST_TOGGLE_HOLD_FRAMES = 5;
const FIST_TOGGLE_COOLDOWN_MS = 1500;
const CAMERA_ROTATE_STEP = 0.03;

const CLIP_START_NORM = 0.0;
const STANDARD_CLIP_END_NORM = 0.42;
const AVATAR_CLIP_END_NORM = 0.33;

const JSON_TIME_OFFSET_SEC = 0.020;

const RETARGET_POSE_INTERVAL = 66;

const YOLO_INPUT_SIZE = 640;
const YOLO_SCORE_THRESHOLD = 0.25;
const YOLO_KPT_THRESHOLD = 0.20;
const YOLO_IOU_THRESHOLD = 0.45;

const COCO_EDGES = [
  [5, 6], [5, 7], [7, 9],
  [6, 8], [8, 10],
  [11, 12], [5, 11], [6, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
];

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
  selectedAvatarId: "default",
  selectedGymId: "gym1",

  webcamStream: null,
  poseData: null,
  poseFPS: 30,
  jsonDepthRange: {
    min: 0,
    max: 1,
  },

  poseLandmarker: null,
  handLandmarker: null,
  yoloSession: null,
yoloReady: false,
yoloBusy: false,
retargetCocoKpts: null,

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

  feedbackDisplay: "guide",
  avatarDisplay: "avatar",
  retargetDisplay: "avatar",
  standardDisplay: "avatarMain",

  standardPipVisible: true,
  feedbackPipVisible: true,
  avatarPipVisible: true,
  retargetPipVisible: true,
  pipSettingsOpen: false,

  avatarFistGestureEnabled: true,
  avatarKnifeGestureEnabled: true,
  avatarOverlayEnabled: true,
  avatarMarkerEnabled: true,

  feedbackWebcamOverlayEnabled: true,
  feedbackGuideOverlayEnabled: true,

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

  centerAlertTimer: null,

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

  studentId: "",
  userName: "",

  lastGuideOverlayKey: "",

  workoutType: "normal",

workoutConfig: {
  setCount: 3,
  repsPerSet: 10,
  restSeconds: 30,
  challengeRepLimitSec: 10,
},

workoutSession: {
  currentSet: 1,
  currentRep: 0,
  isResting: false,
  restEndAt: 0,
  finished: false,
  challengeRepStartAt: 0,
  challengeFailed: false,
},
recording: {
  recorder: null,
  chunks: [],
  active: false,
},
retargetAvatarUrl: "model.glb",
retargetAvatarType: "Mixamo",
retargetVideoUrl: "",
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
let mediapipeAvatarManager = null;
let gymImageBg = null;

renderer = null;
scene = null;
camera = null;
controls = null;
gymRoot = null;
avatarScene = null;
mixer = null;
squatClip = null;
squatAction = null;

let feedbackMarkers = {
  upper: null,
  leftLeg: null,
  rightLeg: null,
  knee: null,
};

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

function getSelectedAvatarInfo() {
  return (
    AVATAR_LIST.find((avatar) => avatar.id === state.selectedAvatarId) ||
    AVATAR_LIST[0]
  );
}

function safePlay(videoEl) {
  return videoEl?.play?.().catch(() => {});
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
  if (!octx || !dom.overlay) return;
  octx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
}

function clearCanvas() {
  if (!ctx || !dom.canvas) return;
  ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
}

function clearGuideOverlay() {
  if (!gctx || !dom.guideCanvas) return;
  gctx.clearRect(0, 0, dom.guideCanvas.width, dom.guideCanvas.height);
  state.lastGuideOverlayKey = "";
}

function resizeStageToVideo(videoWidth, videoHeight) {
  const w = Math.max(1, videoWidth);
  const h = Math.max(1, videoHeight);
  if (dom.canvas) {
    dom.canvas.width = w;
    dom.canvas.height = h;
  }
  if (dom.overlay) {
    dom.overlay.width = w;
    dom.overlay.height = h;
  }
}

function resizeGuideOverlay() {
  if (!dom.guideStage || !dom.guideCanvas) return;

  const rect = dom.guideStage.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  if (dom.guideCanvas.width !== w) dom.guideCanvas.width = w;
  if (dom.guideCanvas.height !== h) dom.guideCanvas.height = h;
}

function getContainRect(srcW, srcH, dstW, dstH) {
  if (!srcW || !srcH || !dstW || !dstH) {
    return { x: 0, y: 0, w: dstW, h: dstH };
  }

  const srcRatio = srcW / srcH;
  const dstRatio = dstW / dstH;

  let w;
  let h;
  let x;
  let y;

  if (srcRatio > dstRatio) {
    w = dstW;
    h = w / srcRatio;
    x = 0;
    y = (dstH - h) * 0.5;
  } else {
    h = dstH;
    w = h * srcRatio;
    x = (dstW - w) * 0.5;
    y = 0;
  }

  return { x, y, w, h };
}

function setStatus(text) {
  if (dom.statusChip) dom.statusChip.textContent = text;
}

function setHelp(text) {
  if (dom.helpChip) dom.helpChip.textContent = text;
}

function setFeedbackText(text = "대기") {
  state.feedbackText = text;

  if (!dom.statusChip) return;

  if (state.currentMode === "feedback" || state.currentMode === "avatar") {
    dom.statusChip.textContent = "READY";
  } else {
    dom.statusChip.textContent = text;
  }
}

function showModeScreen() {
  dom.modeScreen?.classList.remove("hidden");
  dom.appScreen?.classList.add("hidden");
}

function showAppScreen() {
  dom.modeScreen?.classList.add("hidden");
  dom.appScreen?.classList.remove("hidden");
}

function resetViewVisibility() {
  if (dom.video) dom.video.style.display = "none";
  if (dom.webcamStage) dom.webcamStage.style.display = "none";
  if (dom.guideStage) dom.guideStage.style.display = "none";
  if (dom.threeWrap) dom.threeWrap.style.display = "none";

  dom.webcamStage?.classList.remove("stage-main", "stage-pip");
  dom.guideStage?.classList.remove("stage-main", "stage-pip");
  dom.threeWrap?.classList.remove("stage-main", "stage-pip");

  for (const el of [dom.webcamStage, dom.guideStage, dom.threeWrap]) {
    if (!el) continue;
    el.style.inset = "";
    el.style.width = "";
    el.style.height = "";
    el.style.zIndex = "";
  }
}

function setLayerOrder(mainEl, pipEl) {
  const layers = [dom.webcamStage, dom.guideStage, dom.threeWrap];
  layers.forEach((el) => {
    if (!el) return;
    el.style.zIndex = "1";
  });

  if (mainEl) mainEl.style.zIndex = "5";
  if (pipEl) pipEl.style.zIndex = "12";
}

function setTheme() {
  state.themeLight = !state.themeLight;
  dom.body?.classList.toggle("light-mode", state.themeLight);
  if (dom.themeToggle) {
    dom.themeToggle.textContent = state.themeLight
      ? "☀️ LIGHT MODE"
      : "🌙 DARK MODE";
  }

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

  if (isRetargetVideoOnly()) {
    dom.counterChip.classList.add("hidden");
    return;
  }

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
  if (dom.feedbackToastWrap) {
    dom.feedbackToastWrap.innerHTML = "";
  }

  if (state.centerAlertTimer) {
    clearTimeout(state.centerAlertTimer);
    state.centerAlertTimer = null;
  }

  hideCenterAlert();
}

function hidePipSettings() {
  state.pipSettingsOpen = false;
  dom.pipSettingsPanel?.classList.add("hidden");
}

function getCurrentPipVisible() {
  if (state.currentMode === "standard") return state.standardPipVisible;
  if (state.currentMode === "feedback") return state.feedbackPipVisible;
  if (state.currentMode === "avatar") return state.avatarPipVisible;
  if (state.currentMode === "retarget") return state.retargetPipVisible;
  return true;
}

function syncPipSettingsUI() {
  if (!dom.pipVisibleToggle || !dom.pipToggleLabel) return;

  let label = "서브 화면 표시";

  if (state.currentMode === "feedback") {
    label =
      state.feedbackDisplay === "guide"
        ? "웹캠 작은창 표시"
        : "가이드 작은창 표시";
  } else if (state.currentMode === "avatar") {
    label =
      state.avatarDisplay === "avatar"
        ? "웹캠 작은창 표시"
        : "아바타 작은창 표시";
  } else if (state.currentMode === "standard") {
    label =
      state.standardDisplay === "avatarMain"
        ? "비디오 작은창 표시"
        : "아바타 작은창 표시";
  } else if (state.currentMode === "retarget") {
  if (isRetargetVideoOnly()) {
    label =
      state.retargetDisplay === "avatar"
        ? "비디오 작은창 표시"
        : "아바타 작은창 표시";
  } else {
    label =
      state.retargetDisplay === "avatar"
        ? "웹캠 작은창 표시"
        : "아바타 작은창 표시";
  }
}
  dom.pipToggleLabel.textContent = label;
  dom.pipVisibleToggle.checked = getCurrentPipVisible();

  const isAvatarMode =
  state.currentMode === "avatar" || state.currentMode === "retarget";
  const isFeedbackMode = state.currentMode === "feedback";

  const hideAvatarOptions =
  !isAvatarMode || isRetargetVideoOnly();

dom.avatarFistToggleRow?.classList.toggle("hidden", hideAvatarOptions);
dom.avatarKnifeToggleRow?.classList.toggle("hidden", hideAvatarOptions);
dom.avatarOverlayToggleRow?.classList.toggle("hidden", hideAvatarOptions);
dom.avatarMarkerToggleRow?.classList.toggle("hidden", hideAvatarOptions);

  dom.feedbackWebcamOverlayToggleRow?.classList.toggle("hidden", !isFeedbackMode);
  dom.feedbackGuideOverlayToggleRow?.classList.toggle("hidden", !isFeedbackMode);

  if (dom.avatarFistToggle) {
    dom.avatarFistToggle.checked = state.avatarFistGestureEnabled;
  }

  if (dom.avatarKnifeToggle) {
    dom.avatarKnifeToggle.checked = state.avatarKnifeGestureEnabled;
  }

  if (dom.avatarOverlayToggle) {
    dom.avatarOverlayToggle.checked = state.avatarOverlayEnabled;
  }

  if (dom.avatarMarkerToggle) {
    dom.avatarMarkerToggle.checked = state.avatarMarkerEnabled;
  }

  if (dom.feedbackWebcamOverlayToggle) {
    dom.feedbackWebcamOverlayToggle.checked = state.feedbackWebcamOverlayEnabled;
  }

  if (dom.feedbackGuideOverlayToggle) {
    dom.feedbackGuideOverlayToggle.checked = state.feedbackGuideOverlayEnabled;
  }
}

function togglePipSettings() {
  state.pipSettingsOpen = !state.pipSettingsOpen;
  dom.pipSettingsPanel?.classList.toggle("hidden", !state.pipSettingsOpen);
  syncPipSettingsUI();
}

function updatePipSettingsButtonVisibility() {
  const visible =
  !isRetargetVideoOnly() &&
  (
    state.currentMode === "standard" ||
    state.currentMode === "feedback" ||
    state.currentMode === "avatar" ||
    state.currentMode === "retarget"
  );

  if (dom.pipSettingsBtn) {
    dom.pipSettingsBtn.style.display = visible ? "flex" : "none";
  }

  if (!visible) {
    hidePipSettings();
  }
}

function setCurrentPipVisible(visible) {
  if (state.currentMode === "standard") {
    state.standardPipVisible = visible;
    applyStandardDisplayMode();
  } else if (state.currentMode === "feedback") {
    state.feedbackPipVisible = visible;
    applyFeedbackDisplayMode();
  } else if (state.currentMode === "avatar") {
    state.avatarPipVisible = visible;
    applyAvatarDisplayMode();
  } else if (state.currentMode === "retarget") {
  state.retargetPipVisible = visible;
  applyRetargetDisplayMode();
}

  syncPipSettingsUI();
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

function hideCenterAlert() {
  if (!dom.centerAlert) return;
  dom.centerAlert.classList.remove("show", "good", "warn", "bad", "missing");
  dom.centerAlert.classList.add("hidden");
}

function showCenterAlert({
  level = "warn",
  title = "알림",
  text = "",
  duration = 1400,
}) {
  if (isRetargetVideoOnly()) return;

  if (!dom.centerAlert || !dom.centerAlertTitle || !dom.centerAlertText) return;
  if (
  state.currentMode !== "feedback" &&
  state.currentMode !== "avatar" &&
  state.currentMode !== "retarget"
) return;

  if (state.centerAlertTimer) {
    clearTimeout(state.centerAlertTimer);
    state.centerAlertTimer = null;
  }

  dom.centerAlert.classList.remove("hidden", "good", "warn", "bad", "missing");
  dom.centerAlert.classList.add(level);

  dom.centerAlertTitle.textContent = title;
  dom.centerAlertText.textContent = text;

  requestAnimationFrame(() => {
    dom.centerAlert?.classList.add("show");
  });

  state.centerAlertTimer = setTimeout(() => {
    dom.centerAlert?.classList.remove("show");

    setTimeout(() => {
      dom.centerAlert?.classList.add("hidden");
      dom.centerAlert?.classList.remove("good", "warn", "bad", "missing");
    }, 180);

    state.centerAlertTimer = null;
  }, duration);
}

function tryToastFeedback(key, config, cooldownMs = 1400) {
  const now = performance.now();
  const last = state.feedbackToastCooldowns[key] || 0;

  if (now - last < cooldownMs) return;

  state.feedbackToastCooldowns[key] = now;
  showCenterAlert(config);
}

function emitDetailedFeedbackToasts() {
  if (isRetargetVideoOnly()) return;

  if (
  state.currentMode !== "feedback" &&
  state.currentMode !== "avatar" &&
  state.currentMode !== "retarget"
) return;

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

function getEdgePartState(a, b) {
  // 상체
  if (
    [11,12,13,14,15,16].includes(a) ||
    [11,12,13,14,15,16].includes(b)
  ) {
    return state.partFeedback.upper;
  }

  // 왼쪽 다리
  if (
    [23,25,27].includes(a) ||
    [23,25,27].includes(b)
  ) {
    return state.partFeedback.leftLeg;
  }

  // 오른쪽 다리
  if (
    [24,26,28].includes(a) ||
    [24,26,28].includes(b)
  ) {
    return state.partFeedback.rightLeg;
  }

  // 무릎 따로 강조하고 싶으면 추가 가능

  return "normal";
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

function getKSTString() {
  return new Date().toLocaleString("ko-KR", {
    timeZone: "Asia/Seoul",
  });
}

function buildWorkoutPayload() {
  syncUserInfoFromInputs();

  return {
    studentId: state.studentId,
    name: state.userName,
    count: state.squatCount,
    avgDepth: Number(state.squatProgressSmooth?.toFixed?.(3) || 0),
    depthLowCount: state.shallowHoldActive ? 1 : 0,
    torsoWarningCount:
      state.partFeedback.upper === "warn" || state.partFeedback.upper === "bad"
        ? 1
        : 0,
    mode: state.currentMode || "unknown",
    inputMode: "webcam",
    setResults: [
      {
        count: state.squatCount,
        avgDepth: Number(state.squatProgressSmooth?.toFixed?.(3) || 0),
        upper: state.partFeedback.upper,
        leftLeg: state.partFeedback.leftLeg,
        rightLeg: state.partFeedback.rightLeg,
        knee: state.partFeedback.knee,
        feedbackText: state.feedbackText,
      },
    ],
    kst_time: getKSTString(),
  };
}

async function saveWorkoutRecord() {
  try {
    if (!validateUserInfo()) return;

    const payload = buildWorkoutPayload();
    const API_BASE = "http://localhost:3000";

    const res = await fetch(`${API_BASE}/api/workout`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const text = await res.text();
    let data = {};

    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      throw new Error(`서버 응답이 JSON이 아님: ${text || "empty response"}`);
    }

    if (!res.ok) {
      throw new Error(data?.error || data?.message || "저장 실패");
    }

    setStatus("저장 완료");
    alert("운동 기록 저장 완료");
  } catch (err) {
    console.error("SAVE WORKOUT ERROR:", err);
    alert(`저장 실패: ${err.message}`);
  }
}

async function uploadRecordingBlob(blob) {
  try {
    syncUserInfoFromInputs();

    const formData = new FormData();
    formData.append("video", blob, `avatar_recording_${Date.now()}.webm`);
    formData.append("studentId", state.studentId || "");
    formData.append("name", state.userName || "");
    formData.append("mode", state.currentMode || "avatar");
    formData.append("kst_time", getKSTString());

    const res = await fetch("http://localhost:3000/api/recording", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data?.message || "녹화 저장 실패");
    }

    alert(`녹화 저장 완료\n${data.url}`);
  } catch (err) {
    console.error("RECORDING UPLOAD ERROR:", err);
    alert(`녹화 저장 실패: ${err.message}`);
  }
}

function startAvatarRecording() {
  if (state.currentMode !== "avatar" && state.currentMode !== "retarget") {
  alert("아바타/리타게팅 모드에서만 녹화할 수 있습니다.");
  return;
}

  const canvas = renderer?.domElement;
  if (!canvas) {
    alert("아바타 화면을 찾지 못했습니다.");
    return;
  }

  const stream = canvas.captureStream(30);

  state.recording.chunks = [];
  state.recording.recorder = new MediaRecorder(stream, {
    mimeType: "video/webm",
  });

  state.recording.recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      state.recording.chunks.push(e.data);
    }
  };

  state.recording.recorder.onstop = async () => {
    const blob = new Blob(state.recording.chunks, {
      type: "video/webm",
    });

    state.recording.active = false;
    if (dom.recordAvatarBtn) dom.recordAvatarBtn.textContent = "⏺ REC";

    await uploadRecordingBlob(blob);
  };

  state.recording.recorder.start();
  state.recording.active = true;

  if (dom.recordAvatarBtn) dom.recordAvatarBtn.textContent = "⏹ STOP";
}

function stopAvatarRecording() {
  if (!state.recording.recorder) return;

  if (state.recording.recorder.state !== "inactive") {
    state.recording.recorder.stop();
  }
}

function syncUserInfoFromInputs() {
  state.studentId = dom.studentIdInput?.value?.trim() || "";
  state.userName = dom.userNameInput?.value?.trim() || "";
}

function validateUserInfo() {
  syncUserInfoFromInputs();

  if (!state.studentId) {
    alert("학번을 입력해 주세요.");
    dom.studentIdInput?.focus();
    return false;
  }

  if (!state.userName) {
    alert("이름을 입력해 주세요.");
    dom.userNameInput?.focus();
    return false;
  }

  return true;
}

function syncWorkoutConfigFromInputs() {
  state.workoutType = dom.workoutTypeSelect?.value || "normal";

  state.workoutConfig.setCount = clamp(
    Number(dom.setCountInput?.value || 3),
    1,
    20
  );

  state.workoutConfig.repsPerSet = clamp(
    Number(dom.repCountInput?.value || 10),
    1,
    100
  );

  state.workoutConfig.restSeconds = clamp(
    Number(dom.restSecondsInput?.value || 30),
    0,
    600
  );
}

function updateWorkoutSetupUI() {
  const isChallenge = (dom.workoutTypeSelect?.value || "normal") === "challenge";

  dom.normalWorkoutSettings?.classList.toggle("hidden", isChallenge);
  dom.challengeWorkoutSettings?.classList.toggle("hidden", !isChallenge);
}

function resetWorkoutSession() {
  state.workoutSession = {
    currentSet: 1,
    currentRep: 0,
    isResting: false,
    restEndAt: 0,
    finished: false,
    challengeRepStartAt: performance.now(),
    challengeFailed: false,
  };
}

function getWorkoutStatusText() {
  if (state.workoutType === "challenge") {
    if (state.workoutSession.challengeFailed) {
      return "CHALLENGE FAIL";
    }
    return `CHALLENGE · 10초 제한`;
  }

  if (state.workoutSession.finished) {
    return "WORKOUT DONE";
  }

  if (state.workoutSession.isResting) {
    const remainMs = Math.max(0, state.workoutSession.restEndAt - performance.now());
    const remainSec = Math.ceil(remainMs / 1000);
    return `휴식 ${remainSec}s`;
  }

  return `SET ${state.workoutSession.currentSet}/${state.workoutConfig.setCount} · REP ${state.workoutSession.currentRep}/${state.workoutConfig.repsPerSet}`;
}

function handleWorkoutProgressAfterRep() {
  if (state.workoutType === "challenge") {
    state.workoutSession.challengeRepStartAt = performance.now();
    setStatus(`CHALLENGE · REP ${state.squatCount}`);
    return;
  }

  if (state.workoutSession.finished) return;

  state.workoutSession.currentRep += 1;

  const repsPerSet = state.workoutConfig.repsPerSet;
  const setCount = state.workoutConfig.setCount;
  const restSeconds = state.workoutConfig.restSeconds;

  if (state.workoutSession.currentRep >= repsPerSet) {
    if (state.workoutSession.currentSet >= setCount) {
      state.workoutSession.finished = true;
      setStatus("WORKOUT DONE");
      setFeedbackText("모든 세트 완료");
      showCenterAlert({
        level: "good",
        title: "운동 완료",
        text: "설정한 모든 세트를 완료했습니다.",
        duration: 1800,
      });
      return;
    }

    state.workoutSession.currentSet += 1;
    state.workoutSession.currentRep = 0;

    if (restSeconds > 0) {
      state.workoutSession.isResting = true;
      state.workoutSession.restEndAt = performance.now() + restSeconds * 1000;

      showCenterAlert({
        level: "warn",
        title: "휴식 시작",
        text: `${restSeconds}초 휴식 후 다음 세트를 시작합니다.`,
        duration: 1600,
      });
    }
  }

  setStatus(getWorkoutStatusText());
}

function updateWorkoutRuntimeStatus() {
    if (isRetargetVideoOnly()) {
    setStatus("VIDEO ONLY");
    return;
  }

  if (
  state.currentMode !== "feedback" &&
  state.currentMode !== "avatar" &&
  state.currentMode !== "retarget"
) return;

  if (state.workoutType === "challenge") {
    const elapsedSec =
      (performance.now() - state.workoutSession.challengeRepStartAt) / 1000;

    if (elapsedSec > state.workoutConfig.challengeRepLimitSec) {
      state.workoutSession.challengeFailed = true;
      setStatus("CHALLENGE FAIL");
      showCenterAlert({
        level: "bad",
        title: "챌린지 실패",
        text: "한 번의 스쿼트를 10초 안에 완료하지 못했습니다.",
        duration: 1800,
      });

      state.workoutSession.challengeRepStartAt = performance.now();
    } else {
      setStatus(
        `CHALLENGE · ${Math.ceil(
          state.workoutConfig.challengeRepLimitSec - elapsedSec
        )}s`
      );
    }

    return;
  }

  if (state.workoutSession.isResting) {
    const remainMs = state.workoutSession.restEndAt - performance.now();

    if (remainMs <= 0) {
      state.workoutSession.isResting = false;
      setStatus(getWorkoutStatusText());

      showCenterAlert({
        level: "good",
        title: "다음 세트 시작",
        text: `${state.workoutSession.currentSet}세트를 시작하세요.`,
        duration: 1300,
      });
    } else {
      setStatus(getWorkoutStatusText());
    }
  } else {
    setStatus(getWorkoutStatusText());
  }
}

function getSelectedGymInfo() {
  return GYM_LIST[state.selectedGymId] || GYM_LIST.gym1;
}

function isRetargetVideoOnly() {
  return (
    state.currentMode === "retarget" &&
    window.HEALTH_MATE_RETARGET_VIDEO_ONLY === true
  );
}
function isCleanVideoUiMode() {
  return (
    state.currentMode === "standard" ||
    isRetargetVideoOnly()
  );
}

function shouldShowCameraRotate() {
  if (state.currentMode === "avatar") return true;

  if (state.currentMode === "retarget") {
    const exercise = document.getElementById("retargetExerciseSelect")?.value;
    return exercise === "free"; // 웹캠일 때만
  }

  return false;
}

function updateCleanVideoUi() {
  const clean = isCleanVideoUiMode();

  document.body.classList.toggle("clean-video-ui", clean);

  const topbarItems = [
    dom.saveRecordBtn,
    dom.openRecordsBtn,
    dom.openAnalyticsBtn,
    dom.openRecordingsBtn,
    dom.studentIdInput,
    dom.userNameInput,
  ];

  topbarItems.forEach((el) => {
    if (!el) return;
    el.classList.toggle("hidden", clean);
  });
}
// 🔥 camera rotate 조건 제어 추가
const showCamera = shouldShowCameraRotate();

if (dom.cameraRotateToggle) {
  dom.cameraRotateToggle.checked = showCamera && state.cameraRotateEnabled;
  dom.cameraRotateToggle.closest("label")?.classList.toggle("hidden", !showCamera);
}
/* =========================
 * RETARGET MODE - YOLO ONLY
 * ========================= */
async function loadYoloModel() {
  if (state.yoloReady && state.yoloSession) return;

  state.yoloSession = await ort.InferenceSession.create(YOLO_MODEL_URL, {
    executionProviders: ["wasm"],
  });

  state.yoloReady = true;

  console.log("YOLO READY", {
    inputs: state.yoloSession.inputNames,
    outputs: state.yoloSession.outputNames,
  });
}

function yoloIou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);

  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;

  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

  return inter / Math.max(1e-6, areaA + areaB - inter);
}

function yoloNms(boxes) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep = [];

  while (sorted.length) {
    const cur = sorted.shift();
    keep.push(cur);

    for (let i = sorted.length - 1; i >= 0; i--) {
      if (yoloIou(cur, sorted[i]) > YOLO_IOU_THRESHOLD) {
        sorted.splice(i, 1);
      }
    }
  }

  return keep;
}

function preprocessYolo(videoEl) {
  const srcW = videoEl.videoWidth || 640;
  const srcH = videoEl.videoHeight || 480;

  yoloCanvas.width = YOLO_INPUT_SIZE;
  yoloCanvas.height = YOLO_INPUT_SIZE;

  yoloCtx.fillStyle = "black";
  yoloCtx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);

  const scale = Math.min(YOLO_INPUT_SIZE / srcW, YOLO_INPUT_SIZE / srcH);
  const drawW = Math.round(srcW * scale);
  const drawH = Math.round(srcH * scale);
  const padX = Math.floor((YOLO_INPUT_SIZE - drawW) / 2);
  const padY = Math.floor((YOLO_INPUT_SIZE - drawH) / 2);

  yoloCtx.drawImage(videoEl, 0, 0, srcW, srcH, padX, padY, drawW, drawH);

  const imageData = yoloCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  const data = imageData.data;
  const area = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
  const chw = new Float32Array(3 * area);

  for (let i = 0; i < area; i++) {
    chw[i] = data[i * 4] / 255;
    chw[area + i] = data[i * 4 + 1] / 255;
    chw[area * 2 + i] = data[i * 4 + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", chw, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]),
    meta: { srcW, srcH, scale, padX, padY },
  };
}

function decodeYolo(outputTensor, meta) {
  const dims = outputTensor.dims;
  const arr = outputTensor.data;

  let rows = [];
  let featureSize = 0;
  let count = 0;

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
      rows.push(arr.slice(i * featureSize, i * featureSize + featureSize));
    }
  } else {
    throw new Error(`YOLO output dims 오류: ${JSON.stringify(dims)}`);
  }

  const boxes = [];

  for (const row of rows) {
    const cx = row[0];
    const cy = row[1];
    const bw = row[2];
    const bh = row[3];
    const score = row[4];

    if (score < YOLO_SCORE_THRESHOLD) continue;

    const kpts = [];

    for (let k = 0; k < 17; k++) {
      const base = 5 + k * 3;
      const x = row[base];
      const y = row[base + 1];
      const s = row[base + 2];

      const rx = (x - meta.padX) / Math.max(1e-6, meta.scale);
      const ry = (y - meta.padY) / Math.max(1e-6, meta.scale);

      kpts.push([
        clamp01(rx / meta.srcW),
        clamp01(ry / meta.srcH),
        s >= YOLO_KPT_THRESHOLD ? s : 0,
      ]);
    }

    boxes.push({
      x1: cx - bw / 2,
      y1: cy - bh / 2,
      x2: cx + bw / 2,
      y2: cy + bh / 2,
      score,
      kpts,
    });
  }

  const kept = yoloNms(boxes);
  return kept[0]?.kpts || null;
}

async function inferRetargetYoloFrame() {
  if (
    state.stopped ||
    state.currentMode !== "retarget" ||
    !state.yoloReady ||
    !state.yoloSession ||
    state.yoloBusy ||
    !dom.video.videoWidth ||
    dom.video.readyState < 2
  ) return;

  state.yoloBusy = true;

  try {
    const { tensor, meta } = preprocessYolo(dom.video);
    const inputName = state.yoloSession.inputNames[0];
    const outputs = await state.yoloSession.run({ [inputName]: tensor });
    const outputTensor = outputs[state.yoloSession.outputNames[0]];

    const coco = decodeYolo(outputTensor, meta);

    if (!coco) {
      setStatus("YOLO 인식 없음");
      return;
    }

    state.retargetCocoKpts = coco;

    const metrics = computeRetargetSquatMetrics(coco);
    if (metrics) {
      state.squatProgressRaw = metrics.depth;
      state.squatProgressSmooth = lerp(state.squatProgressSmooth, metrics.depth, 0.35);

      if (state.squatState === "UP" && state.squatProgressSmooth > 0.58) {
        state.squatState = "DOWN";
      } else if (state.squatState === "DOWN" && state.squatProgressSmooth < 0.22) {
        state.squatState = "UP";
        state.squatCount += 1;
        updateCounterChip();
      }

      setStatus(`YOLO READY · ${state.squatCount}`);
    }
  } catch (err) {
    console.warn("RETARGET YOLO ERROR:", err);
    setStatus("YOLO ERROR");
  } finally {
    state.yoloBusy = false;
  }
}

function computeRetargetSquatMetrics(k) {
  if (!k || k.length < 17) return null;

  const lh = k[11];
  const rh = k[12];
  const lk = k[13];
  const rk = k[14];
  const la = k[15];
  const ra = k[16];

  const leftOk = kpOk(lh, 0.05) && kpOk(lk, 0.05) && kpOk(la, 0.05);
  const rightOk = kpOk(rh, 0.05) && kpOk(rk, 0.05) && kpOk(ra, 0.05);

  if (!leftOk && !rightOk) return null;

  const hip = leftOk ? lh : rh;
  const knee = leftOk ? lk : rk;
  const ankle = leftOk ? la : ra;

  const kneeAngle = angleDeg2(hip, knee, ankle);
  const kneeBend = clamp01((175 - kneeAngle) / 85);

  return {
    depth: kneeBend,
  };
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

  const cos = clamp(dot / (m1 * m2), -1, 1);
  return (Math.acos(cos) * 180) / Math.PI;
}

function drawRetargetCocoSkeleton(kpts) {
  clearOverlay();
  if (!kpts || !octx || !dom.overlay) return;

  const w = dom.overlay.width;
  const h = dom.overlay.height;
  const vw = dom.video.videoWidth || w;
const vh = dom.video.videoHeight || h;
const rect = getContainRect(vw, vh, w, h);

  octx.lineWidth = 3;
  octx.strokeStyle = FEEDBACK_COLORS.line;
  octx.fillStyle = FEEDBACK_COLORS.joint;

  for (const [a, b] of COCO_EDGES) {
    const pa = kpts[a];
    const pb = kpts[b];

    if (!kpOk(pa, 0.05) || !kpOk(pb, 0.05)) continue;

    octx.beginPath();
    octx.moveTo(rect.x + pa[0] * rect.w, rect.y + pa[1] * rect.h);
octx.lineTo(rect.x + pb[0] * rect.w, rect.y + pb[1] * rect.h);
    octx.stroke();
  }

  for (const p of kpts) {
    if (!kpOk(p, 0.05)) continue;

    octx.beginPath();
    octx.arc(rect.x + p[0] * rect.w, rect.y + p[1] * rect.h, 4, 0, Math.PI * 2);
    octx.fill();
  }
}
function drawRetargetMediapipeSkeleton(kpts) {
  clearOverlay();

  if (!kpts || !octx || !dom.overlay || !dom.video) return;

  const w = dom.overlay.width;
  const h = dom.overlay.height;

  octx.lineWidth = 3;

  for (const [a, b] of MP_POSE_EDGES) {
    const pa = kpts[a];
    const pb = kpts[b];

    if (!kpOk(pa, 0.05) || !kpOk(pb, 0.05)) continue;

    const partState = getEdgePartState(a, b);
    octx.strokeStyle = getPartColor(partState, false);

    octx.beginPath();
    octx.moveTo(pa[0] * w, pa[1] * h);
    octx.lineTo(pb[0] * w, pb[1] * h);
    octx.stroke();
  }

  for (let i = 0; i < kpts.length; i++) {
    const p = kpts[i];
    if (!kpOk(p, 0.05)) continue;

    const partState = getEdgePartState(i, i);
    octx.fillStyle = getPartColor(partState, true);

    octx.beginPath();
    octx.arc(p[0] * w, p[1] * h, 5, 0, Math.PI * 2);
    octx.fill();
  }
}

function startRetargetDrawLoop() {
  if (state.drawTimer) clearInterval(state.drawTimer);

  state.drawTimer = setInterval(() => {
    if (state.stopped || state.currentMode !== "retarget") return;

    const webcamPose = state.smoothPoseKpts;

    /* =========================
     * 1. 자세 분석
     * ========================= */
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
    }

    updateWorkoutRuntimeStatus();

    resizeStageToVideo(dom.video.videoWidth || 640, dom.video.videoHeight || 480);

drawWebcamFrame();

clearOverlay();

if (!isRetargetVideoOnly()) {
  if (
    webcamPose &&
    state.avatarOverlayEnabled &&
    state.currentMode === "retarget"
  ) {
    drawRetargetMediapipeSkeleton(webcamPose, false);
  }

  drawHandOverlay(dom.overlay.width, dom.overlay.height);
}

  }, DRAW_INTERVAL);
}

function applyRetargetDisplayMode() {
  if (state.currentMode !== "retarget") return;

    const videoOnly = isRetargetVideoOnly();

  if (videoOnly) {
    state.avatarFistGestureEnabled = false;
    state.avatarKnifeGestureEnabled = false;
    state.avatarOverlayEnabled = false;
    state.avatarMarkerEnabled = false;
    state.cameraRotateEnabled = false;

    if (dom.recordAvatarBtn) dom.recordAvatarBtn.classList.add("hidden");
    if (dom.pipSettingsBtn) dom.pipSettingsBtn.style.display = "none";
    if (dom.pipSettingsPanel) dom.pipSettingsPanel.classList.add("hidden");

    if (dom.cameraRotateToggle) {
      dom.cameraRotateToggle.checked = false;
    }
  }

  const avatarMain = state.retargetDisplay === "avatar";

  if (dom.video) dom.video.style.display = "none";
  if (dom.webcamStage) dom.webcamStage.style.display = "none";
  if (dom.guideStage) dom.guideStage.style.display = "none";
  if (dom.threeWrap) dom.threeWrap.style.display = "none";

  dom.webcamStage?.classList.remove("stage-main", "stage-pip");
  dom.threeWrap?.classList.remove("stage-main", "stage-pip");

  clearGuideOverlay();
  showCounter(true);

  if (avatarMain) {
    const videoOnly = isRetargetVideoOnly();

if (videoOnly) {
  dom.manualToggleBtn.style.display = "none";
  dom.gestureChip.classList.add("hidden");
  dom.recordAvatarBtn?.classList.add("hidden");

  if (dom.cameraRotateToggle) {
    dom.cameraRotateToggle.closest("label")?.classList.add("hidden");
  }

  if (dom.pipSettingsBtn) {
    dom.pipSettingsBtn.style.display = "flex";
  }

  dom.avatarFistToggleRow?.classList.add("hidden");
  dom.avatarKnifeToggleRow?.classList.add("hidden");
  dom.avatarOverlayToggleRow?.classList.add("hidden");
  dom.avatarMarkerToggleRow?.classList.add("hidden");
}

if (!videoOnly) {
  dom.manualToggleBtn.style.display = "inline-flex";
  dom.gestureChip.classList.remove("hidden");

  if (dom.cameraRotateToggle) {
    dom.cameraRotateToggle.closest("label")?.classList.remove("hidden");
  }
}

    dom.threeWrap.style.display = "block";
    dom.threeWrap.classList.add("stage-main");

    if (state.retargetPipVisible) {
      dom.webcamStage.style.display = "block";
      dom.webcamStage.classList.add("stage-pip");
      setLayerOrder(dom.threeWrap, dom.webcamStage);
    } else {
      setLayerOrder(dom.threeWrap, null);
    }

    dom.modeChip.textContent = "리타게팅 모드 · 아바타 메인";
    setHelp("MediaPipe 기반 포즈로 아바타를 제어합니다.");
  } else {
    dom.webcamStage.style.display = "block";
    dom.webcamStage.classList.add("stage-main");

    if (state.retargetPipVisible) {
      dom.threeWrap.style.display = "block";
      dom.threeWrap.classList.add("stage-pip");
      setLayerOrder(dom.webcamStage, dom.threeWrap);
    } else {
      setLayerOrder(dom.webcamStage, null);
    }

    dom.modeChip.textContent = "리타게팅 모드 · 웹캠 메인";
    setHelp("YOLO 스켈레톤 확인 화면입니다.");
  }

  requestAnimationFrame(() => {
  resizeThreeRenderer();
  resizeStageToVideo(dom.video.videoWidth || 640, dom.video.videoHeight || 480);
  clearOverlay();
});
}

async function startRetargetMode() {
  await destroyCurrentMode();

  syncWorkoutConfigFromInputs();
  resetWorkoutSession();

  state.currentMode = "retarget";
  updateCleanVideoUi();
  state.stopped = false;
  state.retargetDisplay = "avatar";

  state.avatarOverlayEnabled = true;
state.avatarMarkerEnabled = true;
state.avatarFistGestureEnabled = true;
state.avatarKnifeGestureEnabled = true;

  state.squatCount = 0;
  state.squatState = "UP";
  state.squatProgressRaw = 0;
  state.squatProgressSmooth = 0;
  state.retargetCocoKpts = null;
  state.lastRenderedCount = 0;

  showAppScreen();
  if (isRetargetVideoOnly()) {
  clearFeedbackToasts();
  clearOverlay();

  Object.values(feedbackMarkers).forEach((marker) => {
    if (marker) marker.visible = false;
  });
}
  resetViewVisibility();
  showCounter(true);
  updateCounterChip(true);

  dom.gestureChip.textContent = "GESTURE: -";
  setStatus("MEDIAPIPE LOADING...");
  setHelp("리타게팅 모드 준비 중");

  const isVideoOnly = isRetargetVideoOnly();

// PIP 옵션 제어
dom.avatarFistToggleRow?.classList.toggle("hidden", isVideoOnly);
dom.avatarKnifeToggleRow?.classList.toggle("hidden", isVideoOnly);
dom.avatarOverlayToggleRow?.classList.toggle("hidden", isVideoOnly);
dom.avatarMarkerToggleRow?.classList.toggle("hidden", isVideoOnly);

  if (isRetargetVideoOnly()) {
  dom.manualToggleBtn.style.display = isRetargetVideoOnly() ? "none" : "inline-flex";

  if (dom.cameraRotateToggle) {
    dom.cameraRotateToggle.checked = false;
  }

  state.cameraRotateEnabled = false;
  dom.recordAvatarBtn?.classList.add("hidden");
} else {
  dom.manualToggleBtn.style.display = "inline-flex";

  if (dom.cameraRotateToggle) {
    dom.cameraRotateToggle.checked = true;
  }

  state.cameraRotateEnabled = true;
  dom.recordAvatarBtn?.classList.remove("hidden");
}

  await loadPoseModel();
  await loadHandsModel();

if (!mediapipeAvatarManager) {
  mediapipeAvatarManager = new MediapipeAvatarManager();
}

if (state.retargetVideoUrl) {
  await startRetargetVideo();
} else {
  await startWebcam();
}

async function startRetargetVideo() {
  stopWebcam();

  if (!dom.video) return;

  dom.video.pause?.();
  dom.video.srcObject = null;
  dom.video.src = `/${state.retargetVideoUrl}`;
  dom.video.muted = true;
  dom.video.loop = true;
  dom.video.playsInline = true;
  dom.video.autoplay = true;

  await new Promise((resolve, reject) => {
    const onReady = () => {
      cleanup();
      resolve();
    };

    const onError = () => {
      cleanup();
      reject(new Error(`retarget video load failed: ${state.retargetVideoUrl}`));
    };

    const cleanup = () => {
      dom.video.removeEventListener("loadedmetadata", onReady);
      dom.video.removeEventListener("canplay", onReady);
      dom.video.removeEventListener("error", onError);
    };

    dom.video.addEventListener("loadedmetadata", onReady, { once: true });
    dom.video.addEventListener("canplay", onReady, { once: true });
    dom.video.addEventListener("error", onError, { once: true });

    dom.video.load();
  });

  resizeStageToVideo(dom.video.videoWidth || 640, dom.video.videoHeight || 480);
  await safePlay(dom.video);
}

  initThree();

  if (controls) {
    controls.enabled = true;
    controls.enableRotate = true;
    controls.enablePan = false;
    controls.enableZoom = true;
    controls.target.copy(getAvatarLookTarget());
    controls.update();
  }

  const loader = new GLTFLoader();
  await loadGym(loader);
  await loadAnimatedAvatar(loader);

  updatePipSettingsButtonVisibility();
  hidePipSettings();

  applyRetargetDisplayMode();
  startRetargetDrawLoop();

  state.poseTimer = setInterval(() => {
  inferPoseFrame();
}, POSE_INTERVAL);

state.handTimer = setInterval(() => {
  inferHands();
}, HAND_INTERVAL);

  animateThree();
  setStatus("MEDIAPIPE READY");
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
  if (dom.video) dom.video.srcObject = null;
}

async function startGuideVideo() {
  if (!dom.guideVideo) return;

  try {
    dom.guideVideo.pause();
  } catch {}

  dom.guideVideo.currentTime = 0;

  const targetSrc = `${window.location.origin}${GUIDE_VIDEO_URL}`;
  if (dom.guideVideo.currentSrc !== targetSrc) {
    dom.guideVideo.src = GUIDE_VIDEO_URL;
    // ✅ 영상 무한 루프
dom.guideVideo.loop = true;
dom.guideVideo.onended = () => {
  dom.guideVideo.currentTime = 0;
  dom.guideVideo.play().catch(() => {});
};
  }

  dom.guideVideo.muted = true;
  dom.guideVideo.loop = true;
  dom.guideVideo.playsInline = true;
  dom.guideVideo.preload = "auto";

  await new Promise((resolve, reject) => {
    let done = false;

    const onReady = () => {
      if (done) return;
      done = true;
      cleanup();
      resolve();
    };

    const onError = () => {
      if (done) return;
      done = true;
      cleanup();
      reject(new Error("guide video load failed"));
    };

    const cleanup = () => {
      dom.guideVideo.removeEventListener("canplay", onReady);
      dom.guideVideo.removeEventListener("loadeddata", onReady);
      dom.guideVideo.removeEventListener("error", onError);
    };

    dom.guideVideo.addEventListener("canplay", onReady, { once: true });
    dom.guideVideo.addEventListener("loadeddata", onReady, { once: true });
    dom.guideVideo.addEventListener("error", onError, { once: true });

    dom.guideVideo.load();
  });

  dom.guideVideo.currentTime = 0;

  await dom.guideVideo.play().catch((err) => {
    console.warn("guideVideo play 실패:", err);
  });

  dom.guideVideo.currentTime = 0;
  await safePlay(dom.guideVideo);
}

function stopGuideVideo() {
  if (!dom.guideVideo) return;

  try {
    dom.guideVideo.pause();
  } catch {}

  dom.guideVideo.currentTime = 0;
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
  const totalFrames = frames.length;
  if (!totalFrames) return null;

  const rawIndex = Math.floor(Math.max(0, timeSec) * state.poseFPS);
  const idx = ((rawIndex % totalFrames) + totalFrames) % totalFrames;

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

    if (
  state.currentMode === "retarget" &&
  mediapipeAvatarManager &&
  result?.landmarks?.[0] &&
  avatarScene
) {
  mediapipeAvatarManager.update({
    poseLandmarks: result.landmarks[0],
  });
}

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
  const kpts = mpPoseToKpts(lm);
state.lastPoseKpts = kpts;
updateSmoothPose(kpts);
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

  if (!torsoTop || !torsoBottom || !lh || !rh || !lk || !rk || !la || !ra) {
    return null;
  }

  return {
    upper: angle2D(torsoBottom, torsoTop),
    leftLeg: angle2D(lh, lk),
    rightLeg: angle2D(rh, rk),
    leftShin: angle2D(lk, la),
    rightShin: angle2D(rk, ra),
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
  const depthTooShallowBad = tryingSquat && squatDepth < 0.52;
  const depthTooShallowWarn = tryingSquat && squatDepth >= 0.52 && squatDepth < 0.68;

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
  if (!dom.video || dom.video.readyState < 2 || !ctx || !dom.canvas) return;

  const cw = dom.canvas.width;
  const ch = dom.canvas.height;
  if (!cw || !ch) return;

  clearCanvas();

  const vw = dom.video.videoWidth || cw;
  const vh = dom.video.videoHeight || ch;
  const rect = getContainRect(vw, vh, cw, ch);

  ctx.drawImage(dom.video, rect.x, rect.y, rect.w, rect.h);
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
  if (!kpts || !octx || !dom.overlay) return;

  const w = dom.overlay.width;
  const h = dom.overlay.height;

  const vw = dom.video.videoWidth || w;
  const vh = dom.video.videoHeight || h;
  const rect = getContainRect(vw, vh, w, h);

  octx.shadowBlur = 0;

  // 선 그리기
  for (const [a, b] of edges) {
    const pa = kpts[a];
    const pb = kpts[b];

    // 핵심: visibility 낮은 관절은 선 자체를 그리지 않음
    if (!kpOk(pa, 0.08) || !kpOk(pb, 0.08)) continue;

    const part = edgePartResolver(a, b);
    const partState = state.partFeedback[part] || "normal";
    const color = getPartColor(partState, false);

    octx.beginPath();
    octx.lineWidth = 3;
    octx.strokeStyle = color;
    octx.moveTo(rect.x + pa[0] * rect.w, rect.y + pa[1] * rect.h);
    octx.lineTo(rect.x + pb[0] * rect.w, rect.y + pb[1] * rect.h);
    octx.stroke();
  }

  // 관절 점 그리기
  for (let i = 0; i < kpts.length; i++) {
    const p = kpts[i];

    // 핵심: visibility 낮은 점은 아예 안 그림
    if (!kpOk(p, 0.08)) continue;

    const jointState = getJointOverrideState(i);
    const color = getPartColor(jointState, true);

    octx.beginPath();
    octx.fillStyle = color;
    octx.arc(
      rect.x + p[0] * rect.w,
      rect.y + p[1] * rect.h,
      4,
      0,
      Math.PI * 2
    );
    octx.fill();
  }
}

function drawSkeleton2D(kpts) {
  clearOverlay();
  if (!kpts) return;

  const w = dom.overlay.width;
  const h = dom.overlay.height;
  const vw = dom.video.videoWidth || w;
const vh = dom.video.videoHeight || h;
const rect = getContainRect(vw, vh, w, h);

  for (const [a, b] of MP_POSE_EDGES) {
    const pa = kpts[a];
    const pb = kpts[b];

    if (!kpOk(pa) || !kpOk(pb)) continue;

    // 🔥 여기 추가
    const partState = getEdgePartState(a, b);
    const color = getPartColor(partState);

    octx.strokeStyle = color;
    octx.lineWidth = 3;

    octx.beginPath();
    const rect = getContainRect(
  dom.video.videoWidth,
  dom.video.videoHeight,
  w,
  h
);

octx.moveTo(
  rect.x + pa[0] * rect.w,
  rect.y + pa[1] * rect.h
);

octx.lineTo(
  rect.x + pb[0] * rect.w,
  rect.y + pb[1] * rect.h
);
    octx.stroke();
  }

  // 관절도 색 적용하고 싶으면 추가
  for (const p of kpts) {
    if (!kpOk(p)) continue;

    octx.fillStyle = FEEDBACK_COLORS.joint;
    octx.beginPath();
    octx.arc(rect.x + p[0] * rect.w, rect.y + p[1] * rect.h, 4, 0, Math.PI * 2);
    octx.fill();
  }
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

function getJsonJointPart(index) {
  if ([5, 6, 7, 8, 9, 10, 11, 12].includes(index)) return "upper";
  if ([13, 15].includes(index)) return "leftLeg";
  if ([14, 16].includes(index)) return "rightLeg";
  return "upper";
}

function getJsonJointOverrideState(index) {
  if ([13, 14].includes(index)) {
    if (state.partFeedback.knee === "bad") return "bad";
    if (state.partFeedback.knee === "warn") return "warn";
    if (state.partFeedback.knee === "missing") return "missing";
  }

  const part = getJsonJointPart(index);
  return state.partFeedback[part] || "normal";
}

function getGuideOverlayKey(frame, frameIndex) {
  if (!frame?.valid || !frame?.keypoints) return `empty-${frameIndex}`;

  return JSON.stringify({
    fi: frameIndex,
    upper: state.partFeedback.upper,
    leftLeg: state.partFeedback.leftLeg,
    rightLeg: state.partFeedback.rightLeg,
    knee: state.partFeedback.knee,
  });
}

function drawGuideVideoFrameOnly() {
  if (!gctx || !dom.guideCanvas || !dom.guideVideo) return;

  resizeGuideOverlay();

  const cw = dom.guideCanvas.width;
  const ch = dom.guideCanvas.height;

  if (!cw || !ch) return;
  if (!dom.guideVideo.videoWidth || !dom.guideVideo.videoHeight) return;

  gctx.clearRect(0, 0, cw, ch);

  const rect = getContainRect(
    dom.guideVideo.videoWidth,
    dom.guideVideo.videoHeight,
    cw,
    ch
  );

  gctx.drawImage(
    dom.guideVideo,
    rect.x,
    rect.y,
    rect.w,
    rect.h
  );
}

function drawGuidePoseOverlay(frame, frameIndex = -1) {
  if (!gctx || !dom.guideCanvas) return;

  resizeGuideOverlay();

  if (state.currentMode !== "feedback") {
    clearGuideOverlay();
    return;
  }

  if (!state.feedbackGuideOverlayEnabled) {
    const key = `video-only-${frameIndex}`;
    if (key === state.lastGuideOverlayKey) return;
    state.lastGuideOverlayKey = key;
    drawGuideVideoFrameOnly();
    return;
  }

  const key = getGuideOverlayKey(frame, frameIndex);
  if (key === state.lastGuideOverlayKey) return;
  state.lastGuideOverlayKey = key;

  const cw = dom.guideCanvas.width;
  const ch = dom.guideCanvas.height;

  gctx.clearRect(0, 0, cw, ch);

  if (!dom.guideVideo.videoWidth || !dom.guideVideo.videoHeight) return;
  if (!cw || !ch) return;

  const rect = getContainRect(
    dom.guideVideo.videoWidth,
    dom.guideVideo.videoHeight,
    cw,
    ch
  );

  gctx.drawImage(
    dom.guideVideo,
    rect.x,
    rect.y,
    rect.w,
    rect.h
  );

  if (!frame?.valid || !frame?.keypoints) return;

  const kpts = frame.keypoints;

  for (const [a, b] of JSON_POSE_EDGES) {
    const pa = kpts[a];
    const pb = kpts[b];
    if (!pa || !pb) continue;

    const part = getJsonEdgePart(a, b);
    const partState = state.partFeedback[part] || "normal";
    const color = getPartColor(partState, false);

    gctx.beginPath();
    gctx.lineWidth = 3;
    gctx.strokeStyle = color;
    gctx.moveTo(rect.x + pa[0] * rect.w, rect.y + pa[1] * rect.h);
    gctx.lineTo(rect.x + pb[0] * rect.w, rect.y + pb[1] * rect.h);
    gctx.stroke();
  }

  for (let i = 0; i < kpts.length; i++) {
    const p = kpts[i];
    if (!p) continue;

    const jointState = getJsonJointOverrideState(i);
    const color = getPartColor(jointState, true);

    gctx.beginPath();
    gctx.fillStyle = color;
    gctx.arc(
      rect.x + p[0] * rect.w,
      rect.y + p[1] * rect.h,
      4,
      0,
      Math.PI * 2
    );
    gctx.fill();
  }
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
  handleWorkoutProgressAfterRep();
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
    state.feedbackDisplay =
      state.feedbackDisplay === "guide" ? "webcam" : "guide";
    clearFeedbackToasts();
    state.lastGuideOverlayKey = "";
    applyFeedbackDisplayMode();
  } else if (state.currentMode === "avatar") {
    state.avatarDisplay =
      state.avatarDisplay === "avatar" ? "webcam" : "avatar";
    applyAvatarDisplayMode();
  } else if (state.currentMode === "retarget") {
    state.retargetDisplay =
      state.retargetDisplay === "avatar" ? "webcam" : "avatar";
    applyRetargetDisplayMode();
  }
}

function handleFistToggle(lm) {
  if (
  (state.currentMode === "avatar" || state.currentMode === "retarget") &&
  !state.avatarFistGestureEnabled
) {
    state.fistHoldCount = 0;
    return false;
  }

  const fist =
  state.currentMode === "retarget"
    ? isFistLandmarks(lm)
    : isFistLandmarks(lm) && isHandCentered(lm);

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
    (
      state.currentMode !== "feedback" &&
      state.currentMode !== "avatar" &&
      state.currentMode !== "retarget"
    )
  ) return;

  state.handBusy = true;

  try {
    state.lastHandResult = state.handLandmarker.detectForVideo(
      dom.video,
      performance.now()
    );

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

    const isAvatarMain =
      (state.currentMode === "avatar" && state.avatarDisplay === "avatar") ||
      (state.currentMode === "retarget" && state.retargetDisplay === "avatar");

    if (isAvatarMain) {
      if (state.avatarKnifeGestureEnabled) {
        state.handRotateState = getKnifeRotateState(lm, handedness);
        dom.gestureChip.textContent = `GESTURE: ${state.handRotateState} (${handedness})`;
      } else {
        state.handRotateState = "NEUTRAL";
        dom.gestureChip.textContent = `GESTURE: HAND (${handedness})`;
      }
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
function createFeedbackMarker(label = "marker") {
  const group = new THREE.Group();
  group.name = `joint-dot-${label}`;
  group.visible = false;

  const dotGeo = new THREE.SphereGeometry(0.018, 10, 10); // ← 핵심: 작게
  const dotMat = new THREE.MeshBasicMaterial({
    color: 0xff3b30,
    transparent: true,
    opacity: 1,
    depthTest: false,
  });

  const dot = new THREE.Mesh(dotGeo, dotMat);
  dot.renderOrder = 999;

  group.add(dot);
  group.userData.dot = dot;

  return group;
}

function ensureFeedbackMarkers() {
  if (isRetargetVideoOnly()) {
  Object.values(feedbackMarkers).forEach((marker) => {
    if (marker) marker.visible = false;
  });
  return;
}
  if (!scene) return;

  if (!feedbackMarkers.upper) {
    feedbackMarkers.upper = createFeedbackMarker("upper");
  }
  if (!feedbackMarkers.leftLeg) {
    feedbackMarkers.leftLeg = createFeedbackMarker("leftLeg");
  }
  if (!feedbackMarkers.rightLeg) {
    feedbackMarkers.rightLeg = createFeedbackMarker("rightLeg");
  }
  if (!feedbackMarkers.knee) {
    feedbackMarkers.knee = createFeedbackMarker("knee");
  }

  if (feedbackMarkers.upper.parent !== scene) scene.add(feedbackMarkers.upper);
  if (feedbackMarkers.leftLeg.parent !== scene) scene.add(feedbackMarkers.leftLeg);
  if (feedbackMarkers.rightLeg.parent !== scene) scene.add(feedbackMarkers.rightLeg);
  if (feedbackMarkers.knee.parent !== scene) scene.add(feedbackMarkers.knee);
}

function getFeedbackMarkerColor(stateName) {
  if (stateName === "bad") return 0xff5a67;
  if (stateName === "warn") return 0xffb020;
  if (stateName === "missing") return 0x39a0ff;
  return 0x21e17b;
}

function applyMarkerStyle(marker, stateName) {
  if (!marker) return;

  const show =
    stateName === "warn" ||
    stateName === "bad" ||
    stateName === "missing";

  marker.visible = show;
  if (!show) return;

  const dot = marker.userData.dot;
  const color = getFeedbackMarkerColor(stateName);

  if (dot?.material) {
    dot.material.color.setHex(color);

    // 깜빡임 (부드럽게)
    const blink = (Math.sin(performance.now() * 0.01) + 1) * 0.5;
    dot.material.opacity = 0.4 + blink * 0.6;
  }
}

function updateAvatarFeedbackMarkers() {
  if (state.currentMode !== "avatar" && state.currentMode !== "retarget") {
    hideAvatarFeedbackMarkers();
    return;
  }

  if (!state.avatarMarkerEnabled) {
    hideAvatarFeedbackMarkers();
    return;
  }

  if (!avatarScene || !camera) return;

  ensureFeedbackMarkers();

  const upperState = state.partFeedback.upper || "normal";
  const leftLegState = state.partFeedback.leftLeg || "normal";
  const rightLegState = state.partFeedback.rightLeg || "normal";
  const kneeState = state.partFeedback.knee || "normal";

  applyMarkerStyle(feedbackMarkers.upper, upperState);
  applyMarkerStyle(feedbackMarkers.leftLeg, leftLegState);
  applyMarkerStyle(feedbackMarkers.rightLeg, rightLegState);
  applyMarkerStyle(feedbackMarkers.knee, kneeState);

  const base = AVATAR_POS;

  // 상체: 가슴 중앙 쪽
  if (feedbackMarkers.upper) {
    feedbackMarkers.upper.position.set(
      base.x,
      base.y + 1.08,
      base.z + 0.08
    );
  }

  // 왼쪽 다리: 왼쪽 무릎/허벅지 쪽
  if (feedbackMarkers.leftLeg) {
    feedbackMarkers.leftLeg.position.set(
      base.x - 0.18,
      base.y + 0.50,
      base.z + 0.08
    );
  }

  // 오른쪽 다리: 오른쪽 무릎/허벅지 쪽
  if (feedbackMarkers.rightLeg) {
    feedbackMarkers.rightLeg.position.set(
      base.x + 0.18,
      base.y + 0.50,
      base.z + 0.08
    );
  }

  // 무릎: 중앙 말고 양쪽 무릎 사이보다 살짝 아래/앞
  // 너무 가운데 사타구니처럼 보이지 않게 y를 낮추고 z를 앞으로 뺌
  if (feedbackMarkers.knee) {
    feedbackMarkers.knee.position.set(
      base.x,
      base.y + 0.42,
      base.z + 0.10
    );
  }

  const markers = [
    feedbackMarkers.upper,
    feedbackMarkers.leftLeg,
    feedbackMarkers.rightLeg,
    feedbackMarkers.knee,
  ];

  for (const marker of markers) {
    if (!marker || !marker.visible) continue;

    marker.scale.setScalar(1);

    const dot = marker.userData?.dot;
    if (dot) {
      dot.scale.setScalar(1);
      dot.renderOrder = 999;
    }

    marker.lookAt(camera.position);
  }
}

function hideAvatarFeedbackMarkers() {
  Object.values(feedbackMarkers).forEach((marker) => {
    if (marker) marker.visible = false;
  });
}

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
  if (gymRoot?.parent) {
    gymRoot.parent.remove(gymRoot);
  }
  gymRoot = null;

  if (gymImageBg?.parent) {
    gymImageBg.parent.remove(gymImageBg);
  }
  gymImageBg = null;

  const selectedGym = getSelectedGymInfo();

  if (selectedGym.type === "image") {
  const texture = await new THREE.TextureLoader().loadAsync(selectedGym.url);
  texture.colorSpace = THREE.SRGBColorSpace;

  // 구(360 배경)
  const geo = new THREE.SphereGeometry(50, 64, 64);

  const mat = new THREE.MeshBasicMaterial({
    map: texture,
    side: THREE.BackSide, // 안쪽에서 보이게
  });

  if (gymImageBg) {
    scene.remove(gymImageBg);
    gymImageBg.geometry.dispose();
    gymImageBg.material.dispose();
  }

  gymImageBg = new THREE.Mesh(geo, mat);
  gymImageBg.name = "image-gym-360";

  scene.add(gymImageBg);

  console.log("360 배경 적용:", selectedGym.name);
  return;
}

  await new Promise((resolve, reject) => {
    loader.load(
      selectedGym.url,
      (gltf) => {
        gymRoot = gltf.scene;
        gymRoot.traverse((obj) => {
          if (obj.isMesh && obj.material) obj.material.side = THREE.DoubleSide;
        });
        scene.add(gymRoot);
        console.log("헬스장 GLB 로드 완료:", selectedGym.name, selectedGym.url);
        resolve();
      },
      undefined,
      reject
    );
  });
}

async function loadAnimatedAvatar(loader) {
  if (avatarScene?.parent) {
    avatarScene.parent.remove(avatarScene);
  }

  avatarScene = null;
  mixer = null;
  squatClip = null;
  squatAction = null;

  const selectedAvatar =
  state.currentMode === "retarget"
    ? {
        id: "retarget",
        name: state.retargetAvatarType || "Mixamo",
        url: `/models/${state.retargetAvatarUrl || "model.glb"}`,
      }
    : getSelectedAvatarInfo();

  try {
    await new Promise((resolve, reject) => {
      loader.load(
        selectedAvatar.url,
        (gltf) => {
          avatarScene = gltf.scene;
          avatarScene.position.copy(AVATAR_POS);
          scene.add(avatarScene);

          if (mediapipeAvatarManager && state.currentMode === "retarget") {
  mediapipeAvatarManager.bindAvatar(
    avatarScene,
    state.retargetAvatarType || "Mixamo"
  );

  mediapipeAvatarManager.setUseHand?.(false);
  mediapipeAvatarManager.setSlerpRatio?.(0.35);
  mediapipeAvatarManager.initKalmanFilter?.();
  mediapipeAvatarManager.setUseKalmanFilter?.(true);
}

          const clips = gltf.animations || [];
          if (clips.length > 0) {
            mixer = new THREE.AnimationMixer(avatarScene);
            squatClip = clips[0];
            squatAction = mixer.clipAction(squatClip);
            squatAction.enabled = true;
            if (state.currentMode === "standard") {
  // ✅ 정석 모드 → 무한 반복
  squatAction.setLoop(THREE.LoopRepeat, Infinity);
  squatAction.clampWhenFinished = false;
} else {
  // 기존 유지
  squatAction.setLoop(THREE.LoopOnce, 1);
  squatAction.clampWhenFinished = true;
}
            squatAction.play();
            squatAction.paused = true;

            const { clipStart } = getClipRange();
            squatAction.time = clipStart;
            mixer.update(0);
            avatarScene.updateMatrixWorld(true);
          }

          console.log("아바타 로드 완료:", selectedAvatar.name, selectedAvatar.url);
          resolve();
        },
        undefined,
        reject
      );
    });
  } catch (err) {
    console.warn("AVATAR LOAD FAIL:", selectedAvatar.url, err);

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

  ensureFeedbackMarkers();
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
  if (!camera || !avatarScene) return;

  const isAvatarMain =
    (state.currentMode === "avatar" && state.avatarDisplay === "avatar") ||
    (state.currentMode === "retarget" && state.retargetDisplay === "avatar");

  if (!isAvatarMain) return;
  if (!state.cameraRotateEnabled) return;

  if (state.handRotateState === "LEFT") {
    state.handCurrentYaw -= CAMERA_ROTATE_STEP;
  } else if (state.handRotateState === "RIGHT") {
    state.handCurrentYaw += CAMERA_ROTATE_STEP;
  } else {
    return;
  }

  state.handCurrentYaw = clamp(
    state.handCurrentYaw,
    -Math.PI * 0.95,
    Math.PI * 0.95
  );

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
    controls.update();
  }
}

function animateThree(now = 0) {
  if (state.stopped) return;

  renderRafId = requestAnimationFrame(animateThree);

  applyHandCameraControl();

  if (controls && controls.enabled) {
    controls.update();
  }

  if (state.currentMode !== "retarget") {
  updateProgressControlledAnimation();
}

if (state.currentMode === "retarget" && avatarScene) {
  avatarScene.position.copy(AVATAR_POS);
}
  updateAvatarFeedbackMarkers();

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
  // ✅ 정석 모드에서는 자동 회전 UI 숨김
dom.cameraRotateToggle?.closest("label")?.classList.add("hidden");
  if (state.currentMode !== "standard") return;

  const showAvatarMain = state.standardDisplay === "avatarMain";

  if (dom.video) dom.video.style.display = "none";
  if (dom.webcamStage) dom.webcamStage.style.display = "none";
  if (dom.guideStage) dom.guideStage.style.display = "none";
  if (dom.threeWrap) dom.threeWrap.style.display = "none";

  dom.guideStage?.classList.remove("stage-main", "stage-pip");
  dom.threeWrap?.classList.remove("stage-main", "stage-pip");

  showCounter(false);
  clearGuideOverlay();

  if (showAvatarMain) {
    dom.threeWrap.style.display = "block";
    dom.threeWrap.classList.add("stage-main");

    if (state.standardPipVisible) {
      dom.guideStage.style.display = "block";
      dom.guideStage.classList.add("stage-pip");
      setLayerOrder(dom.threeWrap, dom.guideStage);
    } else {
      setLayerOrder(dom.threeWrap, null);
    }

    dom.modeChip.textContent = "정석 모드 · 아바타 메인";
    setHelp("작은 비디오 더블클릭 시 비디오/아바타 위치 전환");
  } else {
    dom.guideStage.style.display = "block";
    dom.guideStage.classList.add("stage-main");

    if (state.standardPipVisible) {
      dom.threeWrap.style.display = "block";
      dom.threeWrap.classList.add("stage-pip");
      setLayerOrder(dom.guideStage, dom.threeWrap);
    } else {
      setLayerOrder(dom.guideStage, null);
    }

    dom.modeChip.textContent = "정석 모드 · 비디오 메인";
    setHelp("작은 아바타 더블클릭 시 비디오/아바타 위치 전환");
  }

  syncPipSettingsUI();

  requestAnimationFrame(() => {
    resizeThreeRenderer();
    resizeGuideOverlay();
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

  const showVideoMain = state.feedbackDisplay === "guide";

  if (dom.video) dom.video.style.display = "none";
  if (dom.webcamStage) dom.webcamStage.style.display = "none";
  if (dom.guideStage) dom.guideStage.style.display = "none";
  if (dom.threeWrap) dom.threeWrap.style.display = "none";

  dom.webcamStage?.classList.remove("stage-main", "stage-pip");
  dom.guideStage?.classList.remove("stage-main", "stage-pip");

  showCounter(false);
  state.lastGuideOverlayKey = "";

  if (showVideoMain) {
    dom.guideStage.style.display = "block";
    dom.guideStage.classList.add("stage-main");

    if (state.feedbackPipVisible) {
      dom.webcamStage.style.display = "block";
      dom.webcamStage.classList.add("stage-pip");
      setLayerOrder(dom.guideStage, dom.webcamStage);
    } else {
      setLayerOrder(dom.guideStage, null);
    }

    dom.modeChip.textContent = "피드백 모드 · 비디오 메인";
    setHelp("주먹 인식 시 video / webcam 위치 전환");
  } else {
    dom.webcamStage.style.display = "block";
    dom.webcamStage.classList.add("stage-main");

    if (state.feedbackPipVisible) {
      dom.guideStage.style.display = "block";
      dom.guideStage.classList.add("stage-pip");
      setLayerOrder(dom.webcamStage, dom.guideStage);
    } else {
      setLayerOrder(dom.webcamStage, null);
    }

    dom.modeChip.textContent = "피드백 모드 · 웹캠 메인";
    setHelp("주먹 인식 시 webcam / video 위치 전환");
  }

  resizeGuideOverlay();
  syncPipSettingsUI();
}

function applyAvatarDisplayMode() {
  // ✅ 다른 모드에서는 다시 표시
dom.cameraRotateToggle?.closest("label")?.classList.remove("hidden");
  if (state.currentMode !== "avatar") return;

  const showAvatarMain = state.avatarDisplay === "avatar";

  if (dom.video) dom.video.style.display = "none";
  if (dom.webcamStage) dom.webcamStage.style.display = "none";
  if (dom.guideStage) dom.guideStage.style.display = "none";
  if (dom.threeWrap) dom.threeWrap.style.display = "none";

  dom.webcamStage?.classList.remove("stage-main", "stage-pip");
  dom.threeWrap?.classList.remove("stage-main", "stage-pip");

  clearGuideOverlay();

  if (showAvatarMain) {
    dom.threeWrap.style.display = "block";
    dom.threeWrap.classList.add("stage-main");

    if (state.avatarPipVisible) {
      dom.webcamStage.style.display = "block";
      dom.webcamStage.classList.add("stage-pip");
      setLayerOrder(dom.threeWrap, dom.webcamStage);
    } else {
      setLayerOrder(dom.threeWrap, null);
    }

    dom.modeChip.textContent = "아바타 모드 · 아바타 메인";
    setHelp("주먹 인식 시 webcam 전환 / 손날 회전으로 카메라 회전");
  } else {
    dom.webcamStage.style.display = "block";
    dom.webcamStage.classList.add("stage-main");

    if (state.avatarPipVisible) {
      dom.threeWrap.style.display = "block";
      dom.threeWrap.classList.add("stage-pip");
      setLayerOrder(dom.webcamStage, dom.threeWrap);
    } else {
      setLayerOrder(dom.webcamStage, null);
    }

    dom.modeChip.textContent = "아바타 모드 · 웹캠 메인";
    setHelp("주먹 인식 시 avatar 전환");
  }

  showCounter(true);

  syncPipSettingsUI();

  requestAnimationFrame(() => {
    resizeThreeRenderer();
  });
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

    updateWorkoutRuntimeStatus();
    drawWebcamFrame();

    if (webcamPose && state.feedbackWebcamOverlayEnabled) {
      drawSkeleton2D(webcamPose);
    } else {
      clearOverlay();
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

    updateWorkoutRuntimeStatus();
    drawWebcamFrame();

    if (webcamPose && state.avatarOverlayEnabled) {
  drawSkeleton2D(webcamPose);
  drawHandOverlay(dom.overlay.width, dom.overlay.height);
} else {
  clearOverlay();
}

clearGuideOverlay();
  }, DRAW_INTERVAL);
}

/* =========================
 * MODES
 * ========================= */
function syncFeedbackGuideOverlay(now, metadata) {
  if (state.stopped || state.currentMode !== "feedback") return;
  if (!dom.guideVideo || dom.guideVideo.readyState < 2) return;

  const mediaTime = Number.isFinite(metadata?.mediaTime)
    ? metadata.mediaTime
    : (dom.guideVideo.currentTime || 0);

  const guideTime = Math.max(0, mediaTime + JSON_TIME_OFFSET_SEC);
  const guideFrameIndex = Math.max(0, Math.floor(guideTime * state.poseFPS));
  const guideFrame = getPoseFrameAtTimeSec(guideTime);

  drawGuidePoseOverlay(guideFrame, guideFrameIndex);

  if (
    typeof dom.guideVideo.requestVideoFrameCallback === "function" &&
    !state.stopped &&
    state.currentMode === "feedback"
  ) {
    state.jsonVideoFrameHandle =
      dom.guideVideo.requestVideoFrameCallback(syncFeedbackGuideOverlay);
  }
}

function syncStandardToVideoFrame(now, metadata) {
  if (state.stopped || state.currentMode !== "standard") return;
  if (!dom.guideVideo || dom.guideVideo.readyState < 2) return;

  const mediaTime =
    Number.isFinite(metadata?.mediaTime)
      ? metadata.mediaTime
      : (dom.guideVideo.currentTime || 0);

  const t = Math.max(0, mediaTime + JSON_TIME_OFFSET_SEC);
  const frame = getPoseFrameAtTimeSec(t);
  const depth = computeDepthFromFrame(frame);

  state.squatProgressRaw = depth;
  state.squatProgressSmooth = depth;

  drawGuideVideoFrameOnly();

  if (
    typeof dom.guideVideo.requestVideoFrameCallback === "function" &&
    !state.stopped &&
    state.currentMode === "standard"
  ) {
    state.jsonVideoFrameHandle =
      dom.guideVideo.requestVideoFrameCallback(syncStandardToVideoFrame);
  }
}

async function startStandardMode() {
  await destroyCurrentMode();

  syncWorkoutConfigFromInputs();
  resetWorkoutSession();

  clearFeedbackToasts();
  state.feedbackToastCooldowns = {};
  state.lastFeedbackSnapshot = "";
  state.lastGuideOverlayKey = "";

  state.currentMode = "standard";
  updateCleanVideoUi();
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
  setHelp("아바타 + 비디오 보기 전용");

  dom.manualToggleBtn.style.display = "none";
  dom.cameraRotateToggle.checked = false;
  state.cameraRotateEnabled = false;
  dom.recordAvatarBtn?.classList.add("hidden");
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

  updatePipSettingsButtonVisibility();
  hidePipSettings();
  applyStandardDisplayMode();

  requestAnimationFrame(() => {
  resizeThreeRenderer();
  resizeGuideOverlay();
  drawGuideVideoFrameOnly();
});

  if (typeof dom.guideVideo.requestVideoFrameCallback === "function") {
    state.jsonVideoFrameHandle =
      dom.guideVideo.requestVideoFrameCallback(syncStandardToVideoFrame);
  } else {
    state.jsonTimer = setInterval(() => {
      if (state.stopped) return;

      const t = Math.max(0, (dom.guideVideo.currentTime || 0) + JSON_TIME_OFFSET_SEC);
      const frame = getPoseFrameAtTimeSec(t);
      const depth = computeDepthFromFrame(frame);

      state.squatProgressRaw = depth;
      state.squatProgressSmooth = depth;
      clearGuideOverlay();
    }, 1000 / Math.max(1, state.poseFPS || 30));
  }

  animateThree();
  setStatus("READY");
}

async function startFeedbackMode() {
  await destroyCurrentMode();

  syncWorkoutConfigFromInputs();
  resetWorkoutSession();

  clearFeedbackToasts();
  state.feedbackToastCooldowns = {};
  state.lastFeedbackSnapshot = "";
  state.lastGuideOverlayKey = "";

  state.currentMode = "feedback";
  if (dom.cameraRotateToggle) {
  dom.cameraRotateToggle.checked = false;
  dom.cameraRotateToggle.closest("label")?.classList.add("hidden");
}
state.cameraRotateEnabled = false;
  dom.statusChip.textContent = "";
dom.statusChip.classList.add("hidden");
  state.stopped = false;
  state.feedbackDisplay = "guide";
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
  state.feedbackWebcamOverlayEnabled = true;
  state.feedbackGuideOverlayEnabled = true;

  showAppScreen();
  resetViewVisibility();
  showCounter(false);
  updateCounterChip(true);

  dom.modeChip.textContent = "피드백 모드";
  dom.gestureChip.textContent = "GESTURE: -";
  setFeedbackText("LOADING...");

  dom.manualToggleBtn.style.display = "inline-flex";
  dom.recordAvatarBtn?.classList.remove("hidden");
  dom.cameraRotateToggle.checked = false;
  state.cameraRotateEnabled = false;
  dom.statusChip.textContent = "READY";

  await startWebcam();
  await startGuideVideo();
  await loadPoseJson();
  await loadPoseModel();
  await loadHandsModel();

  if (controls) {
    controls.enabled = true;
  }

  updatePipSettingsButtonVisibility();
  hidePipSettings();
  applyFeedbackDisplayMode();
  startFeedbackDrawLoop();
  requestAnimationFrame(() => {
  resizeGuideOverlay();
  drawGuideVideoFrameOnly();

  const guideTime = Math.max(
    0,
    (dom.guideVideo.currentTime || 0) + JSON_TIME_OFFSET_SEC
  );
  const guideFrameIndex = Math.max(0, Math.floor(guideTime * state.poseFPS));
  const guideFrame = getPoseFrameAtTimeSec(guideTime);

  drawGuidePoseOverlay(guideFrame, guideFrameIndex);
});

// rVFC (가능하면 사용)
if (typeof dom.guideVideo.requestVideoFrameCallback === "function") {
  state.jsonVideoFrameHandle =
    dom.guideVideo.requestVideoFrameCallback(syncFeedbackGuideOverlay);
}

// 🔥 항상 돌아가는 안전 루프 (핵심)
state.jsonTimer = setInterval(() => {
  if (state.stopped || state.currentMode !== "feedback") return;
  if (!dom.guideVideo || dom.guideVideo.readyState < 2) return;

  const guideTime = Math.max(
    0,
    (dom.guideVideo.currentTime || 0) + JSON_TIME_OFFSET_SEC
  );
  const guideFrameIndex = Math.max(0, Math.floor(guideTime * state.poseFPS));
  const guideFrame = getPoseFrameAtTimeSec(guideTime);

  drawGuidePoseOverlay(guideFrame, guideFrameIndex);
}, 1000 / Math.max(1, state.poseFPS || 30));

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

  syncWorkoutConfigFromInputs();
  resetWorkoutSession();

  clearFeedbackToasts();
  state.feedbackToastCooldowns = {};
  state.lastFeedbackSnapshot = "";
  state.lastGuideOverlayKey = "";

  state.currentMode = "avatar";
  dom.cameraRotateToggle?.closest("label")?.classList.remove("hidden");
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
  state.avatarFistGestureEnabled = true;
  state.avatarKnifeGestureEnabled = true;
  state.avatarOverlayEnabled = true;
  state.avatarMarkerEnabled = true;
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
  dom.recordAvatarBtn?.classList.remove("hidden");
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
  controls.enabled = true;
  controls.enableRotate = true;
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.enableDamping = true;
  controls.target.copy(getAvatarLookTarget());
  controls.update();
}

  const loader = new GLTFLoader();
  await loadGym(loader);
  await loadAnimatedAvatar(loader);

  updatePipSettingsButtonVisibility();
  hidePipSettings();
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

  hideAvatarFeedbackMarkers();

Object.values(feedbackMarkers).forEach((marker) => {
  if (marker?.parent) {
    marker.parent.remove(marker);
  }
});

feedbackMarkers = {
  upper: null,
  leftLeg: null,
  rightLeg: null,
  knee: null,
};

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
  clearGuideOverlay();
  clearFeedbackToasts();
  hidePipSettings();

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
  state.lastGuideOverlayKey = "";

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
  hideAvatarFeedbackMarkers();

  if (renderer) {
    renderer.dispose();
    dom.threeWrap.innerHTML = "";
  }

  if (state.recording.active) {
  stopAvatarRecording();
}

if (gymImageBg?.parent) {
  gymImageBg.parent.remove(gymImageBg);
}
gymImageBg = null;

dom.recordAvatarBtn?.classList.add("hidden");

  renderer = null;
  scene = null;
  camera = null;
  controls = null;
  gymRoot = null;
  avatarScene = null;
  mixer = null;
  squatClip = null;
  squatAction = null;

  updatePipSettingsButtonVisibility();
}

/* =========================
 * EVENTS
 * ========================= */
dom.themeToggle?.addEventListener("click", setTheme);

dom.backBtn?.addEventListener("click", async () => {
  await destroyCurrentMode();
  state.currentMode = null;
  resetViewVisibility();
  setStatus("READY");
  setHelp("설정 대기 중");
  dom.gestureChip.textContent = "GESTURE: -";
  hidePipSettings();
  updatePipSettingsButtonVisibility();
  showModeScreen();
  document.body.classList.remove("clean-video-ui");
});

dom.modeCards?.forEach((card) => {
  card.addEventListener("click", async () => {
    const mode = card.dataset.mode;
    if (mode === "standard") await startStandardMode();
    else if (mode === "feedback") await startFeedbackMode();
    else if (mode === "avatar") await startAvatarMode();
    else if (mode === "retarget") await startRetargetMode();
  });
});

dom.modeRecordsBtn?.addEventListener("click", () => {
  window.location.href = "http://localhost:3000/records.html";
});

dom.modeAnalyticsBtn?.addEventListener("click", () => {
  window.location.href = "http://localhost:3000/analytics.html";
});

dom.avatarSelect?.addEventListener("change", (e) => {
  state.selectedAvatarId = e.target.value;
  updateAvatarPreview();
});

dom.restartBtn?.addEventListener("click", async () => {
  if (state.currentMode === "standard") await startStandardMode();
  else if (state.currentMode === "feedback") await startFeedbackMode();
  else if (state.currentMode === "avatar") await startAvatarMode();
  else if (state.currentMode === "retarget") await startRetargetMode();
});

dom.manualToggleBtn?.addEventListener("click", () => {
  if (state.currentMode === "feedback") {
    state.feedbackDisplay = state.feedbackDisplay === "guide" ? "webcam" : "guide";
    clearFeedbackToasts();
    state.lastGuideOverlayKey = "";
    applyFeedbackDisplayMode();
  } else if (state.currentMode === "avatar") {
    state.avatarDisplay = state.avatarDisplay === "avatar" ? "webcam" : "avatar";
    applyAvatarDisplayMode();
  } else if (state.currentMode === "retarget") {
  state.retargetDisplay =
    state.retargetDisplay === "avatar" ? "webcam" : "avatar";
  applyRetargetDisplayMode();
}
});

dom.cameraRotateToggle?.addEventListener("change", (e) => {
  state.cameraRotateEnabled = !!e.target.checked;
});

dom.guideStage?.addEventListener("dblclick", () => {
  if (state.currentMode === "standard") {
    toggleStandardDisplay();
  }
});

dom.threeWrap?.addEventListener("dblclick", () => {
  if (state.currentMode === "standard") {
    toggleStandardDisplay();
  }
});

dom.pipSettingsBtn?.addEventListener("click", (e) => {
  e.stopPropagation();
  togglePipSettings();
});

dom.pipVisibleToggle?.addEventListener("change", (e) => {
  setCurrentPipVisible(!!e.target.checked);
});

/* ✅ 여기 3개를 document click 밖으로 분리 */
dom.avatarFistToggle?.addEventListener("change", (e) => {
  state.avatarFistGestureEnabled = !!e.target.checked;
  if (!state.avatarFistGestureEnabled) {
    state.fistHoldCount = 0;
  }
});

dom.avatarKnifeToggle?.addEventListener("change", (e) => {
  state.avatarKnifeGestureEnabled = !!e.target.checked;
  if (!state.avatarKnifeGestureEnabled) {
    state.handRotateState = "NEUTRAL";
  }
});

dom.avatarOverlayToggle?.addEventListener("change", (e) => {
  state.avatarOverlayEnabled = !!e.target.checked;
  if (!state.avatarOverlayEnabled) {
    clearOverlay();
  }
});

dom.feedbackWebcamOverlayToggle?.addEventListener("change", (e) => {
  state.feedbackWebcamOverlayEnabled = !!e.target.checked;
  if (!state.feedbackWebcamOverlayEnabled) {
    clearOverlay();
  }
});

dom.feedbackGuideOverlayToggle?.addEventListener("change", (e) => {
  state.feedbackGuideOverlayEnabled = !!e.target.checked;
  state.lastGuideOverlayKey = "";
  if (!state.feedbackGuideOverlayEnabled) {
    drawGuideVideoFrameOnly();
  }
});

dom.avatarMarkerToggle?.addEventListener("change", (e) => {
  state.avatarMarkerEnabled = !!e.target.checked;
  if (!state.avatarMarkerEnabled) {
    hideAvatarFeedbackMarkers();
  }
});

document.addEventListener("click", (e) => {
  if (!state.pipSettingsOpen) return;
  if (!dom.pipSettingsPanel || !dom.pipSettingsBtn) return;

  const target = e.target;
  if (
    dom.pipSettingsPanel.contains(target) ||
    dom.pipSettingsBtn.contains(target)
  ) {
    return;
  }

  hidePipSettings();
});

window.addEventListener("resize", () => {
  resizeThreeRenderer();
  resizeGuideOverlay();
  state.lastGuideOverlayKey = "";
});

dom.saveRecordBtn?.addEventListener("click", async () => {
  await saveWorkoutRecord();
});

dom.openRecordsBtn?.addEventListener("click", () => {
  window.location.href = "http://localhost:3000/records.html";
});

dom.openAnalyticsBtn?.addEventListener("click", () => {
  window.location.href = "http://localhost:3000/analytics.html";
});

dom.workoutTypeSelect?.addEventListener("change", () => {
  syncWorkoutConfigFromInputs();
  updateWorkoutSetupUI();
});

dom.setCountInput?.addEventListener("input", syncWorkoutConfigFromInputs);
dom.repCountInput?.addEventListener("input", syncWorkoutConfigFromInputs);
dom.restSecondsInput?.addEventListener("input", syncWorkoutConfigFromInputs);

dom.gymSelect?.addEventListener("change", (e) => {
  state.selectedGymId = e.target.value;
  updateGymPreview();
  console.log("선택된 헬스장:", getSelectedGymInfo());
});

dom.recordAvatarBtn?.addEventListener("click", () => {
  if (state.recording.active) {
    stopAvatarRecording();
  } else {
    startAvatarRecording();
  }
});
dom.openRecordingsBtn?.addEventListener("click", () => {
  window.location.href = "http://localhost:3000/records.html#recordings";
});
dom.retargetAvatarSelect?.addEventListener("change", (e) => {
  const [url, type] = e.target.value.split("|");

  state.retargetAvatarUrl = url;
  state.retargetAvatarType = type;

  console.log("리타게팅 아바타:", {
    url: state.retargetAvatarUrl,
    type: state.retargetAvatarType,
  });
});

dom.retargetVideoSelect?.addEventListener("change", (e) => {
  state.retargetVideoUrl = e.target.value;
  console.log("리타게팅 영상:", state.retargetVideoUrl || "webcam");
});
dom.threeWrap?.addEventListener("dblclick", () => {
  if (state.currentMode !== "retarget") return;
  if (!isRetargetVideoOnly()) return;

  state.retargetDisplay =
    state.retargetDisplay === "avatar" ? "webcam" : "avatar";

  applyRetargetDisplayMode();
});

dom.webcamStage?.addEventListener("dblclick", () => {
  if (state.currentMode !== "retarget") return;
  if (!isRetargetVideoOnly()) return;

  state.retargetDisplay =
    state.retargetDisplay === "avatar" ? "webcam" : "avatar";

  applyRetargetDisplayMode();
});
/* =========================
 * INIT UI
 * ========================= */
showModeScreen();
resetViewVisibility();
setStatus("READY");
setHelp("설정 대기 중");
if (dom.gestureChip) dom.gestureChip.textContent = "GESTURE: -";
if (dom.manualToggleBtn) dom.manualToggleBtn.style.display = "none";
if (dom.studentIdInput) dom.studentIdInput.value = state.studentId;
if (dom.userNameInput) dom.userNameInput.value = state.userName;
if (dom.avatarSelect) dom.avatarSelect.value = state.selectedAvatarId;
if (dom.gymSelect) dom.gymSelect.value = state.selectedGymId;
if (dom.gymSelect) dom.gymSelect.value = state.selectedGymId;
document.getElementById("retargetExerciseSelect")?.addEventListener("change", () => {
  updateCleanVideoUi();
});
updateAvatarPreview();
updateGymPreview();
showCounter(false);
updateCounterChip(true);
clearFeedbackToasts();
clearGuideOverlay();
hidePipSettings();
updateAvatarPreview();
updatePipSettingsButtonVisibility();
syncWorkoutConfigFromInputs();
updateWorkoutSetupUI();