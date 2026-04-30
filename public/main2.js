/* eslint-disable no-console */
import * as THREE from "three";
import * as ort from "onnxruntime-web";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const ORT_DIST_BASE = "/ort/";

ort.env.wasm.wasmPaths = ORT_DIST_BASE;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

/** =========================
 * UI
 * ========================= */
const INITIAL_VIEW_W = 420;
const INITIAL_VIEW_H = 560;
const VIDEO_DRAW_INTERVAL = 33;

const YOLO_MODEL_URL = "/models/yolo_pose.onnx";
const API_BASE = "http://localhost:3000";

const EXERCISE_CONFIG = {
  squat: {
    label: "Squat",
    videoUrl: "/squat.mp4",
    poseJsonUrl: "/pose_squat.json",
    avatarGlbUrl: "/models/Untitled_squat.glb",
    guideText: "스쿼트: JSON / Webcam+YOLO 둘 다 가능",
  },
  pushup: {
    label: "Push-up",
    videoUrl: "/push_up.mp4",
    poseJsonUrl: "/pose_push_up.json",
    avatarGlbUrl: "/models/Untitled_push_up.glb",
    guideText: "푸쉬업: JSON Video + Pose JSON 전용",
  },
};

const GYM_URL = "/models/Untitled_gym.glb";

const app = document.querySelector("#app");

app.innerHTML = `
  <div style="font-family:sans-serif; padding:12px;">
    <h2 style="margin:0 0 8px;">Health-Mate Preview (Squat + Push-up / JSON + Webcam+YOLO + Gym)</h2>

    <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start;">
      <!-- LEFT -->
      <div style="min-width:420px;">
        <div style="margin:10px 0 6px; font-weight:600;">Canvas (Input + Stickman)</div>
        <div id="canvasStage" style="position:relative; width:${INITIAL_VIEW_W}px; height:${INITIAL_VIEW_H}px; border:1px solid #ddd; background:#111; overflow:hidden;">
          <canvas id="canvas" width="${INITIAL_VIEW_W}" height="${INITIAL_VIEW_H}" style="position:absolute; left:0; top:0;"></canvas>
          <canvas id="overlay" width="${INITIAL_VIEW_W}" height="${INITIAL_VIEW_H}" style="position:absolute; left:0; top:0; pointer-events:none;"></canvas>
        </div>

        <div style="margin-top:6px; font-size:12px; color:#666;">
          <span style="display:inline-block;width:10px;height:10px;background:#0f0;margin-right:6px;"></span>pose lines
          &nbsp;&nbsp;
          <span style="display:inline-block;width:10px;height:10px;background:#f00;margin-right:6px;"></span>pose joints
        </div>

        <div style="margin-top:12px; border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
          <div style="font-weight:600; margin-bottom:8px;">운동 종류 설정</div>

          <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin-bottom:10px;">
            <label style="display:flex; gap:6px; align-items:center;">
              <input type="radio" name="exerciseType" value="squat" checked />
              Squat
            </label>
            <label style="display:flex; gap:6px; align-items:center;">
              <input type="radio" name="exerciseType" value="pushup" />
              Push-up
            </label>
          </div>

          <div id="exerciseGuide" style="font-size:13px; color:#666; line-height:1.5;">
            스쿼트: JSON / Webcam+YOLO 둘 다 가능
          </div>
        </div>

        <div style="margin-top:12px; border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
          <div style="font-weight:600; margin-bottom:8px;">입력 모드 설정</div>

          <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin-bottom:10px;">
            <label style="display:flex; gap:6px; align-items:center;">
              <input type="radio" name="inputMode" value="json" checked />
              JSON Video + Pose JSON
            </label>
            <label style="display:flex; gap:6px; align-items:center;">
              <input id="webcamModeRadio" type="radio" name="inputMode" value="webcam" />
              Webcam + YOLO
            </label>
          </div>

          <div id="jsonInputHint" style="font-size:13px; color:#666; line-height:1.5;">
            비디오와 pose json을 동기화해서 분석합니다.
          </div>
          <div id="webcamInputHint" style="display:none; font-size:13px; color:#666; line-height:1.5; margin-top:4px;">
            웹캠 프레임에 대해 YOLO Pose ONNX를 실시간 추론합니다.
          </div>
        </div>

        <div style="margin-top:12px; border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
          <div style="font-weight:600; margin-bottom:8px;">운동 모드 설정</div>

          <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin-bottom:10px;">
            <label style="display:flex; gap:6px; align-items:center;">
              <input type="radio" name="workoutMode" value="normal" checked />
              Normal
            </label>
            <label style="display:flex; gap:6px; align-items:center;">
              <input type="radio" name="workoutMode" value="challenge" />
              Challenge
            </label>
          </div>

          <div id="normalControls" style="display:grid; grid-template-columns:repeat(3, minmax(100px, 1fr)); gap:10px;">
            <label style="display:flex; flex-direction:column; gap:4px; font-size:13px;">
              세트 수
              <input id="targetSets" type="number" min="1" max="20" value="3" style="padding:6px 8px;" />
            </label>

            <label style="display:flex; flex-direction:column; gap:4px; font-size:13px;">
              세트당 횟수
              <input id="setRepTarget" type="number" min="1" max="100" value="10" style="padding:6px 8px;" />
            </label>

            <label style="display:flex; flex-direction:column; gap:4px; font-size:13px;">
              휴식(초)
              <input id="restSeconds" type="number" min="0" max="600" value="30" style="padding:6px 8px;" />
            </label>
          </div>

          <div id="challengeHint" style="display:none; margin-top:8px; font-size:13px; color:#666;">
            Challenge 모드는 사용자가 멈출 때까지 논스탑으로 진행합니다.
          </div>
        </div>

        <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
          <button id="start">Start Input</button>
          <button id="stop">Stop Input</button>
          <button id="loadPose">Load Pose JSON</button>
          <button id="loadYolo">Load YOLO</button>
          <button id="sync">Sync / Infer 1 Frame</button>
          <button id="live">Live ON/OFF</button>
          <button id="clear">Clear</button>
        </div>

        <div style="margin-top:8px; font-size:12px; color:#666; line-height:1.45;">
          Video: <code id="currentVideoPath">/squat.mp4</code><br/>
          Pose JSON: <code id="currentPosePath">/pose_squat.json</code><br/>
          Avatar GLB: <code id="currentAvatarPath">/models/Untitled_squat.glb</code><br/>
          YOLO ONNX: <code>${YOLO_MODEL_URL}</code><br/>
          ORT DIST: <code>${ORT_DIST_BASE}</code><br/>
          Left: <code>video/webcam + stickman</code><br/>
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

        <div id="resultCards" style="display:grid; grid-template-columns:repeat(3, minmax(180px, 1fr)); gap:12px; max-width:920px; margin-top:12px;">
          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">운동 종류</div>
            <div id="cardExercise" style="font-size:24px; font-weight:700; margin-top:6px;">Squat</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">입력 모드</div>
            <div id="cardInputMode" style="font-size:24px; font-weight:700; margin-top:6px;">JSON</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">운동 모드</div>
            <div id="cardMode" style="font-size:24px; font-weight:700; margin-top:6px;">Normal</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">총 횟수</div>
            <div id="cardCount" style="font-size:32px; font-weight:700; margin-top:6px;">0</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">평균 깊이/진행도</div>
            <div id="cardAvgDepth" style="font-size:32px; font-weight:700; margin-top:6px;">0.000</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">현재 세트</div>
            <div id="cardSet" style="font-size:28px; font-weight:700; margin-top:6px;">1 / 3</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">현재 세트 횟수</div>
            <div id="cardSetRep" style="font-size:28px; font-weight:700; margin-top:6px;">0 / 10</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">휴식 타이머</div>
            <div id="cardRest" style="font-size:28px; font-weight:700; margin-top:6px;">-</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">깊이/가동범위 부족 횟수</div>
            <div id="cardDepthLow" style="font-size:32px; font-weight:700; margin-top:6px;">0</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">자세 경고 횟수</div>
            <div id="cardTorsoWarn" style="font-size:32px; font-weight:700; margin-top:6px;">0</div>
          </div>

          <div style="border:1px solid #ddd; border-radius:12px; padding:12px; background:#fff;">
            <div style="font-size:12px; color:#666;">최근 속도 판정</div>
            <div id="cardSpeed" style="font-size:24px; font-weight:700; margin-top:6px;">대기</div>
            <div id="cardRepTime" style="font-size:13px; color:#666; margin-top:4px;">0.00s</div>
          </div>
        </div>

        <div style="margin-top:12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <button id="saveResult">결과 저장</button>
          <span id="saveStatus" style="font-size:13px; color:#666;">저장 대기</span>
        </div>

        <div style="margin-top:12px; border:1px solid #ddd; border-radius:12px; padding:12px; max-width:920px; background:#fff;">
          <div style="font-weight:600; margin-bottom:6px;">최근 저장 기록</div>
          <div id="latestResult" style="font-size:14px; color:#444;">(없음)</div>
        </div>

        <div style="margin-top:12px; border:1px solid #ddd; border-radius:12px; padding:12px; max-width:920px; background:#fff;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:8px;">
            <div style="font-weight:600;">저장 기록 목록</div>
            <div style="display:flex; gap:8px; align-items:center;">
              <label for="historySort" style="font-size:13px; color:#666;">정렬</label>
              <select id="historySort" style="padding:6px 8px; border:1px solid #ccc; border-radius:8px;">
                <option value="latest">최신순</option>
                <option value="oldest">오래된순</option>
                <option value="count_desc">횟수 높은순</option>
                <option value="depth_desc">평균 깊이 높은순</option>
              </select>
            </div>
          </div>
          <div id="historyList" style="font-size:14px; color:#444; line-height:1.6;">(없음)</div>
        </div>

        <div style="margin-top:10px; font-weight:600;">Info</div>
        <pre id="info" style="background:#0f0f10; color:#9ef; padding:10px; border-radius:8px; max-width:920px; overflow:auto; height:240px;">(no data)</pre>

        <div style="margin-top:8px; color:#888; font-size:12px; line-height:1.45;">
          Gym GLB: <code>${GYM_URL}</code><br/>
          Avatar GLB: <code id="infoAvatarPath">/models/Untitled_squat.glb</code><br/>
          JSON mode: <code>pose json depth/progress → animation progress</code><br/>
          Webcam mode: <code>Squat only / YOLO Pose ONNX → depth → animation progress</code>
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

/** extra hidden canvas for YOLO preprocess */
const inferCanvas = document.createElement("canvas");
const inferCtx = inferCanvas.getContext("2d", { willReadFrequently: true });

/** DOM refs */
const canvasStage = document.getElementById("canvasStage");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const infoEl = document.getElementById("info");
const threeStatusEl = document.getElementById("threeStatus");
const autoRotateEl = document.getElementById("autoRotate");

const jsonInputHintEl = document.getElementById("jsonInputHint");
const webcamInputHintEl = document.getElementById("webcamInputHint");

const exerciseGuideEl = document.getElementById("exerciseGuide");
const webcamModeRadioEl = document.getElementById("webcamModeRadio");
const currentVideoPathEl = document.getElementById("currentVideoPath");
const currentPosePathEl = document.getElementById("currentPosePath");
const currentAvatarPathEl = document.getElementById("currentAvatarPath");
const infoAvatarPathEl = document.getElementById("infoAvatarPath");

const cardExerciseEl = document.getElementById("cardExercise");
const cardInputModeEl = document.getElementById("cardInputMode");
const cardModeEl = document.getElementById("cardMode");
const cardCountEl = document.getElementById("cardCount");
const cardAvgDepthEl = document.getElementById("cardAvgDepth");
const cardSetEl = document.getElementById("cardSet");
const cardSetRepEl = document.getElementById("cardSetRep");
const cardRestEl = document.getElementById("cardRest");
const cardDepthLowEl = document.getElementById("cardDepthLow");
const cardTorsoWarnEl = document.getElementById("cardTorsoWarn");
const cardSpeedEl = document.getElementById("cardSpeed");
const cardRepTimeEl = document.getElementById("cardRepTime");

const saveStatusEl = document.getElementById("saveStatus");
const latestResultEl = document.getElementById("latestResult");
const historyListEl = document.getElementById("historyList");
const historySortEl = document.getElementById("historySort");

const normalControlsEl = document.getElementById("normalControls");
const challengeHintEl = document.getElementById("challengeHint");
const targetSetsInput = document.getElementById("targetSets");
const setRepTargetInput = document.getElementById("setRepTarget");
const restSecondsInput = document.getElementById("restSeconds");

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
let webcamReady = false;
let webcamStream = null;

let inputMode = "json"; // json | webcam
let exerciseType = "squat"; // squat | pushup

let poseData = null;
let poseFPS = 30;

let lastPoseKpts = null;
let displayKpts = null;
let overlayAnimId = null;

/** =========================
 * YOLO 상태
 * ========================= */
let yoloSession = null;
let yoloReady = false;
let yoloBusy = false;

const YOLO_INPUT_SIZE = 640;
const YOLO_SCORE_THRESHOLD = 0.25;
const YOLO_KPT_THRESHOLD = 0.20;
const YOLO_IOU_THRESHOLD = 0.45;

/** =========================
 * YOLO 포즈 유지 / 보정 상태
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
 * 진행도 제어 상태
 * ========================= */
let animationPaused = false;
let motionMetricRaw = 0;
let animProgressRaw = 0;
let animProgressSmooth = 0;
let animProgressRender = 0;

const SYNC_INTERVAL_MS = 33;
const PROGRESS_EMA = 0.35;
const PROGRESS_SNAP_LOW = 0.02;
const PROGRESS_SNAP_HIGH = 0.95;

const CLIP_START_NORM = 0.0;
const CLIP_END_NORM = 0.33;

/** squat thresholds */
const SQUAT_DEPTH_UP = 0.18;
const SQUAT_DEPTH_DOWN = 0.52;

/** push-up thresholds */
const PUSHUP_UP = 0.18;
const PUSHUP_DOWN = 0.70;

/** =========================
 * 운동 모드 / 세트 / 휴식 / 속도 상태
 * ========================= */
let workoutMode = "normal";

let currentSet = 1;
let targetSets = 3;
let setRepTarget = 10;
let currentSetCount = 0;

let restSeconds = 30;
let restRemaining = 0;
let isResting = false;
let restTimer = null;

let repStartTime = 0;
let lastRepDuration = 0;
let speedFeedback = "대기";

let setResults = [];

/** =========================
 * 카운트/피드백 상태
 * ========================= */
let repCount = 0;
let motionState = "UP";
let lastMotionMetric = 0;
let lastFormMetric = 1.0;
let lastPostureMetric = 0;
let lastFeedback = "대기 중";

/** rep 단위 판정 기준 */
const DEPTH_LOW_THRESHOLD_SQUAT = 0.40;
const TORSO_WARN_THRESHOLD_SQUAT = 1.10;

const DEPTH_LOW_THRESHOLD_PUSHUP = 0.45;
const BODYLINE_WARN_THRESHOLD_PUSHUP = 0.18;

let repWarnFlag = false;
let repMaxMetric = 0;

/** =========================
 * 세션 통계
 * ========================= */
let analyzedFrameCount = 0;
let metricAccum = 0;
let depthLowCount = 0;
let torsoWarningCount = 0;
let sessionStartTime = 0;
let sessionSaved = false;

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

let renderer = null;
let scene = null;
let camera = null;
let controls = null;

let gymRoot = null;
let avatarScene = null;
let mixer = null;
let exerciseClip = null;
let exerciseAction = null;

let lastMixerTime = -999;

let lastRenderTime = 0;
const RENDER_FPS = 20;
const RENDER_INTERVAL = 1000 / RENDER_FPS;

/** 운동별 아바타 / 카메라 기준 */
const SQUAT_AVATAR_POS = new THREE.Vector3(0, 0, 0.8);
const PUSHUP_AVATAR_POS = new THREE.Vector3(0, 0.02, 0.25);

const SQUAT_LOOK_TARGET_OFFSET = new THREE.Vector3(0, 1.05, 0);
const PUSHUP_LOOK_TARGET_OFFSET = new THREE.Vector3(0, 0.28, 0);

const SQUAT_FRONT_CAM_OFFSET = new THREE.Vector3(0, 0.2, 2.5);
const PUSHUP_FRONT_CAM_OFFSET = new THREE.Vector3(0, 1.15, 2.15);

/** =========================
 * 유틸
 * ========================= */
function getExerciseCfg() {
  return EXERCISE_CONFIG[exerciseType];
}
function getCurrentVideoUrl() {
  return getExerciseCfg().videoUrl;
}
function getCurrentPoseJsonUrl() {
  return getExerciseCfg().poseJsonUrl;
}
function getCurrentAvatarGlbUrl() {
  return getExerciseCfg().avatarGlbUrl;
}
function getAvatarBasePosition() {
  return exerciseType === "pushup"
    ? PUSHUP_AVATAR_POS.clone()
    : SQUAT_AVATAR_POS.clone();
}
function getLookTargetOffset() {
  return exerciseType === "pushup"
    ? PUSHUP_LOOK_TARGET_OFFSET.clone()
    : SQUAT_LOOK_TARGET_OFFSET.clone();
}
function getFrontCameraOffset() {
  return exerciseType === "pushup"
    ? PUSHUP_FRONT_CAM_OFFSET.clone()
    : SQUAT_FRONT_CAM_OFFSET.clone();
}
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
function mapMotionToProgress(metric) {
  if (exerciseType === "pushup") {
    const p = (metric - PUSHUP_UP) / Math.max(1e-6, PUSHUP_DOWN - PUSHUP_UP);
    return clamp01(p);
  }
  const p = (metric - SQUAT_DEPTH_UP) / Math.max(1e-6, SQUAT_DEPTH_DOWN - SQUAT_DEPTH_UP);
  return clamp01(p);
}
function postProcessProgress(p) {
  let out = p;
  if (out < PROGRESS_SNAP_LOW) out = 0;
  if (out > PROGRESS_SNAP_HIGH) out = 1;
  return out;
}
function getClipRange() {
  if (!exerciseClip) {
    return { clipStart: 0, clipEnd: 0, usableDuration: 0 };
  }
  const clipDuration = Math.max(0.0001, exerciseClip.duration);
  const clipStart = clipDuration * CLIP_START_NORM;
  const clipEnd = clipDuration * CLIP_END_NORM;
  const usableDuration = Math.max(0.0001, clipEnd - clipStart);
  return { clipStart, clipEnd, usableDuration };
}
function resetAvatarPoseToStart() {
  if (exerciseAction && mixer && exerciseClip) {
    const { clipStart } = getClipRange();
    exerciseAction.time = clipStart;
    mixer.update(0);
    avatarScene?.updateMatrixWorld(true);
    lastMixerTime = clipStart;
  }
}
function getCurrentSortValue() {
  return historySortEl?.value || "latest";
}
function getModeLabel() {
  return workoutMode === "normal" ? "Normal" : "Challenge";
}
function getInputModeLabel() {
  return inputMode === "json" ? "JSON" : "Webcam";
}
function getExerciseLabel() {
  return getExerciseCfg().label;
}
function getNowSec() {
  if (inputMode === "json" && videoReady) return video.currentTime;
  return performance.now() / 1000;
}
function countVisibleJoints(kpts, th = 0.05) {
  if (!kpts) return 0;
  let n = 0;
  for (const p of kpts) {
    if (p && (p[2] ?? 0) >= th) n += 1;
  }
  return n;
}
function cloneKpts(kpts) {
  return kpts ? kpts.map((p) => [...p]) : null;
}
function carryForwardMissingJoints(currKpts, prevKpts, confTh = 0.18) {
  if (!currKpts && !prevKpts) return null;
  if (!currKpts) return prevKpts ? prevKpts.map((p) => [...p]) : null;
  if (!prevKpts) return currKpts.map((p) => [...p]);

  const out = currKpts.map((p, i) => {
    const cur = p ? [...p] : null;
    const prev = prevKpts[i] ? [...prevKpts[i]] : null;

    if (!cur && prev) return prev;
    if (!cur) return [0, 0, 0];

    if ((cur[2] ?? 0) < confTh && prev) {
      return [prev[0], prev[1], Math.max(cur[2] ?? 0, prev[2] ?? 0.4)];
    }

    return cur;
  });

  return out;
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
      const shoulderOrder = ls[0] <= rs[0];
      const hipOrder = lh[0] <= rh[0];

      if (!(shoulderOrder && hipOrder)) {
        const leftMeanX =
          ((out[5]?.[0] ?? 0) + (out[11]?.[0] ?? 0) + (out[13]?.[0] ?? 0)) / 3;
        const rightMeanX =
          ((out[6]?.[0] ?? 0) + (out[12]?.[0] ?? 0) + (out[14]?.[0] ?? 0)) / 3;

        if (leftMeanX > rightMeanX) swapWholeSide(out);
      }
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

  if (shoulderOrderOk !== hipOrderOk) {
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

  return out;
}
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

    if (upperSet.has(i)) {
      out.push(blendJoint(prev, curr, upperAlpha));
    } else if (lowerSet.has(i)) {
      out.push(blendJoint(prev, curr, lowerAlpha));
    } else {
      out.push(blendJoint(prev, curr, (upperAlpha + lowerAlpha) * 0.5));
    }
  }
  return out;
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

    const widthCost =
      Math.abs(sw - prevSW) * 1.5 +
      Math.abs(hw - prevHW) * 1.5;

    const visiblePenalty = Math.max(0, 10 - countVisibleJoints(kpts, 0.05)) * 0.03;
    const scoreBonus = (1 - cand.score) * 0.08;

    const totalCost = centerCost * 2.5 + widthCost + visiblePenalty + scoreBonus;

    if (totalCost < bestCost) {
      bestCost = totalCost;
      best = {
        score: cand.score,
        keypoints: kpts,
      };
    }
  }

  return best;
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
function refreshExercisePathTexts() {
  currentVideoPathEl.textContent = getCurrentVideoUrl();
  currentPosePathEl.textContent = getCurrentPoseJsonUrl();
  currentAvatarPathEl.textContent = getCurrentAvatarGlbUrl();
  infoAvatarPathEl.textContent = getCurrentAvatarGlbUrl();
  exerciseGuideEl.textContent = getExerciseCfg().guideText;
}
function applyExerciseRestrictions() {
  if (exerciseType === "pushup") {
    if (webcamModeRadioEl.checked) {
      const jsonRadio = document.querySelector('input[name="inputMode"][value="json"]');
      if (jsonRadio) jsonRadio.checked = true;
      inputMode = "json";
    }
    webcamModeRadioEl.disabled = true;
    webcamInputHintEl.style.display = "none";
  } else {
    webcamModeRadioEl.disabled = false;
  }
}
function readExerciseType() {
  exerciseType =
    document.querySelector('input[name="exerciseType"]:checked')?.value || "squat";

  applyExerciseRestrictions();
  refreshExercisePathTexts();
  updateResultCards();
}
function readWorkoutInputs() {
  workoutMode =
    document.querySelector('input[name="workoutMode"]:checked')?.value || "normal";

  targetSets = Math.max(1, Number(targetSetsInput.value || 3));
  setRepTarget = Math.max(1, Number(setRepTargetInput.value || 10));
  restSeconds = Math.max(0, Number(restSecondsInput.value || 30));

  normalControlsEl.style.display = workoutMode === "normal" ? "grid" : "none";
  challengeHintEl.style.display = workoutMode === "challenge" ? "block" : "none";

  updateResultCards();
}
function readInputMode() {
  inputMode =
    document.querySelector('input[name="inputMode"]:checked')?.value || "json";

  if (exerciseType === "pushup" && inputMode === "webcam") {
    inputMode = "json";
    const jsonRadio = document.querySelector('input[name="inputMode"][value="json"]');
    if (jsonRadio) jsonRadio.checked = true;
  }

  jsonInputHintEl.style.display = inputMode === "json" ? "block" : "none";
  webcamInputHintEl.style.display = inputMode === "webcam" ? "block" : "none";

  updateResultCards();
}
function stopRestTimer() {
  if (restTimer) {
    clearInterval(restTimer);
    restTimer = null;
  }
  isResting = false;
  restRemaining = 0;
}
function startRestTimer() {
  stopRestTimer();

  isResting = true;
  restRemaining = restSeconds;
  updateResultCards();
  setThreeStatus(`휴식 중... ${restRemaining}초`);

  restTimer = setInterval(() => {
    restRemaining -= 1;
    updateResultCards();

    if (restRemaining > 0) {
      setThreeStatus(`휴식 중... ${restRemaining}초`);
      return;
    }

    stopRestTimer();
    setThreeStatus(`${currentSet}세트 시작`);
    updateResultCards();
  }, 1000);
}
function finishCurrentSet() {
  const avgMetric = analyzedFrameCount > 0 ? metricAccum / analyzedFrameCount : 0;

  setResults.push({
    setNumber: currentSet,
    reps: currentSetCount,
    avgMetric,
    depthLowCount,
    torsoWarningCount,
    exerciseType,
  });

  if (currentSet >= targetSets) {
    stopRestTimer();
    setThreeStatus("모든 세트 완료");
    setLive(false);
    updateResultCards();
    return;
  }

  currentSet += 1;
  currentSetCount = 0;
  startRestTimer();
  updateResultCards();
}
function updateResultCards() {
  const avgMetric = analyzedFrameCount > 0 ? metricAccum / analyzedFrameCount : 0;

  cardExerciseEl.textContent = getExerciseLabel();
  cardInputModeEl.textContent = getInputModeLabel();
  cardModeEl.textContent = getModeLabel();
  cardCountEl.textContent = String(repCount);
  cardAvgDepthEl.textContent = avgMetric.toFixed(3);
  cardDepthLowEl.textContent = String(depthLowCount);
  cardTorsoWarnEl.textContent = String(torsoWarningCount);

  if (workoutMode === "normal") {
    cardSetEl.textContent = `${currentSet} / ${targetSets}`;
    cardSetRepEl.textContent = `${currentSetCount} / ${setRepTarget}`;
    cardRestEl.textContent = isResting ? `${restRemaining}s` : "-";
  } else {
    cardSetEl.textContent = "-";
    cardSetRepEl.textContent = "-";
    cardRestEl.textContent = "-";
  }

  cardSpeedEl.textContent = speedFeedback;
  cardRepTimeEl.textContent = `${lastRepDuration.toFixed(2)}s`;
}
function resetSessionStats() {
  repCount = 0;
  motionState = "UP";
  lastMotionMetric = 0;
  lastFormMetric = 1.0;
  lastPostureMetric = 0;
  lastFeedback = "대기 중";

  motionMetricRaw = 0;
  animProgressRaw = 0;
  animProgressSmooth = 0;
  animProgressRender = 0;

  analyzedFrameCount = 0;
  metricAccum = 0;
  depthLowCount = 0;
  torsoWarningCount = 0;
  sessionSaved = false;
  sessionStartTime = getNowSec();

  currentSet = 1;
  currentSetCount = 0;
  setResults = [];

  repWarnFlag = false;
  repMaxMetric = 0;

  repStartTime = 0;
  lastRepDuration = 0;
  speedFeedback = "대기";

  stopRestTimer();

  lastStablePoseKpts = null;
  lastStablePoseTime = 0;
  lostPoseFrames = 0;

  updateResultCards();
  saveStatusEl.textContent = "저장 대기";
}
function buildSessionPayload() {
  const avgMetric = analyzedFrameCount > 0 ? metricAccum / analyzedFrameCount : 0;
  const durationSec = Math.max(0, getNowSec() - (sessionStartTime ?? 0));

  return {
    sessionName: `${exerciseType}_${inputMode}_${workoutMode}_${new Date().toISOString()}`,
    exerciseType,
    totalCount: repCount,
    avgDepth: avgMetric,
    depthLowCount,
    torsoWarningCount,
    durationSec,
    inputMode,
    mode: workoutMode,
    targetSets,
    setRepTarget,
    restSeconds,
    setResults,
  };
}

/** =========================
 * API
 * ========================= */
async function saveSessionResult() {
  try {
    const payload = buildSessionPayload();

    if (payload.totalCount <= 0 && payload.durationSec <= 0) {
      saveStatusEl.textContent = "저장할 결과 없음";
      return;
    }

    saveStatusEl.textContent = "저장 중...";

    const res = await fetch(`${API_BASE}/api/sessions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!data.ok) {
      throw new Error(data.message || "저장 실패");
    }

    sessionSaved = true;
    saveStatusEl.textContent = "저장 완료";
    await loadLatestResult();
    await loadSessionHistory();
  } catch (err) {
    console.error(err);
    saveStatusEl.textContent = "저장 실패";
    logInfo("Save error: " + err.message);
  }
}
async function loadLatestResult() {
  try {
    const res = await fetch(`${API_BASE}/api/sessions/latest`);
    const data = await res.json();

    if (!data.ok || !data.row) {
      latestResultEl.textContent = "(없음)";
      return;
    }

    const row = data.row;
    latestResultEl.textContent =
      `운동: ${row.exercise_type ?? "-"} | ` +
      `이름: ${row.session_name} | ` +
      `횟수: ${row.total_count} | ` +
      `평균값: ${Number(row.avg_depth).toFixed(3)} | ` +
      `부족 횟수: ${row.depth_low_count} | ` +
      `자세 경고: ${row.torso_warning_count} | ` +
      `시간: ${Number(row.duration_sec).toFixed(1)}초 | ` +
      `저장시각: ${row.created_at}`;
  } catch (err) {
    console.error(err);
    latestResultEl.textContent = "기록 불러오기 실패";
  }
}
async function deleteSession(id) {
  const ok = window.confirm(`기록 #${id} 를 삭제할까?`);
  if (!ok) return;

  try {
    const res = await fetch(`${API_BASE}/api/sessions/${id}`, {
      method: "DELETE",
    });
    const data = await res.json();

    if (!data.ok) {
      throw new Error(data.message || "삭제 실패");
    }

    await loadLatestResult();
    await loadSessionHistory();
    saveStatusEl.textContent = `기록 ${id} 삭제 완료`;
  } catch (err) {
    console.error(err);
    saveStatusEl.textContent = "삭제 실패";
    logInfo("Delete error: " + err.message);
  }
}
async function loadSessionHistory() {
  try {
    const sort = getCurrentSortValue();
    const res = await fetch(`${API_BASE}/api/sessions?sort=${encodeURIComponent(sort)}`);
    const data = await res.json();

    if (!data.ok || !data.rows || data.rows.length === 0) {
      historyListEl.textContent = "(없음)";
      return;
    }

    historyListEl.innerHTML = data.rows
      .map((row) => {
        return `
          <div style="padding:10px 0; border-bottom:1px solid #f1f1f1; display:flex; justify-content:space-between; gap:12px; align-items:flex-start;">
            <div style="flex:1; min-width:0;">
              <div><b>${row.session_name}</b></div>
              <div style="margin-top:4px;">
                운동: ${row.exercise_type ?? "-"} |
                횟수: ${row.total_count} |
                평균값: ${Number(row.avg_depth).toFixed(3)} |
                부족: ${row.depth_low_count} |
                자세 경고: ${row.torso_warning_count} |
                시간: ${Number(row.duration_sec).toFixed(1)}초 |
                저장시각: ${row.created_at}
              </div>
            </div>
            <div>
              <button
                class="delete-session-btn"
                data-id="${row.id}"
                style="padding:6px 10px; border:1px solid #e57373; background:#fff5f5; color:#c62828; border-radius:8px; cursor:pointer;"
              >
                삭제
              </button>
            </div>
          </div>
        `;
      })
      .join("");

    const deleteButtons = historyListEl.querySelectorAll(".delete-session-btn");
    deleteButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = Number(btn.dataset.id);
        deleteSession(id);
      });
    });
  } catch (err) {
    console.error(err);
    historyListEl.textContent = "기록 목록 불러오기 실패";
  }
}

/** =========================
 * JSON Pose
 * ========================= */
async function loadPoseJson() {
  try {
    const poseJsonUrl = getCurrentPoseJsonUrl();
    const res = await fetch(poseJsonUrl);
    if (!res.ok) {
      throw new Error(`pose json load failed: ${res.status}`);
    }

    poseData = await res.json();
    poseFPS = poseData.fps || poseData.meta?.fps || 30;
    poseReady = Array.isArray(poseData.frames);

    setThreeStatus("Pose JSON loaded");
    logInfo(`${poseJsonUrl} loaded\nfps=${poseFPS}\nframes=${poseData.frames?.length ?? 0}`);
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
  if (!frame) return null;

  if (Array.isArray(frame.keypoints)) return frame.keypoints;
  if (Array.isArray(frame.raw_keypoints)) return frame.raw_keypoints;
  if (Array.isArray(frame.kpts)) return frame.kpts;
  if (Array.isArray(frame.pose)) return frame.pose;

  return null;
}

/** =========================
 * YOLO Pose
 * ========================= */
async function loadYoloModel() {
  try {
    setThreeStatus("YOLO 모델 로딩 중...");
    logInfo(
      "YOLO model loading...\n" +
        `ortDist=${ORT_DIST_BASE}\n` +
        `model=${YOLO_MODEL_URL}`
    );

    ort.env.wasm.wasmPaths = ORT_DIST_BASE;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;

    yoloSession = await ort.InferenceSession.create(YOLO_MODEL_URL, {
      executionProviders: ["wasm"],
    });

    yoloReady = true;

    setThreeStatus("YOLO 모델 로드 완료");
    logInfo(
      "YOLO model loaded\n" +
        `ortDist=${ORT_DIST_BASE}\n` +
        `inputs=${JSON.stringify(yoloSession.inputNames)}\n` +
        `outputs=${JSON.stringify(yoloSession.outputNames)}`
    );
  } catch (err) {
    console.error(err);
    yoloReady = false;
    setThreeStatus("YOLO 모델 로드 실패");
    logInfo(
      "YOLO load error\n" +
        `ortDist=${ORT_DIST_BASE}\n` +
        `message=${err.message}`
    );
  }
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
    meta: { srcW, srcH, scale, padX, padY, drawW, drawH },
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

  return {
    candidates: mappedCandidates,
    best: mappedCandidates[0],
  };
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

    const selected = selectBestCandidate(decoded.candidates, lastStablePoseKpts);
    if (!selected) return null;

    return selected;
  } finally {
    yoloBusy = false;
  }
}

/** =========================
 * 2D Drawing
 * ========================= */
function clearOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
}
function drawThresholdByIndex() {
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

  lastStablePoseKpts = null;
  lastStablePoseTime = 0;
  lostPoseFrames = 0;
}

/** =========================
 * 스쿼트/푸쉬업 분석
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
    metric: depth,
    postureWarnMetric: torsoLean,
    formMetric: 1.0,
  };
}
function computePushupMetrics(k) {
  if (!k || k.length < 17) return null;

  const ls = k[5];
  const rs = k[6];
  const le = k[7];
  const re = k[8];
  const lw = k[9];
  const rw = k[10];
  const lh = k[11];
  const rh = k[12];
  const la = k[15];
  const ra = k[16];

  const leftArmOk = kpOk(ls) && kpOk(le) && kpOk(lw);
  const rightArmOk = kpOk(rs) && kpOk(re) && kpOk(rw);
  if (!leftArmOk && !rightArmOk) return null;

  let sh, el, wr;
  if (leftArmOk && rightArmOk) {
    const leftScore = (ls[2] ?? 1) + (le[2] ?? 1) + (lw[2] ?? 1);
    const rightScore = (rs[2] ?? 1) + (re[2] ?? 1) + (rw[2] ?? 1);
    if (leftScore >= rightScore) {
      sh = ls; el = le; wr = lw;
    } else {
      sh = rs; el = re; wr = rw;
    }
  } else if (leftArmOk) {
    sh = ls; el = le; wr = lw;
  } else {
    sh = rs; el = re; wr = rw;
  }

  const upperArmLen = dist2(sh, el);
  const foreArmLen = dist2(el, wr);
  const armLen = Math.max(1e-6, upperArmLen + foreArmLen);

  const elbowAngle = angleDeg2(sh, el, wr);
  const elbowBend = clamp01((175 - elbowAngle) / 95);

  let shoulderDrop = 0;
  if (kpOk(lh) && kpOk(rh)) {
    const hipCenter = [(lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5];
    const shoulderCenter = kpOk(ls) && kpOk(rs)
      ? [(ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5]
      : [sh[0], sh[1]];

    shoulderDrop = clamp01(
      (shoulderCenter[1] - hipCenter[1] + armLen * 0.15) /
        Math.max(1e-6, armLen * 0.9)
    );
  }

  const metric = clamp01(elbowBend * 0.7 + shoulderDrop * 0.3);

  let bodyLineWarn = 0;
  if (kpOk(ls) && kpOk(rs) && kpOk(lh) && kpOk(rh) && kpOk(la) && kpOk(ra)) {
    const shoulderCenter = [(ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5];
    const hipCenter = [(lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5];
    const ankleCenter = [(la[0] + ra[0]) * 0.5, (la[1] + ra[1]) * 0.5];

    const lineDx = ankleCenter[0] - shoulderCenter[0];
    const lineDy = ankleCenter[1] - shoulderCenter[1];
    const lineLen = Math.max(1e-6, Math.hypot(lineDx, lineDy));

    const proj =
      ((hipCenter[0] - shoulderCenter[0]) * lineDx +
        (hipCenter[1] - shoulderCenter[1]) * lineDy) /
      (lineLen * lineLen);

    const projX = shoulderCenter[0] + proj * lineDx;
    const projY = shoulderCenter[1] + proj * lineDy;

    const perp = Math.hypot(hipCenter[0] - projX, hipCenter[1] - projY);
    bodyLineWarn = perp / lineLen;
  }

  return {
    metric,
    postureWarnMetric: bodyLineWarn,
    formMetric: elbowAngle,
  };
}
function evaluateSpeed(durationSec) {
  if (durationSec < 0.8) return "너무 빠름";
  if (durationSec < 2.0) return "적정 속도";
  return "너무 느림";
}
function updateRepCounterAndFeedback(kpts) {
  const m = exerciseType === "pushup" ? computePushupMetrics(kpts) : computeSquatMetrics(kpts);
  if (!m) {
    lastFeedback = "관절 인식 부족";
    return;
  }

  lastMotionMetric = m.metric;
  lastPostureMetric = m.postureWarnMetric;
  lastFormMetric = m.formMetric;

  analyzedFrameCount += 1;
  metricAccum += m.metric;

  const postureWarnNow =
    exerciseType === "pushup"
      ? m.postureWarnMetric > BODYLINE_WARN_THRESHOLD_PUSHUP
      : m.postureWarnMetric > TORSO_WARN_THRESHOLD_SQUAT;

  if (isResting) {
    lastFeedback = `휴식 중 / ${restRemaining}초 남음`;
    updateResultCards();
    return;
  }

  const downEnter = exerciseType === "pushup" ? 0.60 : 0.58;
  const upExit = exerciseType === "pushup" ? 0.20 : 0.22;

  if (motionState === "UP" && m.metric >= downEnter) {
    motionState = "DOWN";
    repWarnFlag = false;
    repMaxMetric = m.metric;
    repStartTime = getNowSec();
  }

  if (motionState === "DOWN") {
    repMaxMetric = Math.max(repMaxMetric, m.metric);
    if (postureWarnNow) repWarnFlag = true;
  }

  const lowThreshold =
    exerciseType === "pushup" ? DEPTH_LOW_THRESHOLD_PUSHUP : DEPTH_LOW_THRESHOLD_SQUAT;

  const currentLowDisplay =
    motionState === "DOWN" ? repMaxMetric < lowThreshold : false;

  if (motionState === "DOWN" && m.metric <= upExit) {
    motionState = "UP";
    repCount += 1;

    if (workoutMode === "normal") {
      currentSetCount += 1;
    }

    if (repMaxMetric < lowThreshold) {
      depthLowCount += 1;
    }
    if (repWarnFlag) {
      torsoWarningCount += 1;
    }

    lastRepDuration = Math.max(0, getNowSec() - repStartTime);
    speedFeedback = evaluateSpeed(lastRepDuration);

    if (workoutMode === "normal" && currentSetCount >= setRepTarget) {
      finishCurrentSet();
    }

    repWarnFlag = false;
    repMaxMetric = 0;
  }

  const feedbacks = [];
  if (isResting) {
    feedbacks.push(`휴식 중 ${restRemaining}초`);
  } else if (motionState === "DOWN") {
    feedbacks.push(currentLowDisplay ? "가동범위 부족" : "가동범위 양호");
    feedbacks.push(
      postureWarnNow
        ? (exerciseType === "pushup" ? "몸 일직선 무너짐" : "상체 숙임 큼")
        : (exerciseType === "pushup" ? "몸 일직선 양호" : "상체 각도 양호")
    );
  } else {
    feedbacks.push("준비 자세");
    feedbacks.push(
      postureWarnNow
        ? (exerciseType === "pushup" ? "몸통 정렬 주의" : "상체 약간 숙임")
        : (exerciseType === "pushup" ? "몸통 정렬 양호" : "상체 각도 양호")
    );
  }

  if (speedFeedback && speedFeedback !== "대기") {
    feedbacks.push(speedFeedback);
  }

  lastFeedback = feedbacks.join(" / ");
  updateResultCards();
}

/** =========================
 * Video / Webcam Input
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
async function startJsonVideoInput() {
  const currentVideoUrl = getCurrentVideoUrl();

  video.srcObject = null;
  video.src = currentVideoUrl;
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
  webcamReady = false;

  setThreeStatus("입력 영상 시작됨");
  logInfo(`입력 영상 시작 완료\n${currentVideoUrl}`);
}
async function startWebcamInput() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("브라우저가 getUserMedia를 지원하지 않음");
  }

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

  resizeVideoStage(video.videoWidth || 640, video.videoHeight || 480);
  await video.play();

  videoReady = true;
  webcamReady = true;

  setThreeStatus("웹캠 시작됨");
  logInfo("웹캠 시작 완료");
}
async function startInput() {
  try {
    stopInput(false);
    readExerciseType();
    readInputMode();
    readWorkoutInputs();

    if (inputMode === "json") {
      await startJsonVideoInput();
    } else {
      if (exerciseType !== "squat") {
        throw new Error("Push-up은 Webcam+YOLO 미지원. JSON 모드를 사용해.");
      }
      if (!yoloReady) {
        await loadYoloModel();
      }
      await startWebcamInput();
    }

    resetSessionStats();

    if (videoTimer) {
      clearInterval(videoTimer);
      videoTimer = null;
    }

    drawVideoFrameToCanvas();
    videoTimer = setInterval(drawVideoFrameToCanvas, VIDEO_DRAW_INTERVAL);
  } catch (err) {
    console.error(err);
    videoReady = false;
    webcamReady = false;
    setThreeStatus("Input error");
    logInfo("Input error: " + err.message);
  }
}
function stopWebcamStream() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
}
function stopInput(resetTime = true) {
  if (videoTimer) {
    clearInterval(videoTimer);
    videoTimer = null;
  }

  const shouldSave = !sessionSaved && (repCount > 0 || analyzedFrameCount > 0);

  setLive(false);

  if (!video.paused) video.pause();

  if (shouldSave) {
    saveSessionResult();
  }

  if (inputMode === "webcam") {
    stopWebcamStream();
  }

  if (resetTime) {
    try {
      video.currentTime = 0;
    } catch {}
  }

  video.srcObject = null;
  videoReady = false;
  webcamReady = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  resetOverlayState();
  resetAvatarPoseToStart();
  stopRestTimer();

  setThreeStatus("입력 정지");
  logInfo("입력 정지");
}

/** =========================
 * JSON Sync / Webcam YOLO Sync
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

  if (exerciseType === "pushup" && frame) {
    analyzedFrameCount += 1;

    const rawDepth = Number(frame.depth ?? 0);
    const normDepth = clamp01(rawDepth / 500);
    lastMotionMetric = normDepth;
    motionMetricRaw = lastMotionMetric;
    animProgressRaw = mapMotionToProgress(motionMetricRaw);

    lastFeedback = String(frame.feedback ?? "push-up json");
    repCount = Number(frame.count ?? repCount);

    metricAccum += lastMotionMetric;

    lastPostureMetric = Number(frame.bodyline_err ?? 0);
    lastFormMetric = Number(frame.elbow_angle ?? 0);

    updateResultCards();

    logInfoThrottled(
      `exercise=${exerciseType}\n` +
        `inputMode=${inputMode}\n` +
        `mode=${workoutMode}\n` +
        `videoTime=${video.currentTime.toFixed(2)}s\n` +
        `poseFPS=${poseFPS}\n\n` +
        `repCount=${repCount}\n` +
        `depthRaw=${rawDepth.toFixed(3)}\n` +
        `metric=${lastMotionMetric.toFixed(3)}\n` +
        `progressRaw=${animProgressRaw.toFixed(3)}\n` +
        `progressSmooth=${animProgressSmooth.toFixed(3)}\n` +
        `bodylineErr=${lastPostureMetric.toFixed(3)}\n` +
        `elbowAngle=${lastFormMetric.toFixed(3)}\n` +
        `feedback=${lastFeedback}\n\n` +
        `clip=${exerciseClip ? exerciseClip.name || "(no name)" : "none"}\n` +
        `clipDuration=${exerciseClip ? exerciseClip.duration.toFixed(3) : "0"}\n` +
        `range=${CLIP_START_NORM} ~ ${CLIP_END_NORM}\n` +
        `actionTime=${exerciseAction ? exerciseAction.time.toFixed(3) : "0"}`,
      2
    );

    setThreeStatus(`${getExerciseLabel()} JSON 분석 중 / count ${repCount} / ${lastFeedback}`);
    return;
  }

  updateRepCounterAndFeedback(lastPoseKpts);

  motionMetricRaw = lastMotionMetric;
  animProgressRaw = mapMotionToProgress(motionMetricRaw);

  logInfoThrottled(
    `exercise=${exerciseType}\n` +
      `inputMode=${inputMode}\n` +
      `mode=${workoutMode}\n` +
      `videoTime=${video.currentTime.toFixed(2)}s\n` +
      `poseFPS=${poseFPS}\n\n` +
      `repCount=${repCount}\n` +
      `currentSet=${currentSet}\n` +
      `currentSetCount=${currentSetCount}\n` +
      `isResting=${isResting}\n` +
      `restRemaining=${restRemaining}\n\n` +
      `metric=${lastMotionMetric.toFixed(3)}\n` +
      `repMaxMetric=${repMaxMetric.toFixed(3)}\n` +
      `progressRaw=${animProgressRaw.toFixed(3)}\n` +
      `progressSmooth=${animProgressSmooth.toFixed(3)}\n` +
      `postureMetric=${lastPostureMetric.toFixed(3)}\n` +
      `speed=${speedFeedback}\n` +
      `repTime=${lastRepDuration.toFixed(2)}s\n` +
      `feedback=${lastFeedback}\n\n` +
      `clip=${exerciseClip ? exerciseClip.name || "(no name)" : "none"}\n` +
      `clipDuration=${exerciseClip ? exerciseClip.duration.toFixed(3) : "0"}\n` +
      `range=${CLIP_START_NORM} ~ ${CLIP_END_NORM}\n` +
      `actionTime=${exerciseAction ? exerciseAction.time.toFixed(3) : "0"}`,
    2
  );

  setThreeStatus(`${getExerciseLabel()} JSON 분석 중 / count ${repCount} / ${lastFeedback}`);
}
async function inferPoseFromWebcam() {
  if (!videoReady || !yoloReady) return;
  if (exerciseType !== "squat") {
    lastFeedback = "Push-up webcam 미지원";
    return;
  }

  const result = await inferYoloFromCurrentFrame();
  const nowSec = getNowSec();

  if (!result?.keypoints) {
    lostPoseFrames += 1;

    const withinHoldTime = (nowSec - lastStablePoseTime) * 1000 <= POSE_HOLD_MS;
    const withinHoldFrames = lostPoseFrames <= POSE_MAX_LOST_FRAMES;

    if (lastStablePoseKpts && withinHoldTime && withinHoldFrames) {
      lastPoseKpts = cloneKpts(lastStablePoseKpts);
      updateRepCounterAndFeedback(lastPoseKpts);
      lastFeedback = "이전 포즈 유지 중";
    } else {
      lastFeedback = "YOLO 인식 없음";
    }

    logInfoThrottled(
      `exercise=${exerciseType}\n` +
        `inputMode=${inputMode}\n` +
        `mode=${workoutMode}\n` +
        `yolo=none\n` +
        `lostPoseFrames=${lostPoseFrames}\n` +
        `feedback=${lastFeedback}`,
      3
    );
    return;
  }

  const stabilizedKpts = stabilizeWebcamPose(result.keypoints, nowSec);

  if (!stabilizedKpts || countVisibleJoints(stabilizedKpts, 0.05) < 6) {
    lastFeedback = "YOLO 관절 부족";
    logInfoThrottled(
      `exercise=${exerciseType}\n` +
        `inputMode=${inputMode}\n` +
        `mode=${workoutMode}\n` +
        `yoloScore=${result.score.toFixed(3)}\n` +
        `visible=${countVisibleJoints(stabilizedKpts, 0.05)}\n` +
        `feedback=${lastFeedback}`,
      2
    );
    return;
  }

  lastPoseKpts = fixTorsoCross(stabilizedKpts, lastStablePoseKpts);
  updateRepCounterAndFeedback(lastPoseKpts);

  motionMetricRaw = lastMotionMetric;
  animProgressRaw = mapMotionToProgress(motionMetricRaw);

  logInfoThrottled(
    `exercise=${exerciseType}\n` +
      `inputMode=${inputMode}\n` +
      `mode=${workoutMode}\n` +
      `yoloScore=${result.score.toFixed(3)}\n` +
      `visible=${countVisibleJoints(stabilizedKpts, 0.05)}\n` +
      `lostPoseFrames=${lostPoseFrames}\n\n` +
      `repCount=${repCount}\n` +
      `currentSet=${currentSet}\n` +
      `currentSetCount=${currentSetCount}\n` +
      `isResting=${isResting}\n` +
      `restRemaining=${restRemaining}\n\n` +
      `metric=${lastMotionMetric.toFixed(3)}\n` +
      `repMaxMetric=${repMaxMetric.toFixed(3)}\n` +
      `progressRaw=${animProgressRaw.toFixed(3)}\n` +
      `progressSmooth=${animProgressSmooth.toFixed(3)}\n` +
      `postureMetric=${lastPostureMetric.toFixed(3)}\n` +
      `speed=${speedFeedback}\n` +
      `repTime=${lastRepDuration.toFixed(2)}s\n` +
      `feedback=${lastFeedback}\n\n` +
      `clip=${exerciseClip ? exerciseClip.name || "(no name)" : "none"}\n` +
      `clipDuration=${exerciseClip ? exerciseClip.duration.toFixed(3) : "0"}\n` +
      `range=${CLIP_START_NORM} ~ ${CLIP_END_NORM}\n` +
      `actionTime=${exerciseAction ? exerciseAction.time.toFixed(3) : "0"}`,
    2
  );

  setThreeStatus(`YOLO 분석 중 / count ${repCount} / ${lastFeedback}`);
}
async function syncOrInferOneFrame() {
  if (inputMode === "json") {
    syncPoseFromVideoTime();
  } else {
    await inferPoseFromWebcam();
  }
}
async function liveLoop() {
  if (!liveOn) return;

  await syncOrInferOneFrame();

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
  return getAvatarBasePosition().clone().add(getLookTargetOffset());
}
function updateProgressControlledAnimation() {
  if (!mixer || !exerciseClip || !exerciseAction || animationPaused) return;

  const prev = animProgressSmooth;
  const next = lerp(prev, animProgressRaw, PROGRESS_EMA);

  animProgressSmooth = postProcessProgress(next);
  animProgressRender = animProgressSmooth;

  const { clipStart, usableDuration } = getClipRange();
  const targetTime = clipStart + usableDuration * animProgressRender;

  if (Math.abs(targetTime - lastMixerTime) > 0.001) {
    exerciseAction.time = targetTime;
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

    const radius = exerciseType === "pushup" ? 2.0 : 2.5;
    const camY = exerciseType === "pushup" ? target.y + 0.95 : target.y + 0.2;

    camera.position.x = target.x + Math.sin(t) * radius;
    camera.position.z = target.z + Math.cos(t) * radius;
    camera.position.y = camY;
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
  const avatarUrl = getCurrentAvatarGlbUrl();

  return new Promise((resolve, reject) => {
    loader.load(
      avatarUrl,
      (gltf) => {
        if (avatarScene) scene.remove(avatarScene);

        avatarScene = gltf.scene;
        avatarScene.position.copy(getAvatarBasePosition());
        scene.add(avatarScene);

        mixer = null;
        exerciseClip = null;
        exerciseAction = null;
        lastMixerTime = -999;

        resetSessionStats();

        const clips = gltf.animations || [];
        const clipLines = clips.map(
          (a, i) => `[${i}] ${a.name || "(no name)"} / duration=${a.duration.toFixed(3)} / tracks=${a.tracks.length}`
        );

        if (clips.length > 0) {
          mixer = new THREE.AnimationMixer(avatarScene);

          const lowerType = exerciseType.toLowerCase();
          const byName = clips.find((a) =>
            (a.name || "").toLowerCase().includes(lowerType)
          );
          const byDuration = [...clips].sort((a, b) => b.duration - a.duration)[0];

          exerciseClip = byName || byDuration;
          exerciseAction = mixer.clipAction(exerciseClip);

          exerciseAction.enabled = true;
          exerciseAction.setLoop(THREE.LoopOnce, 1);
          exerciseAction.clampWhenFinished = true;
          exerciseAction.play();
          exerciseAction.paused = true;

          const { clipStart } = getClipRange();
          exerciseAction.time = clipStart;

          mixer.update(0);
          avatarScene.updateMatrixWorld(true);

          lastMixerTime = clipStart;
        }

        const target = getAvatarLookTarget();
        const camOffset = getFrontCameraOffset();

        camera.position.set(
          target.x + camOffset.x,
          target.y + camOffset.y,
          target.z + camOffset.z
        );
        camera.lookAt(target);

        animationPaused = false;
        setThreeStatus("장면 로드 완료 - 진행도 제어 준비 완료");

        logInfo(
          "장면 로드 완료\n" +
            `exercise=${exerciseType}\n` +
            `avatar=${avatarUrl}\n` +
            `avatarPos=${JSON.stringify(getAvatarBasePosition().toArray())}\n` +
            `camOffset=${JSON.stringify(camOffset.toArray())}\n` +
            `animation clips: ${clips.length}\n` +
            (clipLines.length ? clipLines.join("\n") + "\n\n" : "") +
            `selected clip: ${exerciseClip ? exerciseClip.name || "(no name)" : "none"}\n` +
            `duration=${exerciseClip ? exerciseClip.duration.toFixed(3) : "0"}\n` +
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

    setThreeStatus("장면 로드 완료");
  } catch (e) {
    console.error(e);
    setThreeStatus("Load Scene error");
    logInfo("Scene load error: " + e.message);
  }
}

/** =========================
 * Buttons / Events
 * ========================= */
document.getElementById("start").addEventListener("click", startInput);
document.getElementById("stop").addEventListener("click", () => stopInput(true));

document.getElementById("loadPose").addEventListener("click", loadPoseJson);
document.getElementById("loadYolo").addEventListener("click", async () => {
  if (exerciseType !== "squat") {
    logInfo("Push-up은 YOLO 사용 안 함. JSON 모드만 지원.");
    return;
  }
  await loadYoloModel();
});

document.getElementById("sync").addEventListener("click", async () => {
  if (inputMode === "json") {
    if (!poseReady) {
      logInfo("Load Pose JSON 먼저.");
      return;
    }
    if (!videoReady) {
      logInfo("Start Input 먼저.");
      return;
    }
  } else {
    if (!yoloReady) {
      logInfo("Load YOLO 먼저.");
      return;
    }
    if (!videoReady) {
      logInfo("Start Input 먼저.");
      return;
    }
  }

  await syncOrInferOneFrame();
});

document.getElementById("clear").addEventListener("click", () => {
  resetOverlayState();

  if (videoReady) {
    drawVideoFrameToCanvas();
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  resetSessionStats();
  resetAvatarPoseToStart();

  logInfo("(cleared)");
  setThreeStatus("초기화됨");
});

document.getElementById("live").addEventListener("click", () => {
  if (inputMode === "json") {
    if (!poseReady) {
      logInfo("Load Pose JSON 먼저.");
      return;
    }
    if (!videoReady) {
      logInfo("Start Input 먼저.");
      return;
    }
  } else {
    if (!yoloReady) {
      logInfo("Load YOLO 먼저.");
      return;
    }
    if (!videoReady) {
      logInfo("Start Input 먼저.");
      return;
    }
  }

  setLive(!liveOn);
  setThreeStatus(liveOn ? "분석 LIVE ON" : "분석 LIVE OFF");
});

document.getElementById("loadScene").addEventListener("click", loadScene);

document.getElementById("toggleAnim").addEventListener("click", () => {
  if (!mixer || !exerciseClip) {
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

  resetSessionStats();
  resetAvatarPoseToStart();
  setLive(false);

  setThreeStatus("애니메이션 진행도 리셋 완료");
  logInfo("애니메이션 진행도 리셋됨");
});

document.getElementById("testPose").addEventListener("click", () => {
  if (!exerciseAction || !mixer || !exerciseClip) {
    logInfo("먼저 Scene을 로드하세요");
    return;
  }

  const { clipStart, usableDuration } = getClipRange();
  const t = clipStart + usableDuration * 0.5;

  exerciseAction.time = t;
  mixer.update(0);
  avatarScene?.updateMatrixWorld(true);
  lastMixerTime = t;

  setThreeStatus(`Test pose applied / t=${t.toFixed(2)}`);
  logInfo(
    `Test pose applied\nclip=${exerciseClip.name || "(no name)"}\nduration=${exerciseClip.duration.toFixed(3)}\nrange=${CLIP_START_NORM} ~ ${CLIP_END_NORM}\ntime=${t.toFixed(3)}`
  );
});

document.getElementById("saveResult").addEventListener("click", async () => {
  await saveSessionResult();
});

historySortEl.addEventListener("change", async () => {
  await loadSessionHistory();
});

document.querySelectorAll('input[name="workoutMode"]').forEach((el) => {
  el.addEventListener("change", () => {
    readWorkoutInputs();
  });
});

document.querySelectorAll('input[name="inputMode"]').forEach((el) => {
  el.addEventListener("change", async () => {
    const prevMode = inputMode;
    readInputMode();

    if (liveOn) setLive(false);
    if (videoReady) stopInput(false);

    logInfo(`입력 모드 변경: ${prevMode} -> ${inputMode}`);
    setThreeStatus(`입력 모드 변경: ${getInputModeLabel()}`);
  });
});

document.querySelectorAll('input[name="exerciseType"]').forEach((el) => {
  el.addEventListener("change", async () => {
    const prevExercise = exerciseType;
    readExerciseType();
    readInputMode();

    if (exerciseType === "pushup") {
      autoRotateEl.checked = false;
    }

    if (liveOn) setLive(false);
    if (videoReady) stopInput(false);

    poseReady = false;
    poseData = null;

    resetSessionStats();
    resetOverlayState();
    resetAvatarPoseToStart();

    logInfo(`운동 종류 변경: ${prevExercise} -> ${exerciseType}`);
    setThreeStatus(`운동 종류 변경: ${getExerciseLabel()}`);
  });
});

[targetSetsInput, setRepTargetInput, restSecondsInput].forEach((el) => {
  el.addEventListener("input", () => {
    readWorkoutInputs();
  });
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
readExerciseType();
readInputMode();
readWorkoutInputs();
refreshExercisePathTexts();
setThreeStatus("Ready (운동 선택 → 입력 모드 선택 → Start Input → Load Pose/YOLO → Load Scene → Live ON)");
animateOverlay();
loadLatestResult();
loadSessionHistory();
updateResultCards();