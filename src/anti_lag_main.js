/* eslint-disable no-console */
import * as THREE from "https://unpkg.com/three@0.160.1/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.1/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://unpkg.com/three@0.160.1/examples/jsm/loaders/GLTFLoader.js";
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const GUIDE_VIDEO_URL = "/squat.mp4";
const HAND_MODEL_URL = "/models/hand_landmarker.task";
const GYM_URL = "/models/Untitled_gym.glb";
const AVATAR_URL = "/models/Untitled_squat.glb";

const state = {
  currentMode: "standard", // standard | feedback | avatar
  avatarView: "avatar", // webcam | avatar
  themeLight: false,
  webcamStream: null,
  handLandmarker: null,
  handBusy: false,
  running: false,
  renderLoopId: 0,
  fistFrames: 0,
  fistCooldownUntil: 0,
  lastHandLabel: "READY",
};

const dom = {
  modeScreen: document.getElementById("mode-screen"),
  appScreen: document.getElementById("app-screen"),
  themeToggle: document.getElementById("themeToggle"),
  modeCards: [...document.querySelectorAll(".mode-card")],
  backBtn: document.getElementById("backBtn"),
  modeSelect: document.getElementById("modeSelect"),
  manualToggleBtn: document.getElementById("manualToggleBtn"),
  guideVideoMain: document.getElementById("guideVideoMain"),
  webcamVideoMain: document.getElementById("webcamVideoMain"),
  avatarMainWrap: document.getElementById("avatarMainWrap"),
  guideVideoSub: document.getElementById("guideVideoSub"),
  webcamVideoSub: document.getElementById("webcamVideoSub"),
  avatarSubWrap: document.getElementById("avatarSubWrap"),
  subBox: document.getElementById("subBox"),
  modeBadge: document.getElementById("modeBadge"),
  viewBadge: document.getElementById("viewBadge"),
  guideBadge: document.getElementById("guideBadge"),
  handStatus: document.getElementById("handStatus"),
  cameraStatus: document.getElementById("cameraStatus"),
  avatarStatus: document.getElementById("avatarStatus"),
  modeInfo: document.getElementById("modeInfo"),
};

const threeCtx = {
  main: null,
  sub: null,
};

function setText(el, text) {
  if (el) el.textContent = text;
}

function show(el) {
  el?.classList.remove("hidden");
}
function hide(el) {
  el?.classList.add("hidden");
}

function setTheme() {
  document.body.classList.toggle("light-mode", state.themeLight);
  dom.themeToggle.textContent = state.themeLight ? "☀️ THEME" : "🌙 THEME";
}

function createThreeRenderer(targetEl) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0c);

  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
  camera.position.set(0, 1.5, 3.6);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(targetEl.clientWidth || 300, targetEl.clientHeight || 300);
  targetEl.innerHTML = "";
  targetEl.appendChild(renderer.domElement);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 1.4);
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(2, 3, 2);
  scene.add(hemi, dir);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 1.0, 0);
  controls.enablePan = false;
  controls.minDistance = 2.2;
  controls.maxDistance = 5.2;

  return { scene, camera, renderer, controls, avatar: null, gym: null, clock: new THREE.Clock() };
}

async function loadSceneAssets(ctx) {
  const loader = new GLTFLoader();
  const load = (url) => new Promise((resolve, reject) => {
    loader.load(url, resolve, undefined, reject);
  });

  try {
    const [gymGltf, avatarGltf] = await Promise.all([
      load(GYM_URL).catch(() => null),
      load(AVATAR_URL),
    ]);

    if (gymGltf?.scene) {
      ctx.gym = gymGltf.scene;
      ctx.gym.position.set(0, 0, 0);
      ctx.gym.scale.set(1, 1, 1);
      ctx.scene.add(ctx.gym);
    }

    ctx.avatar = avatarGltf.scene;
    ctx.avatar.position.set(0, 0, 0.8);
    ctx.avatar.scale.set(1, 1, 1);
    ctx.scene.add(ctx.avatar);
    setText(dom.avatarStatus, "READY");
  } catch (err) {
    console.error(err);
    setText(dom.avatarStatus, "LOAD FAIL");
  }
}

async function initThree() {
  if (!threeCtx.main) {
    threeCtx.main = createThreeRenderer(dom.avatarMainWrap);
    await loadSceneAssets(threeCtx.main);
  }
  if (!threeCtx.sub) {
    threeCtx.sub = createThreeRenderer(dom.avatarSubWrap);
    await loadSceneAssets(threeCtx.sub);
  }
  resizeThree();
}

function resizeThree() {
  [threeCtx.main, threeCtx.sub].forEach((ctx) => {
    if (!ctx) return;
    const parent = ctx.renderer.domElement.parentElement;
    const w = Math.max(1, parent.clientWidth || 300);
    const h = Math.max(1, parent.clientHeight || 300);
    ctx.camera.aspect = w / h;
    ctx.camera.updateProjectionMatrix();
    ctx.renderer.setSize(w, h, false);
  });
}

async function ensureGuideVideo(videoEl) {
  if (!videoEl.dataset.ready) {
    videoEl.src = GUIDE_VIDEO_URL;
    videoEl.muted = true;
    videoEl.loop = true;
    videoEl.playsInline = true;
    await new Promise((resolve, reject) => {
      const ok = () => {
        cleanup();
        videoEl.dataset.ready = "1";
        resolve();
      };
      const fail = () => {
        cleanup();
        reject(new Error("guide video load failed"));
      };
      const cleanup = () => {
        videoEl.removeEventListener("loadedmetadata", ok);
        videoEl.removeEventListener("error", fail);
      };
      videoEl.addEventListener("loadedmetadata", ok);
      videoEl.addEventListener("error", fail);
      videoEl.load();
    });
  }
  try { await videoEl.play(); } catch {}
}

async function initGuideVideos() {
  await Promise.all([
    ensureGuideVideo(dom.guideVideoMain),
    ensureGuideVideo(dom.guideVideoSub),
  ]);
}

async function startWebcam() {
  if (state.webcamStream) return;
  state.webcamStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  dom.webcamVideoMain.srcObject = state.webcamStream;
  dom.webcamVideoSub.srcObject = state.webcamStream;
  await Promise.allSettled([dom.webcamVideoMain.play(), dom.webcamVideoSub.play()]);
  setText(dom.cameraStatus, "READY");
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach((t) => t.stop());
    state.webcamStream = null;
  }
  dom.webcamVideoMain.srcObject = null;
  dom.webcamVideoSub.srcObject = null;
  setText(dom.cameraStatus, "STOP");
}

async function initHands() {
  if (state.handLandmarker) return;
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL_URL },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });
  setText(dom.handStatus, "READY");
}

function isFingerFolded(lm, tip, pip) {
  const a = lm[tip];
  const b = lm[pip];
  if (!a || !b) return false;
  return a.y > b.y;
}

function detectFist(result) {
  const lm = result?.landmarks?.[0];
  if (!lm) return false;

  const indexFold = isFingerFolded(lm, 8, 6);
  const middleFold = isFingerFolded(lm, 12, 10);
  const ringFold = isFingerFolded(lm, 16, 14);
  const pinkyFold = isFingerFolded(lm, 20, 18);

  return indexFold && middleFold && ringFold && pinkyFold;
}

function handleGestureFromWebcam() {
  if (state.currentMode !== "avatar") return;
  if (!state.handLandmarker || state.handBusy) return;
  const video = dom.webcamVideoMain;
  if (!video.videoWidth || !video.videoHeight) return;

  state.handBusy = true;
  try {
    const res = state.handLandmarker.detectForVideo(video, performance.now());
    const fist = detectFist(res);
    setText(dom.handStatus, fist ? "FIST" : (res?.landmarks?.length ? "HAND" : "NONE"));

    if (fist) state.fistFrames += 1;
    else state.fistFrames = 0;

    if (
      fist &&
      state.fistFrames >= 5 &&
      performance.now() > state.fistCooldownUntil
    ) {
      state.avatarView = state.avatarView === "avatar" ? "webcam" : "avatar";
      state.fistFrames = 0;
      state.fistCooldownUntil = performance.now() + 1200;
      updateLayout();
    }
  } catch (err) {
    console.error(err);
  } finally {
    state.handBusy = false;
  }
}

function hideAllVisuals() {
  [
    dom.guideVideoMain,
    dom.webcamVideoMain,
    dom.avatarMainWrap,
    dom.guideVideoSub,
    dom.webcamVideoSub,
    dom.avatarSubWrap,
  ].forEach(hide);
  hide(dom.subBox);
}

function updateLayout() {
  hideAllVisuals();

  dom.modeSelect.value = state.currentMode;
  setText(dom.modeBadge, `MODE : ${state.currentMode.toUpperCase()}`);
  setText(dom.modeInfo, state.currentMode.toUpperCase());

  if (state.currentMode === "standard") {
    show(dom.avatarMainWrap);
    show(dom.subBox);
    show(dom.guideVideoSub);
    setText(dom.viewBadge, "VIEW : AVATAR + VIDEO");
    setText(dom.guideBadge, "정석 모드: 메인 아바타 / 서브 기준 비디오");
    hide(dom.manualToggleBtn);
    return;
  }

  if (state.currentMode === "feedback") {
    show(dom.webcamVideoMain);
    show(dom.subBox);
    show(dom.guideVideoSub);
    setText(dom.viewBadge, "VIEW : WEBCAM + VIDEO");
    setText(dom.guideBadge, "피드백 모드: 메인 웹캠 / 서브 기준 비디오");
    hide(dom.manualToggleBtn);
    return;
  }

  if (state.currentMode === "avatar") {
    if (state.avatarView === "avatar") show(dom.avatarMainWrap);
    else show(dom.webcamVideoMain);

    setText(dom.viewBadge, `VIEW : ${state.avatarView.toUpperCase()}`);
    setText(dom.guideBadge, "아바타 모드: 주먹 5프레임 유지 → Webcam / Avatar 전환");
    show(dom.manualToggleBtn);
  }
}

function renderAvatarMotion(ctx, t) {
  if (!ctx?.avatar) return;
  ctx.avatar.position.y = -Math.abs(Math.sin(t * 1.7)) * 0.35;
  ctx.avatar.rotation.y = Math.sin(t * 0.8) * 0.18;
}

function renderLoop() {
  state.renderLoopId = requestAnimationFrame(renderLoop);
  const t = performance.now() * 0.001;

  handleGestureFromWebcam();

  [threeCtx.main, threeCtx.sub].forEach((ctx) => {
    if (!ctx) return;
    const isMainVisible = !dom.avatarMainWrap.classList.contains("hidden") && ctx === threeCtx.main;
    const isSubVisible = !dom.avatarSubWrap.classList.contains("hidden") && ctx === threeCtx.sub;
    if (!isMainVisible && !isSubVisible) return;
    renderAvatarMotion(ctx, t);
    ctx.controls.update();
    ctx.renderer.render(ctx.scene, ctx.camera);
  });
}

async function startApp(mode) {
  state.currentMode = mode;
  state.avatarView = "avatar";
  dom.modeScreen.classList.add("hidden");
  dom.appScreen.classList.remove("hidden");

  await initThree();
  await initGuideVideos();
  await startWebcam();
  await initHands();

  state.running = true;
  updateLayout();
  if (!state.renderLoopId) renderLoop();
}

function stopApp() {
  state.running = false;
  cancelAnimationFrame(state.renderLoopId);
  state.renderLoopId = 0;
  stopWebcam();
  dom.appScreen.classList.add("hidden");
  dom.modeScreen.classList.remove("hidden");
  setText(dom.handStatus, "READY");
}

function bindEvents() {
  dom.themeToggle.addEventListener("click", () => {
    state.themeLight = !state.themeLight;
    setTheme();
  });

  dom.modeCards.forEach((card) => {
    card.addEventListener("click", () => {
      startApp(card.dataset.mode).catch((err) => {
        console.error(err);
        alert("초기화 실패: " + err.message);
      });
    });
  });

  dom.backBtn.addEventListener("click", stopApp);
  dom.modeSelect.addEventListener("change", () => {
    state.currentMode = dom.modeSelect.value;
    if (state.currentMode !== "avatar") state.avatarView = "avatar";
    updateLayout();
  });
  dom.manualToggleBtn.addEventListener("click", () => {
    state.avatarView = state.avatarView === "avatar" ? "webcam" : "avatar";
    updateLayout();
  });
  window.addEventListener("resize", resizeThree);
}

setTheme();
bindEvents();
