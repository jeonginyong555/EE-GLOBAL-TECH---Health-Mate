/* eslint-disable no-console */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/** =========================
 * UI - 아바타 창만
 * ========================= */
const app = document.querySelector("#app");

app.innerHTML = `
  <div style="font-family:sans-serif; padding:12px;">
    <h2 style="margin:0 0 8px;">Health-Mate Avatar Preview</h2>

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
    <pre id="info" style="background:#0f0f10; color:#9ef; padding:10px; border-radius:8px; max-width:920px; overflow:auto; height:160px;">(no data)</pre>

    <div style="margin-top:8px; color:#888; font-size:12px; line-height:1.4;">
      Gym GLB: <code>/models/Untitled_gym.glb</code><br/>
      Anim GLB: <code>/models/Untitled_squat.glb</code>
    </div>
  </div>
`;

const infoEl = document.getElementById("info");
const threeStatusEl = document.getElementById("threeStatus");
const autoRotateEl = document.getElementById("autoRotate");
const THREE_WRAP = document.getElementById("threeWrap");

function logInfo(msg) {
  infoEl.textContent = String(msg);
}

function setThreeStatus(msg) {
  threeStatusEl.textContent = msg;
}

/** =========================
 * Three.js Scene
 * ========================= */
const GYM_URL = "/models/Untitled_gym.glb";
const ANIM_AVATAR_URL = "/models/Untitled_squat.glb";

let renderer = null;
let scene = null;
let camera = null;
let controls = null;

let gymRoot = null;
let avatarScene = null;
let mixer = null;
let currentAction = null;

const clock = new THREE.Clock();
let rafId = null;
let lastRenderTime = 0;
const RENDER_FPS = 30;
const RENDER_INTERVAL = 1000 / RENDER_FPS;

let animationPaused = false;

const AVATAR_POS = new THREE.Vector3(0, 0, 0.8);
const LOOK_TARGET_OFFSET = new THREE.Vector3(0, 1.15, 0);
const FRONT_CAM_OFFSET = new THREE.Vector3(0, 0.35, 2.75);

function getAvatarLookTarget() {
  return AVATAR_POS.clone().add(LOOK_TARGET_OFFSET);
}

function initThreeIfNeeded() {
  if (renderer) return;

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
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
        currentAction = null;
        animationPaused = false;

        if (gltf.animations && gltf.animations.length > 0) {
          mixer = new THREE.AnimationMixer(avatarScene);
          currentAction = mixer.clipAction(gltf.animations[0]);
          currentAction.setLoop(THREE.LoopRepeat);
          currentAction.timeScale = 0.3;
          currentAction.play();
        }

        const target = getAvatarLookTarget();

        camera.position.set(
          target.x + FRONT_CAM_OFFSET.x,
          target.y + FRONT_CAM_OFFSET.y,
          target.z + FRONT_CAM_OFFSET.z
        );
        camera.lookAt(target);

        setThreeStatus("장면 로드 완료");
        logInfo("Gym + Avatar 로드 완료");
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
    setThreeStatus("Loading scene...");

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
 * Buttons
 * ========================= */
document.getElementById("loadScene").addEventListener("click", loadScene);

document.getElementById("toggleAnim").addEventListener("click", () => {
  if (!mixer) {
    setThreeStatus("먼저 Load Scene 하세요");
    return;
  }

  animationPaused = !animationPaused;
  setThreeStatus(animationPaused ? "Animation paused" : "Animation playing");
});

document.getElementById("resetAnim").addEventListener("click", () => {
  if (!mixer || !currentAction) {
    setThreeStatus("먼저 Load Scene 하세요");
    return;
  }

  mixer.setTime(0);
  animationPaused = true;
  setThreeStatus("애니메이션 리셋 완료");
  logInfo("애니메이션을 처음 프레임으로 되돌렸습니다.");
});

/** =========================
 * 첫 안내
 * ========================= */
setThreeStatus("Ready (Load Scene)");
logInfo("Load Scene 버튼을 눌러 아바타 장면을 불러오세요.");