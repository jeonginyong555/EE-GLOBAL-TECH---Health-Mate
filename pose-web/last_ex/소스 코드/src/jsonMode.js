import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const INPUT_VIDEO_URL = "/squat.mp4";
const INPUT_POSE_JSON_URL = "/pose_squat.json";
const GYM_URL = "/models/Untitled_gym.glb";
const ANIM_AVATAR_URL = "/models/Untitled_squat.glb";

export async function createMode({ dom, shared }) {
  let renderer = null;
  let scene = null;
  let camera = null;
  let controls = null;
  let gymRoot = null;
  let avatarScene = null;
  let mixer = null;
  let squatClip = null;
  let squatAction = null;
  let rafId = 0;
  let lastRenderTime = 0;
  let poseData = null;
  let poseFPS = 30;
  let syncTimer = null;
  let stopped = false;

  const RENDER_FPS = 20;
  const RENDER_INTERVAL = 1000 / RENDER_FPS;

  const AVATAR_POS = new THREE.Vector3(0, 0, 0.8);
  const LOOK_TARGET_OFFSET = new THREE.Vector3(0, 0.9, 0);
  const FRONT_CAM_OFFSET = new THREE.Vector3(0, 0.55, 3.4);
  const CLIP_START_NORM = 0.0;
  const CLIP_END_NORM = 0.33;

  let squatProgressSmooth = 0;

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

  function updateProgressControlledAnimation() {
    if (!mixer || !squatClip || !squatAction) return;
    const { clipStart, usableDuration } = getClipRange();
    const targetTime = clipStart + usableDuration * squatProgressSmooth;
    squatAction.time = targetTime;
    mixer.update(0);
    avatarScene?.updateMatrixWorld(true);
  }

  function initThree() {
    if (renderer) return;

    renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(1);
    renderer.setSize(dom.threeWrap.clientWidth || 640, dom.threeWrap.clientHeight || 480);
    dom.threeWrap.innerHTML = "";
    dom.threeWrap.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(shared.themeLight() ? 0xf4f7f6 : 0x0a0a0c);

    camera = new THREE.PerspectiveCamera(
      52,
      (dom.threeWrap.clientWidth || 640) / Math.max(1, (dom.threeWrap.clientHeight || 480)),
      0.1,
      2000
    );
    camera.position.set(0, 1.3, 2.6);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444466, 1.15);
    hemi.position.set(0, 2, 0);
    scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(2, 4, 2);
    scene.add(dir);
  }

  async function loadPoseJson() {
    const res = await fetch(INPUT_POSE_JSON_URL);
    poseData = await res.json();
    poseFPS = poseData.fps || 30;
  }

  async function loadGym(loader) {
    return new Promise((resolve, reject) => {
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
    return new Promise((resolve, reject) => {
      loader.load(
        ANIM_AVATAR_URL,
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
          }

          const target = getAvatarLookTarget();
          camera.position.set(
            target.x + FRONT_CAM_OFFSET.x,
            target.y + FRONT_CAM_OFFSET.y,
            target.z + FRONT_CAM_OFFSET.z
          );
          camera.lookAt(target);
          resolve();
        },
        undefined,
        reject
      );
    });
  }

  function computeDepthFromFrame(frame) {
    if (!frame?.valid || !frame.keypoints) return 0;
    const k = frame.keypoints;
    const lh = k[11], rh = k[12], lk = k[13], rk = k[14];
    if (!lh || !rh || !lk || !rk) return 0;

    const hipY = (lh[1] + rh[1]) * 0.5;
    const kneeY = (lk[1] + rk[1]) * 0.5;
    const d = 1 - Math.max(0, kneeY - hipY) / 0.35;
    return Math.max(0, Math.min(1, d));
  }

  function startJsonSync() {
    if (syncTimer) clearInterval(syncTimer);

    syncTimer = setInterval(() => {
      if (stopped) return;
      if (!dom.video.duration || !poseData?.frames?.length) return;

      const idx = Math.min(
        poseData.frames.length - 1,
        Math.max(0, Math.floor((dom.video.currentTime || 0) * poseFPS))
      );
      const frame = poseData.frames[idx];
      const depth = computeDepthFromFrame(frame);

      squatProgressSmooth = depth;

      shared.setFeedback({
        main: "정석 시범 재생 중",
        detail: "정석 비디오와 아바타를 동기화하고 있습니다.",
        state: "JSON 분석 중",
        pose: `재생 시간 ${dom.video.currentTime.toFixed(2)}s`,
        guide: "발표용 정석 모드입니다."
      });

      shared.setDebug(
        "MODE = standard",
        `videoTime = ${dom.video.currentTime.toFixed(2)}s`,
        `poseFPS = ${poseFPS}`,
        `depth = ${depth.toFixed(3)}`
      );
    }, 33);
  }

  function animate(now = 0) {
    if (stopped) return;
    rafId = requestAnimationFrame(animate);

    controls?.update();
    updateProgressControlledAnimation();

    if (now - lastRenderTime < RENDER_INTERVAL) return;
    lastRenderTime = now;

    if (renderer && scene && camera && dom.threeWrap.style.display !== "none") {
      renderer.render(scene, camera);
    }
  }

  async function start() {
    shared.resetViewVisibility();
    dom.video.style.display = "block";
    dom.canvas.style.display = "none";
    dom.overlay.style.display = "none";
    dom.threeWrap.style.display = "block";
    dom.guideCard.style.display = "none";

    dom.modeChip.textContent = "정석 모드";
    dom.gestureChip.textContent = "GESTURE: -";

    shared.setStatus("LOADING...");
    shared.setInfo("정석 모드", "비디오 + 아바타 동기화");
    shared.setHelp("정석 모드 = 기준 비디오 + 아바타");

    dom.video.srcObject = null;
    dom.video.src = INPUT_VIDEO_URL;
    dom.video.muted = true;
    dom.video.loop = true;
    dom.video.playsInline = true;
    await dom.video.play().catch(() => {});

    await loadPoseJson();
    initThree();

    const loader = new GLTFLoader();
    await loadGym(loader);
    await loadAnimatedAvatar(loader);

    startJsonSync();
    animate();

    shared.setStatus("READY");
  }

  async function destroy() {
    stopped = true;

    if (syncTimer) {
      clearInterval(syncTimer);
      syncTimer = null;
    }

    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }

    try { dom.video.pause(); } catch {}
    dom.video.removeAttribute("src");
    dom.video.srcObject = null;

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

  return { start, destroy };
}