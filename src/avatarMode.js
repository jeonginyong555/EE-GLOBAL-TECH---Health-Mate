import * as THREE from "three";
import * as ort from "onnxruntime-web";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const ORT_DIST_BASE = "/ort/";
const YOLO_MODEL_URL = "/models/yolo_pose.onnx";
const HAND_MODEL_URL = "/models/hand_landmarker.task";
const GYM_URL = "/models/Untitled_gym.glb";
const ANIM_AVATAR_URL = "/models/Untitled_squat.glb";

ort.env.wasm.wasmPaths = ORT_DIST_BASE;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

export async function createMode({ dom, shared }) {
  let stopped = false;
  let webcamStream = null;

  let yoloSession = null;
  let yoloBusy = false;

  let handLandmarker = null;
  let handBusy = false;
  let lastHandResult = null;

  let renderer = null;
  let scene = null;
  let camera = null;
  let controls = null;
  let avatarScene = null;
  let gymRoot = null;
  let mixer = null;
  let rafId = 0;

  let avatarDisplayMode = "avatar";
  let fistHoldCount = 0;
  let fistCooldown = false;

  const inferCanvas = document.createElement("canvas");
  const inferCtx = inferCanvas.getContext("2d", { willReadFrequently: true });
  const ctx = dom.canvas.getContext("2d", { willReadFrequently: true });
  const octx = dom.overlay.getContext("2d");

  const COCO_EDGES = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [11, 12], [5, 11], [6, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  ];

  const HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
  ];

  const YOLO_INPUT_SIZE = 640;
  const YOLO_SCORE_THRESHOLD = 0.25;
  const YOLO_KPT_THRESHOLD = 0.2;
  const YOLO_IOU_THRESHOLD = 0.45;

  const AVATAR_POS = new THREE.Vector3(0, 0, 0.8);
  const LOOK_TARGET_OFFSET = new THREE.Vector3(0, 0.9, 0);
  const FRONT_CAM_OFFSET = new THREE.Vector3(0, 0.55, 3.4);

  let handCurrentYaw = 0;
  let handRotateState = "NEUTRAL";
  let lastOpenPalm = false;
  const HAND_YAW_LIMIT = Math.PI * 0.95;
  const HAND_ROTATE_SPEED = 0.03;

  function clearOverlay() {
    octx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
  }

  function clamp01(v) {
    return Math.max(0, Math.min(1, v));
  }

  function resizeStageToVideo(videoWidth, videoHeight) {
    const w = Math.max(1, videoWidth);
    const h = Math.max(1, videoHeight);
    dom.canvas.width = w;
    dom.canvas.height = h;
    dom.overlay.width = w;
    dom.overlay.height = h;
  }

  function drawVideoFrameToCanvas() {
    if (!dom.video.videoWidth) return;
    ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
    ctx.drawImage(dom.video, 0, 0, dom.canvas.width, dom.canvas.height);
  }

  async function startWebcam() {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });

    dom.video.src = "";
    dom.video.srcObject = webcamStream;
    dom.video.muted = true;
    dom.video.playsInline = true;

    await new Promise((resolve) => {
      dom.video.onloadedmetadata = () => resolve();
    });

    resizeStageToVideo(dom.video.videoWidth || 640, dom.video.videoHeight || 480);
    await dom.video.play().catch(() => {});
  }

  async function loadYoloModel() {
    yoloSession = await ort.InferenceSession.create(YOLO_MODEL_URL, {
      executionProviders: ["wasm"],
    });
  }

  async function loadHandsModel() {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });
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
        if (iouXYXY(cur, sorted[i]) > iouThreshold) sorted.splice(i, 1);
      }
    }
    return keep;
  }

  function preprocessForYolo() {
    const srcW = dom.video.videoWidth || dom.canvas.width || 640;
    const srcH = dom.video.videoHeight || dom.canvas.height || 640;

    inferCanvas.width = YOLO_INPUT_SIZE;
    inferCanvas.height = YOLO_INPUT_SIZE;

    inferCtx.fillStyle = "black";
    inferCtx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);

    const scale = Math.min(YOLO_INPUT_SIZE / srcW, YOLO_INPUT_SIZE / srcH);
    const drawW = Math.round(srcW * scale);
    const drawH = Math.round(srcH * scale);
    const padX = Math.floor((YOLO_INPUT_SIZE - drawW) / 2);
    const padY = Math.floor((YOLO_INPUT_SIZE - drawH) / 2);

    inferCtx.drawImage(dom.video, 0, 0, srcW, srcH, padX, padY, drawW, drawH);

    const imageData = inferCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
    const { data } = imageData;

    const chw = new Float32Array(1 * 3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
    const area = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;

    for (let i = 0; i < area; i++) {
      chw[i] = data[i * 4 + 0] / 255;
      chw[area + i] = data[i * 4 + 1] / 255;
      chw[area * 2 + i] = data[i * 4 + 2] / 255;
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

    if (dims[1] === 56) {
      featureSize = dims[1];
      count = dims[2];
      for (let i = 0; i < count; i++) {
        const row = new Float32Array(featureSize);
        for (let j = 0; j < featureSize; j++) row[j] = arr[j * count + i];
        rows.push(row);
      }
    } else if (dims[2] === 56) {
      count = dims[1];
      featureSize = dims[2];
      for (let i = 0; i < count; i++) {
        rows.push(arr.slice(i * featureSize, (i + 1) * featureSize));
      }
    } else {
      return null;
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
        kpts.push([row[base], row[base + 1], row[base + 2]]);
      }

      candidates.push({ x1, y1, x2, y2, score, kpts });
    }

    if (!candidates.length) return null;
    const kept = nmsBoxes(candidates, YOLO_IOU_THRESHOLD);
    if (!kept.length) return null;

    const { srcW, srcH, scale, padX, padY } = meta;
    const best = kept[0];

    return best.kpts.map(([x, y, s]) => {
      const rx = (x - padX) / Math.max(1e-6, scale);
      const ry = (y - padY) / Math.max(1e-6, scale);
      return [
        clamp01(rx / Math.max(1, srcW)),
        clamp01(ry / Math.max(1, srcH)),
        s >= YOLO_KPT_THRESHOLD ? s : 0,
      ];
    });
  }

  function drawSkeleton2D(kpts) {
    if (avatarDisplayMode === "avatar") {
      clearOverlay();
      return;
    }

    clearOverlay();

    const w = dom.overlay.width;
    const h = dom.overlay.height;
    octx.lineWidth = 2;
    octx.strokeStyle = "lime";
    octx.fillStyle = "red";

    for (const [a, b] of COCO_EDGES) {
      const pa = kpts[a];
      const pb = kpts[b];
      if (!pa || !pb) continue;
      if ((pa[2] ?? 0) < 0.05 || (pb[2] ?? 0) < 0.05) continue;

      octx.beginPath();
      octx.moveTo(pa[0] * w, pa[1] * h);
      octx.lineTo(pb[0] * w, pb[1] * h);
      octx.stroke();
    }

    for (const p of kpts) {
      if (!p || (p[2] ?? 0) < 0.05) continue;
      octx.beginPath();
      octx.arc(p[0] * w, p[1] * h, 3, 0, Math.PI * 2);
      octx.fill();
    }
  }

  function handLmOk(p) {
    return !!p && Number.isFinite(p.x) && Number.isFinite(p.y);
  }

  function drawHandOverlay() {
    if (avatarDisplayMode === "avatar") return;
    if (!lastHandResult?.landmarks?.length) return;

    const lm = lastHandResult.landmarks[0];
    const w = dom.overlay.width;
    const h = dom.overlay.height;

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

  function isFingerCurled(lm, tipIdx, pipIdx) {
    const tip = lm?.[tipIdx];
    const pip = lm?.[pipIdx];
    if (!tip || !pip) return false;
    return tip.y > pip.y;
  }

  function isFistLandmarks(lm) {
    if (!lm || lm.length < 21) return false;
    return (
      isFingerCurled(lm, 8, 6) &&
      isFingerCurled(lm, 12, 10) &&
      isFingerCurled(lm, 16, 14) &&
      isFingerCurled(lm, 20, 18)
    );
  }

  function handleFistToggle() {
    const lm = lastHandResult?.landmarks?.[0] || null;
    const fist = isFistLandmarks(lm);

    if (fist) {
      fistHoldCount += 1;
      dom.gestureChip.textContent = `GESTURE: FIST ${fistHoldCount}/5`;
    } else {
      fistHoldCount = 0;
      dom.gestureChip.textContent = "GESTURE: OPEN";
    }

    if (fistHoldCount >= 5 && !fistCooldown) {
      avatarDisplayMode = avatarDisplayMode === "avatar" ? "webcam" : "avatar";
      applyDisplayMode();

      fistCooldown = true;
      fistHoldCount = 0;

      setTimeout(() => {
        fistCooldown = false;
      }, 1000);
    }
  }

  async function inferHands() {
    if (handBusy || !handLandmarker || !dom.video.videoWidth) return;
    handBusy = true;
    try {
      const ts = performance.now();
      lastHandResult = handLandmarker.detectForVideo(dom.video, ts) || null;
      handleFistToggle();
    } finally {
      handBusy = false;
    }
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
    ) return "NEUTRAL";

    const palmDx = pinkyMcp.x - indexMcp.x;
    const palmDy = pinkyMcp.y - indexMcp.y;
    const fingerDx = middleTip.x - middleMcp.x;
    const fingerDy = middleTip.y - middleMcp.y;

    const cross = palmDx * fingerDy - palmDy * fingerDx;
    if (Math.abs(cross) < 0.008) return "NEUTRAL";
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
    if (!lastOpenPalm) {
      handRotateState = "NEUTRAL";
      return;
    }

    handRotateState = getHandFacingDirection(lm);
  }

  function getAvatarLookTarget() {
    return AVATAR_POS.clone().add(LOOK_TARGET_OFFSET);
  }

  function getBaseCameraPositionByYaw(yawRad) {
    const target = getAvatarLookTarget();
    const r = Math.sqrt(
      FRONT_CAM_OFFSET.x * FRONT_CAM_OFFSET.x +
      FRONT_CAM_OFFSET.z * FRONT_CAM_OFFSET.z
    );
    const x = target.x + Math.sin(yawRad) * r;
    const z = target.z + Math.cos(yawRad) * r;
    const y = target.y + FRONT_CAM_OFFSET.y;
    return { x, y, z, target };
  }

  function applyHandCameraControl() {
    if (!camera || !avatarScene || !handLandmarker || !dom.video.videoWidth) return;

    updateHandRotateState();

    if (handRotateState === "LEFT") handCurrentYaw -= HAND_ROTATE_SPEED;
    else if (handRotateState === "RIGHT") handCurrentYaw += HAND_ROTATE_SPEED;

    handCurrentYaw = Math.max(-HAND_YAW_LIMIT, Math.min(HAND_YAW_LIMIT, handCurrentYaw));

    const camPos = getBaseCameraPositionByYaw(handCurrentYaw);
    camera.position.set(camPos.x, camPos.y, camPos.z);
    camera.lookAt(camPos.target);
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
            const clip = clips[0];
            const action = mixer.clipAction(clip);
            action.play();
            action.paused = false;
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

  async function inferYolo() {
    if (yoloBusy || !yoloSession || !dom.video.videoWidth) return;
    yoloBusy = true;

    try {
      const srcW = dom.video.videoWidth || dom.canvas.width || 640;
      const srcH = dom.video.videoHeight || dom.canvas.height || 640;

      inferCanvas.width = YOLO_INPUT_SIZE;
      inferCanvas.height = YOLO_INPUT_SIZE;

      inferCtx.fillStyle = "black";
      inferCtx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);

      const scale = Math.min(YOLO_INPUT_SIZE / srcW, YOLO_INPUT_SIZE / srcH);
      const drawW = Math.round(srcW * scale);
      const drawH = Math.round(srcH * scale);
      const padX = Math.floor((YOLO_INPUT_SIZE - drawW) / 2);
      const padY = Math.floor((YOLO_INPUT_SIZE - drawH) / 2);

      inferCtx.drawImage(dom.video, 0, 0, srcW, srcH, padX, padY, drawW, drawH);

      const imageData = inferCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
      const { data } = imageData;

      const chw = new Float32Array(1 * 3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
      const area = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
      for (let i = 0; i < area; i++) {
        chw[i] = data[i * 4 + 0] / 255;
        chw[area + i] = data[i * 4 + 1] / 255;
        chw[area * 2 + i] = data[i * 4 + 2] / 255;
      }

      const tensor = new ort.Tensor("float32", chw, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
      const inputName = yoloSession.inputNames[0];
      const outputs = await yoloSession.run({ [inputName]: tensor });
      const outputName = yoloSession.outputNames[0];
      const outputTensor = outputs[outputName];
      if (!outputTensor) return;

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
          for (let j = 0; j < featureSize; j++) row[j] = arr[j * count + i];
          rows.push(row);
        }
      } else if (dims[2] === 56) {
        count = dims[1];
        featureSize = dims[2];
        for (let i = 0; i < count; i++) {
          rows.push(arr.slice(i * featureSize, (i + 1) * featureSize));
        }
      }

      const candidates = [];
      for (const row of rows) {
        const score = row[4];
        if (score < YOLO_SCORE_THRESHOLD) continue;

        const cx = row[0];
        const cy = row[1];
        const w = row[2];
        const h = row[3];

        const x1 = cx - w / 2;
        const y1 = cy - h / 2;
        const x2 = cx + w / 2;
        const y2 = cy + h / 2;

        const kpts = [];
        for (let k = 0; k < 17; k++) {
          const base = 5 + k * 3;
          kpts.push([row[base], row[base + 1], row[base + 2]]);
        }

        candidates.push({ x1, y1, x2, y2, score, kpts });
      }

      if (!candidates.length) {
        clearOverlay();
        return;
      }

      const kept = nmsBoxes(candidates, YOLO_IOU_THRESHOLD);
      const best = kept[0];

      const kpts = best.kpts.map(([x, y, s]) => {
        const rx = (x - padX) / Math.max(1e-6, scale);
        const ry = (y - padY) / Math.max(1e-6, scale);
        return [
          clamp01(rx / Math.max(1, srcW)),
          clamp01(ry / Math.max(1, srcH)),
          s >= YOLO_KPT_THRESHOLD ? s : 0,
        ];
      });

      drawSkeleton2D(kpts);
      drawHandOverlay();

      shared.setFeedback({
        main: avatarDisplayMode === "avatar" ? "아바타 모드" : "웹캠 모드",
        detail: "웹캠 추론과 아바타를 함께 유지하고 있습니다.",
        state: "YOLO + Hands 활성화",
        pose: "주먹 5프레임 유지 시 화면 전환",
        guide: "현재 보이지 않는 화면도 내부적으로 계속 유지됩니다."
      });

      shared.setDebug(
        "MODE = avatar",
        `display = ${avatarDisplayMode}`,
        `fistHold = ${fistHoldCount}`
      );
    } finally {
      yoloBusy = false;
    }
  }

  function applyDisplayMode() {
    const showWebcam = avatarDisplayMode === "webcam";
    dom.video.style.display = showWebcam ? "block" : "none";
    dom.canvas.style.display = showWebcam ? "block" : "none";
    dom.overlay.style.display = showWebcam ? "block" : "none";
    dom.threeWrap.style.display = showWebcam ? "none" : "block";
    shared.setInfo(
      `아바타 모드 (${avatarDisplayMode})`,
      "한 창만 표시, 내부 추론은 계속 유지"
    );
  }

  function animate() {
    if (stopped) return;
    rafId = requestAnimationFrame(animate);

    drawVideoFrameToCanvas();
    inferHands();
    inferYolo();

    controls?.update();
    applyHandCameraControl();
    if (mixer) mixer.update(1 / 60);

    if (renderer && scene && camera && dom.threeWrap.style.display !== "none") {
      renderer.render(scene, camera);
    }
  }

  async function start() {
    shared.resetViewVisibility();

    dom.video.style.display = "none";
    dom.canvas.style.display = "none";
    dom.overlay.style.display = "none";
    dom.threeWrap.style.display = "block";
    dom.guideCard.style.display = "none";

    dom.modeChip.textContent = "아바타 모드";
    dom.gestureChip.textContent = "GESTURE: -";

    shared.setStatus("LOADING...");
    shared.setInfo("아바타 모드", "웹캠 + 아바타");
    shared.setHelp("주먹 5프레임 유지 시 webcam / avatar 전환");

    await startWebcam();
    await loadYoloModel();
    await loadHandsModel();

    initThree();
    const loader = new GLTFLoader();
    await loadGym(loader);
    await loadAnimatedAvatar(loader);

    applyDisplayMode();
    animate();

    shared.setStatus("READY");
  }

  async function destroy() {
    stopped = true;

    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }

    if (webcamStream) {
      webcamStream.getTracks().forEach((t) => t.stop());
      webcamStream = null;
    }

    try { dom.video.pause(); } catch {}
    dom.video.srcObject = null;
    dom.video.removeAttribute("src");

    if (renderer) {
      renderer.dispose();
      dom.threeWrap.innerHTML = "";
    }

    clearOverlay();
    ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);

    renderer = null;
    scene = null;
    camera = null;
    controls = null;
    avatarScene = null;
    gymRoot = null;
    mixer = null;
    handLandmarker = null;
    yoloSession = null;
  }

  return {
    start,
    destroy
  };
}