import * as ort from "onnxruntime-web";

const ORT_DIST_BASE = "/ort/";
const INPUT_VIDEO_URL = "/squat.mp4";
const YOLO_MODEL_URL = "/models/yolo_pose.onnx";

ort.env.wasm.wasmPaths = ORT_DIST_BASE;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

export async function createMode({ dom, shared }) {
  let webcamStream = null;
  let yoloSession = null;
  let yoloBusy = false;
  let liveTimer = null;
  let drawTimer = null;
  let stopped = false;

  const YOLO_INPUT_SIZE = 640;
  const YOLO_SCORE_THRESHOLD = 0.25;
  const YOLO_KPT_THRESHOLD = 0.2;
  const YOLO_IOU_THRESHOLD = 0.45;

  const inferCanvas = document.createElement("canvas");
  const inferCtx = inferCanvas.getContext("2d", { willReadFrequently: true });
  const ctx = dom.canvas.getContext("2d", { willReadFrequently: true });
  const octx = dom.overlay.getContext("2d");

  const COCO_EDGES = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [11, 12], [5, 11], [6, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  ];

  function clearOverlay() {
    octx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
  }

  function clamp01(v) {
    return Math.max(0, Math.min(1, v));
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

  async function inferOneFrame() {
    if (stopped || yoloBusy || !yoloSession || !dom.video.videoWidth) return;
    yoloBusy = true;

    try {
      const { tensor, meta } = preprocessForYolo();
      const inputName = yoloSession.inputNames[0];
      const outputs = await yoloSession.run({ [inputName]: tensor });
      const outputName = yoloSession.outputNames[0];
      const outputTensor = outputs[outputName];
      if (!outputTensor) return;

      const kpts = decodeYoloPoseOutput(outputTensor, meta);

      if (!kpts) {
        shared.setFeedback({
          main: "사람을 인식하지 못했습니다",
          detail: "카메라 안으로 몸이 더 잘 들어오도록 조정해주세요.",
          state: "인식 실패",
          pose: "사람 또는 자세가 충분히 검출되지 않았습니다.",
          guide: "어깨, 골반, 무릎이 함께 보이도록 맞춰주세요."
        });
        shared.setDebug("MODE = feedback", "YOLO = no pose");
        clearOverlay();
        return;
      }

      drawSkeleton2D(kpts);

      shared.setFeedback({
        main: "자세 분석 중",
        detail: "웹캠과 기준 영상을 비교하는 모드입니다.",
        state: "YOLO 분석 중",
        pose: "현재 전신 자세를 인식했습니다.",
        guide: "기준 영상을 보며 자세를 맞춰주세요."
      });

      shared.setDebug("MODE = feedback", "YOLO = OK");
    } finally {
      yoloBusy = false;
    }
  }

  function startLoops() {
    if (drawTimer) clearInterval(drawTimer);
    drawTimer = setInterval(() => {
      if (stopped) return;
      drawVideoFrameToCanvas();
    }, 33);

    if (liveTimer) clearInterval(liveTimer);
    liveTimer = setInterval(() => {
      inferOneFrame();
    }, 66);
  }

  async function start() {
    shared.resetViewVisibility();

    dom.video.style.display = "block";
    dom.canvas.style.display = "block";
    dom.overlay.style.display = "block";
    dom.threeWrap.style.display = "none";
    dom.guideCard.style.display = "block";

    dom.modeChip.textContent = "피드백 모드";
    dom.gestureChip.textContent = "GESTURE: -";
    dom.guideVideo.src = INPUT_VIDEO_URL;
    dom.guideVideo.play().catch(() => {});

    shared.setStatus("LOADING...");
    shared.setInfo("피드백 모드", "웹캠 + 기준 영상");
    shared.setHelp("피드백 모드 = 웹캠 + 기준 영상");
    shared.setFeedback({
      main: "로딩 중",
      detail: "웹캠과 YOLO를 준비하고 있습니다.",
      state: "시스템 준비 중",
      pose: "잠시만 기다려주세요.",
      guide: "피드백 모드 초기화 중입니다."
    });

    await startWebcam();
    await loadYoloModel();
    startLoops();

    shared.setStatus("READY");
  }

  async function destroy() {
    stopped = true;

    if (liveTimer) {
      clearInterval(liveTimer);
      liveTimer = null;
    }

    if (drawTimer) {
      clearInterval(drawTimer);
      drawTimer = null;
    }

    if (webcamStream) {
      webcamStream.getTracks().forEach((t) => t.stop());
      webcamStream = null;
    }

    try { dom.video.pause(); } catch {}
    dom.video.srcObject = null;
    dom.video.removeAttribute("src");

    try { dom.guideVideo.pause(); } catch {}
    dom.guideVideo.removeAttribute("src");

    clearOverlay();
    ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);

    yoloSession = null;
  }

  return { start, destroy };
}