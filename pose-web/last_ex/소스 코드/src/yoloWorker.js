import * as ort from "onnxruntime-web";

let session = null;
let ortReady = false;

const YOLO_INPUT_SIZE = 640;
const YOLO_SCORE_THRESHOLD = 0.23;
const YOLO_KPT_THRESHOLD = 0.18;
const YOLO_IOU_THRESHOLD = 0.45;

const inferCanvas = new OffscreenCanvas(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
const inferCtx = inferCanvas.getContext("2d", { willReadFrequently: true });

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
      if (iouXYXY(cur, sorted[i]) > iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }
  return keep;
}

function preprocessForYolo(imageBitmap, srcW, srcH) {
  inferCtx.fillStyle = "black";
  inferCtx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);

  const scale = Math.min(YOLO_INPUT_SIZE / srcW, YOLO_INPUT_SIZE / srcH);
  const drawW = Math.round(srcW * scale);
  const drawH = Math.round(srcH * scale);
  const padX = Math.floor((YOLO_INPUT_SIZE - drawW) / 2);
  const padY = Math.floor((YOLO_INPUT_SIZE - drawH) / 2);

  inferCtx.drawImage(imageBitmap, 0, 0, srcW, srcH, padX, padY, drawW, drawH);

  const imageData = inferCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  const { data } = imageData;

  const area = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
  const chw = new Float32Array(3 * area);

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
      rows.push(arr.slice(i * featureSize, (i + 1) * featureSize));
    }
  } else {
    throw new Error(`56 feature 출력이 아님: ${JSON.stringify(dims)}`);
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

  if (!candidates.length) return null;

  const kept = nmsBoxes(candidates, YOLO_IOU_THRESHOLD);
  if (!kept.length) return null;

  const best = kept[0];
  const { srcW, srcH, scale, padX, padY } = meta;

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

self.onmessage = async (event) => {
  const msg = event.data;

  try {
    if (msg.type === "init") {
      if (!ortReady) {
        ort.env.wasm.wasmPaths = msg.ortBase || "/ort/";
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
        ortReady = true;
      }

      if (!session) {
        session = await ort.InferenceSession.create(msg.modelUrl, {
          executionProviders: ["wasm"],
        });
      }

      self.postMessage({ type: "ready" });
      return;
    }

    if (msg.type === "infer") {
      if (!session) {
        self.postMessage({ type: "error", error: "session not initialized" });
        return;
      }

      const { imageBitmap, srcW, srcH } = msg;

      if (!imageBitmap) {
        self.postMessage({ type: "error", error: "imageBitmap missing" });
        return;
      }

      const { tensor, meta } = preprocessForYolo(imageBitmap, srcW, srcH);
      imageBitmap.close?.();

      const inputName = session.inputNames[0];
      const outputs = await session.run({ [inputName]: tensor });

      const outputName = session.outputNames[0];
      const outputTensor = outputs[outputName];

      if (!outputTensor) {
        self.postMessage({ type: "inferResult", kpts: null });
        return;
      }

      const kpts = decodeYoloPoseOutput(outputTensor, meta);
      self.postMessage({ type: "inferResult", kpts: kpts || null });
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      error: err?.message || String(err),
    });
  }
};