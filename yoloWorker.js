/* eslint-disable no-restricted-globals */
import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = {
  "ort-wasm-simd-threaded.wasm": "/ort/ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.jsep.wasm": "/ort/ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.asyncify.wasm":
    "/ort/ort-wasm-simd-threaded.asyncify.wasm",
};

let ortInited = false;
let session = null;
let busy = false;

async function ensureOrtInit() {
  if (ortInited) return;

  const res = await fetch("/ort/ort-wasm-simd-threaded.jsep.wasm");
  if (!res.ok) {
    throw new Error(`Failed to fetch wasm: ${res.status} ${res.statusText}`);
  }

  const buf = await res.arrayBuffer();
  ort.env.wasm.wasmBinary = new Uint8Array(buf);
  ort.env.wasm.numThreads = 1;
  ortInited = true;
}

function decodeYoloV8Pose(outputTensor) {
  const data = outputTensor.data;
  const [, , N] = outputTensor.dims;
  const get = (c, i) => data[c * N + i];

  let bestIdx = -1;
  let bestScore = -1;

  for (let i = 0; i < N; i++) {
    const conf = get(4, i);
    if (conf > bestScore) {
      bestScore = conf;
      bestIdx = i;
    }
  }

  if (bestIdx === -1 || bestScore < 0.25) return null;

  const kptStart = 5;
  const kpts = [];
  for (let k = 0; k < 17; k++) {
    const x = get(kptStart + k * 3 + 0, bestIdx);
    const y = get(kptStart + k * 3 + 1, bestIdx);
    const s = get(kptStart + k * 3 + 2, bestIdx);
    kpts.push({ x, y, s });
  }

  return { conf: bestScore, kpts };
}

function maybeDenormalizeTo640(kpts) {
  let maxX = 0;
  let maxY = 0;

  for (const p of kpts) {
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }

  if (maxX <= 1.5 && maxY <= 1.5) {
    return kpts.map((p) => ({
      x: p.x * 640,
      y: p.y * 640,
      s: p.s,
    }));
  }

  return kpts;
}

function makeTensorFromImageData(imageData, width, height) {
  const dstSize = 640;

  const srcCanvas = new OffscreenCanvas(width, height);
  const sctx = srcCanvas.getContext("2d");
  sctx.putImageData(imageData, 0, 0);

  const dstCanvas = new OffscreenCanvas(dstSize, dstSize);
  const dctx = dstCanvas.getContext("2d");

  const r = Math.min(dstSize / width, dstSize / height);
  const newW = Math.round(width * r);
  const newH = Math.round(height * r);
  const padX = Math.floor((dstSize - newW) / 2);
  const padY = Math.floor((dstSize - newH) / 2);

  dctx.fillStyle = "rgb(114,114,114)";
  dctx.fillRect(0, 0, dstSize, dstSize);
  dctx.drawImage(srcCanvas, 0, 0, width, height, padX, padY, newW, newH);

  const img = dctx.getImageData(0, 0, dstSize, dstSize);
  const data = img.data;

  const input = new Float32Array(1 * 3 * dstSize * dstSize);
  const plane = dstSize * dstSize;

  for (let i = 0; i < plane; i++) {
    const rVal = data[i * 4 + 0] / 255;
    const gVal = data[i * 4 + 1] / 255;
    const bVal = data[i * 4 + 2] / 255;

    input[i] = rVal;
    input[i + plane] = gVal;
    input[i + 2 * plane] = bVal;
  }

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, 640, 640]),
    r,
    padX,
    padY,
  };
}

function yoloKpts640ToCanvas(kpts640, srcW, srcH, r, padX, padY) {
  const out = kpts640.map((p) => {
    const x = (p.x - padX) / r;
    const y = (p.y - padY) / r;
    return { x, y, s: p.s };
  });

  for (const p of out) {
    p.x = Math.max(0, Math.min(srcW, p.x));
    p.y = Math.max(0, Math.min(srcH, p.y));
  }

  return out;
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  try {
    if (type === "init") {
      await ensureOrtInit();
      session = await ort.InferenceSession.create(payload.modelUrl);

      self.postMessage({
        type: "inited",
      });
      return;
    }

    if (type === "infer") {
      if (!session || busy) return;
      busy = true;

      const { imageData, width, height } = payload;
      const t0 = performance.now();

      const { tensor, r, padX, padY } = makeTensorFromImageData(
        imageData,
        width,
        height
      );

      const outputs = await session.run({ images: tensor });
      const t1 = performance.now();

      const out = outputs.output0;
      const decoded = decodeYoloV8Pose(out);

      if (!decoded) {
        self.postMessage({
          type: "inferResult",
          payload: {
            ok: false,
            inferMs: t1 - t0,
          },
        });
        busy = false;
        return;
      }

      const kpts640 = maybeDenormalizeTo640(decoded.kpts);
      const kpts = yoloKpts640ToCanvas(kpts640, width, height, r, padX, padY);

      self.postMessage({
        type: "inferResult",
        payload: {
          ok: true,
          conf: decoded.conf,
          inferMs: t1 - t0,
          kpts,
        },
      });

      busy = false;
      return;
    }
  } catch (err) {
    busy = false;
    self.postMessage({
      type: "error",
      payload: {
        message: err.message || String(err),
      },
    });
  }
};