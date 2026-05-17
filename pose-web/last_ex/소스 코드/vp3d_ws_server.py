import asyncio
import json
import numpy as np
import websockets

import torch
from common.model import TemporalModel
from common.camera import normalize_screen_coordinates

# -----------------------------
# 설정
# -----------------------------
HOST = "127.0.0.1"
PORT = 8765

CHECKPOINT_PATH = "checkpoint/pretrained_h36m_detectron_coco.bin"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 243  # VP3D receptive field에 맞추면 안정적
buffer_2d = []  # [ [17,2], ... ]


def load_vp3d_model():
    model = TemporalModel(
        17,  # joints
        2,   # input 2D
        17,  # output joints
        filter_widths=[3, 3, 3, 3, 3],
        causal=False,
        dropout=0.25,
        channels=1024
    ).to(DEVICE)

    chk = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(chk["model_pos"])
    model.eval()
    print("[VP3D] Loaded checkpoint:", CHECKPOINT_PATH)
    return model


MODEL = load_vp3d_model()


def vp3d_infer(kpts2d_seq):
    """
    kpts2d_seq: np.ndarray [T, 17, 2]  (pixel coordinates in 480x360 space)
    return: np.ndarray [17, 3] (last frame 3D)
    """
    # ✅ VP3D 입력은 -1~1 범위 정규화가 필요
    # normalize_screen_coordinates: w,h 기준으로 정규화
    kpts2d_norm = normalize_screen_coordinates(kpts2d_seq, w=480, h=360)

    inp = torch.from_numpy(kpts2d_norm.astype(np.float32)).unsqueeze(0).to(DEVICE)  # [1,T,17,2]

    with torch.no_grad():
        pred3d = MODEL(inp)  # [1, T, 17, 3]
        pred3d = pred3d.cpu().numpy()

    return pred3d[0, -1]  # last frame [17,3]


async def handler(websocket):
    global buffer_2d

    print("[WS] Client connected")

    try:
        async for msg in websocket:
            print("[WS] got TEXT message len:", len(msg))
            print("[WS] text preview:", msg[:200] + ("..." if len(msg) > 200 else ""))

            data = json.loads(msg)

            if data.get("type") != "pose2d":
                continue

            kpts = data["kpts"]  # [{x,y,s}*17]
            pts = np.array([[p["x"], p["y"]] for p in kpts], dtype=np.float32)  # [17,2]

            # 간단 sanity log
            mn = pts.min(axis=0).tolist()
            mx = pts.max(axis=0).tolist()

            buffer_2d.append(pts)

            # 버퍼 길이 유지
            if len(buffer_2d) > SEQ_LEN:
                buffer_2d = buffer_2d[-SEQ_LEN:]

            print(f"[WS] pose2d OK: min={mn} max={mx} buffer={len(buffer_2d)}/{SEQ_LEN}")

            # ✅ 핵심: 아직 SEQ_LEN이 안 차도 "패딩"해서 즉시 3D 뽑기
            if len(buffer_2d) < SEQ_LEN:
                last = buffer_2d[-1]
                pad_count = SEQ_LEN - len(buffer_2d)
                pad = [last] * pad_count
                seq = np.stack(pad + buffer_2d, axis=0)  # [SEQ_LEN,17,2]
                # status도 같이 보내고 싶으면 아래 주석 해제
                # await websocket.send(json.dumps({
                #     "type": "status",
                #     "msg": f"padding... {len(buffer_2d)}/{SEQ_LEN}"
                # }))
            else:
                seq = np.stack(buffer_2d[-SEQ_LEN:], axis=0)

            # 3D 추론
            pred3d = vp3d_infer(seq)  # [17,3]

            # 웹으로 전송
            await websocket.send(json.dumps({
                "type": "pose3d",
                "kpts3d": pred3d.tolist()
            }))
            print("[WS] sent pose3d (17x3)")

    except websockets.exceptions.ConnectionClosed:
        print("[WS] Client disconnected")
    except Exception as e:
        print("[WS] ERROR:", repr(e))


async def main():
    print(f"[WS] Running on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT, max_size=2**23):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
