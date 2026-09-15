"""仅监听本机的课程演示工具；数据处理调用真实被测类。"""
import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SteelRailWay"))
from datasets.rail_dataset import RailDualModalDataset


def png(array):
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def gray(array):
    low, high = float(array.min()), float(array.max())
    return np.clip((array - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)


def generate(root, view, abnormal):
    """生成同一坐标系内的 RGB / 深度 / 人工设定 GT；不是模型输出。"""
    y, x = np.mgrid[:640, :320]
    rail = (x > 68) & (x < 252)
    surface = np.clip(65 + 90 * rail + 12 * np.sin(x / 8) + 5 * np.cos(y / 13), 0, 255).astype(np.uint8)
    rgb = np.stack([surface, surface, surface], axis=-1)
    depth = (800 + 180 * rail + y // 5 + x // 9).astype(np.uint16)
    mask = np.zeros((640, 320), np.uint8)
    if abnormal:
        cv2.line(mask, (118, 190), (204, 252), 255, 13)
        cv2.ellipse(mask, (154, 464), (26, 16), -20, 0, 360, 255, -1)
        rgb[mask > 0] = (40, 48, 55)
        depth[mask > 0] -= 160
    category = "broken" if abnormal else "good"
    rgb_path = root / f"rail_mvtec/cam{view}/test/{category}/demo.jpg"
    depth_path = root / f"rail_mvtec_depth/cam{view}/test/{category}/demo.tiff"
    gt_path = root / f"rail_mvtec/cam{view}/ground_truth/broken/demo.png"
    for path, data in [(rgb_path, rgb), (depth_path, depth)] + ([(gt_path, mask)] if abnormal else []):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), data):
            raise RuntimeError(f"无法写入临时样本: {path}")
    return cv2.imread(str(rgb_path)), depth, mask, depth_path


def render(options):
    view = int(options.get("view", 1))
    size = int(options.get("size", 160))
    stride = int(options.get("stride", 120))
    output = int(options.get("output", 256))
    index = int(options.get("index", 0))
    norm = options.get("norm", "zscore")
    sample = options.get("sample", "broken")
    # 限制本地演示的内存占用；view_id 的业务校验交给被测类。
    if size not in (80, 160, 240) or stride not in (80, 120, 160) or output not in (128, 256):
        raise ValueError("请选择页面提供的 patch、步长和输出尺寸")
    if sample not in ("broken", "good", "corrupt"):
        raise ValueError("未知样本")
    runtime = Path(__file__).parent / ".runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rail-demo-", dir=runtime) as folder:
        root = Path(folder)
        rgb, depth, mask, depth_path = generate(root, view, sample != "good")
        dataset = RailDualModalDataset(str(root), str(root), view, split="test", img_size=output,
                                       patch_size=size, patch_stride=stride, depth_norm=norm, preload=False)
        if not len(dataset):
            raise ValueError("当前被测类的 test 扫描逻辑仅扫描 cam1–6；view_id=7/8 参数合法，但返回空数据集。")
        if not 0 <= index < len(dataset):
            raise ValueError(f"patch 索引应在 0–{len(dataset)-1} 之间")
        if sample == "corrupt":
            depth_path.write_bytes(b"deliberately invalid TIFF for demonstration")
        item = dataset[index]
        left, top = (320 - size) // 2, index * stride
        source_images = []
        for source in (rgb, cv2.cvtColor(gray(depth.astype(float)), cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)):
            source = source.copy()
            cv2.rectangle(source, (left, top), (left + size - 1, top + size - 1), (180, 220, 70), 3)
            source_images.append(png(source))
        intensity = item["intensity"].numpy()
        display_rgb = np.clip((intensity * np.array([.229, .224, .225])[:, None, None]
                               + np.array([.485, .456, .406])[:, None, None]) * 255, 0, 255)
        display_rgb = display_rgb.transpose(1, 2, 0).astype(np.uint8)[:, :, ::-1].copy()
        tensors = {name: {"shape": list(item[name].shape), "dtype": str(item[name].dtype),
                          "min": float(item[name].min()), "max": float(item[name].max()),
                          "finite": bool(item[name].isfinite().all())} for name in ("intensity", "depth", "gt")}
        return {"sources": source_images, "patches": [png(display_rgb), png(gray(item["depth"][0].numpy())),
                                                     png((item["gt"].numpy() * 255).astype(np.uint8))],
                "count": len(dataset), "index": index, "box": [left, top, size, size], "tensors": tensors,
                "label": item["label"], "view_id": item["view_id"], "frame_id": item["frame_id"],
                "gt_pixels": int(item["gt"].sum()), "gt_values": item["gt"].unique().tolist()}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.reply(200, (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.reply(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/render":
            self.reply(404, b"Not found", "text/plain")
            return
        # 拒绝从其他站点发起的请求。
        if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
            self.reply(403, b"Forbidden", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 4096:
                raise ValueError("请求体大小无效")
            result = render(json.loads(self.rfile.read(length)))
            status = 200
        except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
            result, status = {"error": type(exc).__name__, "message": str(exc)}, 400
        except Exception as exc:
            result, status = {"error": type(exc).__name__, "message": str(exc)}, 500
        self.reply(status, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def reply(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"演示页面: http://127.0.0.1:{args.port}  |  Ctrl+C 停止", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
