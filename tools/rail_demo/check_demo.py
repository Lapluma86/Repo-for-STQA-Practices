"""演示工具的 HTTP 验证，不计入课程模块一用例数。先启动 server.py。"""
import base64
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import cv2
import numpy as np

base = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8765")


def request(options, expected=200):
    req = Request(base + "/api/render", data=json.dumps(options).encode(), headers={"Content-Type": "application/json"})
    try:
        response = urlopen(req, timeout=30)
    except HTTPError as exc:
        response = exc
    with response:
        assert response.status == expected, response.status
        return json.load(response)


def decode(data):
    return cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",")[1]), np.uint8), cv2.IMREAD_GRAYSCALE)


mask = np.zeros((640, 320), np.uint8)
cv2.line(mask, (118, 190), (204, 252), 255, 13)
cv2.ellipse(mask, (154, 464), (26, 16), -20, 0, 360, 255, -1)
for index in range(5):
    data = request({"index": index})
    expected = cv2.resize(mask[index * 120:index * 120 + 160, 80:240], (256, 256), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(decode(data["patches"][2]), expected)
    assert data["count"] == 5 and data["tensors"]["depth"]["shape"] == [3, 256, 256]
    assert data["gt_pixels"] == int((expected > 0).sum())
print("PASS: 5 patches 的 GT 像素级对齐与张量形状")
for norm in ("zscore", "minmax", "log"):
    data = request({"norm": norm, "index": 1})
    assert all(t["finite"] for t in data["tensors"].values())
    if norm == "minmax":
        assert 0 <= data["tensors"]["depth"]["min"] <= data["tensors"]["depth"]["max"] <= 1
print("PASS: 三种归一化与有限值")
for size in (80, 160, 240):
    for stride in (80, 120, 160):
        data = request({"size": size, "stride": stride, "output": 128})
        assert data["count"] == (640-size)//stride+1
        assert decode(data["patches"][0]).shape == (128, 128)
print("PASS: 9 种裁剪参数组合与输出尺寸")
data = request({"sample": "good"})
assert data["label"] == 0 and data["gt_pixels"] == 0
for view in (0, 9):
    data = request({"view": view}, 400)
    assert data["error"] == "ValueError" and "between 1 and 8" in data["message"]
for view in (7, 8):
    assert "空数据集" in request({"view": view}, 400)["message"]
assert request({"sample": "corrupt"}, 400)["error"] == "FileNotFoundError"
assert request({"index": 99}, 400)["error"] == "ValueError"
assert request({})["count"] == 5
print("PASS: 正常样本、非法视角、空集、损坏文件、越界与异常恢复")
