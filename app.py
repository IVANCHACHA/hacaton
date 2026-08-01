#!/usr/bin/env python3
"""Веб-интерфейс: стрим с iPhone + моменты распознавания.

Continuity Camera на части macOS отдаёт чёрные кадры (баг/ограничение).
Надёжный путь: открыть /phone на iPhone — камера через Safari → Mac.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from camera import find_iphone_source, open_capture
from classify import (
    DEFAULT_ROI_QUAD,
    MotionTracker,
    PassageDecider,
    calibrate_mm_per_px,
    detect_objects,
    draw_overlay,
    normalize_quad,
)
from side_view import DEFAULT_SIDE_ROI_QUAD, SideView, draw_side_overlay

ROOT = Path(__file__).resolve().parent
CERT_FILE = ROOT / "certs" / "cert.pem"
KEY_FILE = ROOT / "certs" / "key.pem"
CALIB_FILE = ROOT / "calibration.json"


def load_calibration(default_mm: float = 0.5) -> dict:
    """mm/px и ROI для вида сверху и для боковой камеры."""
    out: dict = {"mm_per_px": default_mm, "roi_quad": None}
    try:
        data = json.loads(CALIB_FILE.read_text(encoding="utf-8"))
    except Exception:
        return out
    out["mm_per_px"] = max(0.01, float(data.get("mm_per_px", default_mm)))
    roi = data.get("roi_quad")
    if roi and len(roi) == 4:
        out["roi_quad"] = [list(p) for p in normalize_quad(roi)]
    side_mm = data.get("side_mm_per_px")
    if side_mm:
        out["side_mm_per_px"] = max(0.01, float(side_mm))
    side_roi = data.get("side_roi_quad")
    if side_roi and len(side_roi) == 4:
        out["side_roi_quad"] = [list(p) for p in normalize_quad(side_roi)]
    if data.get("side_rot") is not None:
        out["side_rot"] = int(data["side_rot"]) % 360
    return out


def save_calibration(
    mm_per_px: float,
    roi_quad=None,
    side_mm_per_px: float | None = None,
    side_roi_quad=None,
    side_rot: int | None = None,
) -> None:
    try:
        payload = json.loads(CALIB_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    payload["mm_per_px"] = round(float(mm_per_px), 5)
    if roi_quad is not None:
        payload["roi_quad"] = [list(p) for p in normalize_quad(roi_quad)]
    if side_mm_per_px is not None:
        payload["side_mm_per_px"] = round(float(side_mm_per_px), 5)
    if side_roi_quad is not None:
        payload["side_roi_quad"] = [list(p) for p in normalize_quad(side_roi_quad)]
    if side_rot is not None:
        payload["side_rot"] = int(side_rot) % 360
    CALIB_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

app = Flask(__name__)


@dataclass
class Moment:
    id: int
    ts: float
    time_str: str
    zone: str
    category: str
    size_mm: list[float]
    circle_ratio: float
    thumb_jpeg_b64: str
    k_xy: float = 0.0
    k_side: float | None = None
    height_mm: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts,
            "time": self.time_str,
            "zone": self.zone,
            "category": self.category,
            "size_mm": self.size_mm,
            "circle_ratio": self.circle_ratio,
            "k_xy": self.k_xy,
            "k_side": self.k_side,
            "height_mm": self.height_mm,
            "thumb": self.thumb_jpeg_b64,
        }


class PhoneSource:
    """Кадры с iPhone Safari (POST /api/phone_frame)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.ts = 0.0

    def push_jpeg(self, data: bytes) -> bool:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self.lock:
            self.frame = frame
            self.ts = time.time()
        return True

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            if time.time() - self.ts > 2.5:
                return False, None
            return True, self.frame.copy()

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        with self.lock:
            self.frame = None


phone_source = PhoneSource()
side_phone_source = PhoneSource()


def lan_ip() -> str:
    """Предпочитаем обычный LAN (192.168/10), не VPN вроде 198.18.x."""
    candidates: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                candidates.append(ip)
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        candidates.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass

    def score(ip: str) -> tuple[int, str]:
        if ip.startswith("192.168."):
            return (0, ip)
        if ip.startswith("10."):
            return (1, ip)
        if ip.startswith("172.") and 16 <= int(ip.split(".")[1]) <= 31:
            return (2, ip)
        if ip.startswith("198.18.") or ip.startswith("100."):
            return (9, ip)  # VPN / CGNAT — в конец
        return (5, ip)

    if not candidates:
        return "127.0.0.1"
    return sorted(set(candidates), key=score)[0]


def is_black(frame: np.ndarray, thr: float = 2.0) -> bool:
    return float(frame.mean()) < thr and int(frame.max()) < 8


def placeholder(_message: str = "", _sub: str = "") -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@dataclass
class Pipeline:
    source: int | str = "phone"
    camera_label: str = "iPhone Safari"
    mm_per_px: float = 0.5
    min_area: float = 1000.0
    phone_url: str = ""
    roi_quad: list = field(default_factory=lambda: [list(p) for p in DEFAULT_ROI_QUAD])
    frame_size: list[int] = field(default_factory=lambda: [1280, 720])
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame_jpeg: bytes | None = None
    raw_jpeg: bytes | None = None
    live_det: dict | None = None
    live_drop: str | None = None
    latest: dict | None = None
    moments: deque = field(default_factory=lambda: deque(maxlen=40))
    fps: float = 0.0
    running: bool = False
    error: str | None = None
    black_frames: bool = False
    _cap: object | None = None
    _thread: threading.Thread | None = None
    _moment_i: int = 0
    _black_streak: int = 0
    tracker: MotionTracker = field(default_factory=lambda: MotionTracker(use_sam=True))
    passage: PassageDecider = field(default_factory=PassageDecider)
    # боковая камера: третье измерение и K по вертикальному сечению
    side_source: int | str | None = None
    side_url: str = ""
    side_mm_per_px: float = 0.5
    side_min_area: float = 600.0
    side_rot: int = 0  # поворот кадра боковой камеры: 0/90/180/270
    side_roi_quad: list = field(
        default_factory=lambda: [list(p) for p in DEFAULT_SIDE_ROI_QUAD]
    )
    side_frame_size: list[int] = field(default_factory=lambda: [1280, 720])
    side_view: SideView = field(default_factory=SideView)
    side_jpeg: bytes | None = None
    side_raw_jpeg: bytes | None = None
    side_latest: dict | None = None
    side_ts: float = 0.0
    side_error: str | None = None
    side_fps: float = 0.0
    _side_cap: object | None = None
    _side_thread: threading.Thread | None = None
    # фонарики телефонов: чего хотим с Mac и что телефон реально смог
    torch_want: dict = field(default_factory=lambda: {"top": False, "side": False})
    torch_seq: dict = field(default_factory=lambda: {"top": 0, "side": 0})
    torch_state: dict = field(
        default_factory=lambda: {
            "top": {"on": False, "supported": None, "ts": 0.0},
            "side": {"on": False, "supported": None, "ts": 0.0},
        }
    )

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.side_source is not None:
            self._side_thread = threading.Thread(target=self._side_loop, daemon=True)
            self._side_thread.start()

    def status(self) -> dict:
        with self.lock:
            moments = [m.to_dict() for m in list(self.moments)]
            return {
                "camera": self.camera_label,
                "source": str(self.source),
                "fps": round(self.fps, 1),
                "mm_per_px": round(self.mm_per_px, 4),
                "error": self.error,
                "black_frames": self.black_frames,
                "phone_url": self.phone_url,
                "roi_quad": self.roi_quad,
                "frame_size": self.frame_size,
                "latest": self.latest,
                "live_det": self.live_det,
                "live_drop": self.live_drop,
                "moments": moments,
                "running": self.running and self.error is None,
                "torch": {
                    cam: {
                        "want": bool(self.torch_want[cam]),
                        "on": bool(self.torch_state[cam]["on"]),
                        "supported": self.torch_state[cam]["supported"],
                        "fresh": time.time() - self.torch_state[cam]["ts"] < 4.0,
                    }
                    for cam in ("top", "side")
                },
                "side": {
                    "enabled": self.side_source is not None,
                    "source": None if self.side_source is None else str(self.side_source),
                    "url": self.side_url,
                    "mm_per_px": round(self.side_mm_per_px, 4),
                    "rot": self.side_rot,
                    "roi_quad": self.side_roi_quad,
                    "frame_size": self.side_frame_size,
                    "fps": round(self.side_fps, 1),
                    "fresh": bool(self.side_latest and time.time() - self.side_ts < 1.5),
                    "measure": self.side_latest,
                    "error": self.side_error,
                },
            }

    def _open(self) -> None:
        if self.source == "phone":
            self._cap = phone_source
            self.camera_label = "iPhone Safari"
            self.error = None
            print(f"[cam] режим phone → {self.phone_url}", flush=True)
            return

        if self.source == "auto":
            src, label = find_iphone_source()
            self.source = src
            self.camera_label = label
        elif isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
            self.camera_label = self.camera_label or f"Camera {self.source}"
        elif isinstance(self.source, int):
            self.camera_label = self.camera_label or f"Camera {self.source}"
        else:
            self.camera_label = str(self.source)

        self._cap = open_capture(self.source)
        self.error = None
        print(f"[cam] подключено: {self.camera_label} ({self.source})", flush=True)

    def enable_side(self, source: int | str) -> None:
        self.side_source = source
        self.side_view.reset()
        if self._side_thread is None or not self._side_thread.is_alive():
            self._side_thread = threading.Thread(target=self._side_loop, daemon=True)
            self._side_thread.start()

    def _open_side(self) -> None:
        if self.side_source == "phone":
            self._side_cap = side_phone_source
            print(f"[side] режим phone → {self.side_url}", flush=True)
        else:
            src = self.side_source
            if isinstance(src, str) and src.isdigit():
                src = int(src)
            self._side_cap = open_capture(src)
            print(f"[side] подключено: {src}", flush=True)
        self.side_error = None

    def _rotate_side(self, frame: np.ndarray) -> np.ndarray:
        """Телефон сбоку часто стоит боком — приводим кадр к «лента внизу»."""
        code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(int(self.side_rot) % 360)
        return frame if code is None else cv2.rotate(frame, code)

    def _side_loop(self) -> None:
        t_prev = time.time()
        while self.running and self.side_source is not None:
            try:
                if self._side_cap is None:
                    self._open_side()
                assert self._side_cap is not None
                ok, frame = self._side_cap.read()
                if ok and frame is not None:
                    frame = self._rotate_side(frame)
                if not ok or frame is None:
                    with self.lock:
                        self.side_latest = None
                    time.sleep(0.12)
                    continue
                now = time.time()
                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    self.side_fps = (
                        0.9 * self.side_fps + 0.1 / dt if self.side_fps else 1.0 / dt
                    )
                with self.lock:
                    quad = [list(p) for p in self.side_roi_quad]
                    self.side_frame_size = [int(frame.shape[1]), int(frame.shape[0])]
                measure = self.side_view.measure(
                    frame,
                    self.side_mm_per_px,
                    self.side_min_area,
                    roi_quad=quad,
                )
                overlay = draw_side_overlay(frame, measure, quad, self.side_mm_per_px)
                ok_j, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                ok_r, buf_r = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                with self.lock:
                    if ok_j:
                        self.side_jpeg = buf.tobytes()
                    if ok_r:
                        self.side_raw_jpeg = buf_r.tobytes()
                    if measure is not None:
                        self.side_latest = measure.to_dict()
                        self.side_ts = now
                    else:
                        self.side_latest = None
                    self.side_error = None
            except Exception as exc:  # noqa: BLE001
                self.side_error = str(exc)
                print(f"[side] ошибка: {exc}", flush=True)
                if self._side_cap is not None and self.side_source != "phone":
                    try:
                        self._side_cap.release()
                    except Exception:
                        pass
                    self._side_cap = None
                time.sleep(1.0)

    def _feed_side_sample(self) -> None:
        """Свежая проба сбоку — в текущий проход."""
        with self.lock:
            measure = self.side_latest
            fresh = measure is not None and time.time() - self.side_ts < 1.0
        if fresh and measure is not None:
            self.passage.feed_side(
                float(measure["k_side"]), float(measure["height_mm"])
            )

    def _record_moment(self, frame, det) -> None:
        """Один момент на проход — по лучшему кадру."""
        x, y, w, h = det.bbox
        pad = 24
        H, W = frame.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        thumb = cv2.resize(crop, (160, 120), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if not ok:
            return
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        self._moment_i += 1
        info = det.to_dict()
        now = time.time()
        moment = Moment(
            id=self._moment_i,
            ts=now,
            time_str=datetime.now().strftime("%H:%M:%S"),
            zone=info["zone"],
            category=info["category"],
            size_mm=info["size_mm"],
            circle_ratio=info["circle_ratio"],
            thumb_jpeg_b64=b64,
            k_xy=info.get("k_xy", 0.0),
            k_side=info.get("k_side"),
            height_mm=info.get("height_mm"),
        )
        with self.lock:
            self.moments.appendleft(moment)
        print(
            f"[decision] #{moment.id} [{moment.zone}] {moment.category} "
            f"{moment.size_mm[0]}×{moment.size_mm[1]} mm "
            f"K_xy={moment.k_xy}"
            + (f" K_side={moment.k_side}" if moment.k_side is not None else "")
            + (f" H={moment.height_mm}mm" if moment.height_mm is not None else ""),
            flush=True,
        )

    def _loop(self) -> None:
        t_prev = time.time()
        while self.running:
            try:
                if self._cap is None:
                    self._open()
                assert self._cap is not None
                ok, frame = self._cap.read()

                if not ok or frame is None:
                    frame = placeholder(
                        "Ждём камеру iPhone",
                        f"На iPhone открой Safari: {self.phone_url}",
                    )
                    self.black_frames = False
                    with self.lock:
                        self.latest = None
                        self.frame_size = [frame.shape[1], frame.shape[0]]
                    ok_j, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok_j:
                        with self.lock:
                            self.frame_jpeg = buf.tobytes()
                    time.sleep(0.08)
                    continue

                with self.lock:
                    self.frame_size = [int(frame.shape[1]), int(frame.shape[0])]

                if is_black(frame):
                    self._black_streak += 1
                else:
                    self._black_streak = 0
                self.black_frames = self._black_streak >= 8

                if self.black_frames:
                    frame = placeholder(
                        "Continuity: чёрный кадр",
                        f"Открой на iPhone: {self.phone_url}  (надёжный режим)",
                    )
                    dets = []
                    decided = self.passage.feed(frame, None)
                    if decided is not None:
                        thumb_src = self.passage.final_frame()
                        self._record_moment(thumb_src if thumb_src is not None else frame, decided)
                else:
                    now = time.time()
                    dt = now - t_prev
                    t_prev = now
                    if dt > 0:
                        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
                    with self.lock:
                        quad = [list(p) for p in self.roi_quad]
                    dets = detect_objects(
                        frame,
                        self.mm_per_px,
                        self.min_area,
                        roi_quad=quad,
                        tracker=self.tracker,
                    )
                    raw = frame
                    ok_r, buf_r = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    with self.lock:
                        if ok_r:
                            self.raw_jpeg = buf_r.tobytes()
                        self.live_det = dets[0].to_dict() if dets else None
                        self.live_drop = (
                            None
                            if dets
                            else f"{self.tracker.drop} | pick: {self.tracker.drop_pick}"
                        )
                    roi_xy = None
                    roi_hw = self.tracker._roi_hw
                    if self.tracker._cx is not None and self.tracker._cy is not None:
                        roi_xy = (float(self.tracker._cx), float(self.tracker._cy))
                    decided = self.passage.feed(
                        raw,
                        dets[0] if dets else None,
                        track_miss=int(self.tracker._miss),
                        sam_score=float(self.tracker._sam_score),
                        roi_xy=roi_xy,
                        roi_hw=roi_hw,
                    )
                    if self.side_source is not None and dets:
                        self._feed_side_sample()
                    if decided is not None:
                        thumb_src = self.passage.final_frame()
                        self._record_moment(thumb_src if thumb_src is not None else raw, decided)
                    # ROI рисует интерактивный SVG; на стриме — тонкая линия для сверки
                    frame = draw_overlay(
                        frame,
                        dets,
                        self.fps,
                        self.mm_per_px,
                        self.camera_label,
                        roi_quad=quad,
                        draw_roi=True,
                    )

                latest = self.passage.ui_dict()

                ok_j, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_j:
                    with self.lock:
                        self.frame_jpeg = buf.tobytes()
                        self.latest = latest
                        self.error = None
            except Exception as exc:  # noqa: BLE001
                import traceback

                self.error = str(exc)
                print(f"[cam] ошибка: {exc}", flush=True)
                traceback.print_exc()
                if self._cap is not None and self.source != "phone":
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                time.sleep(1.0)


pipeline = Pipeline()


@app.route("/")
def index():
    return render_template("index.html", phone_url=pipeline.phone_url)


@app.route("/phone")
def phone_page():
    return render_template("phone.html", cam="top")


@app.route("/api/phone_frame", methods=["POST"])
def phone_frame():
    data = request.get_data(cache=False)
    if not data:
        return jsonify({"ok": False, "error": "empty"}), 400
    if not phone_source.push_jpeg(data):
        return jsonify({"ok": False, "error": "bad jpeg"}), 400
    # авто-переключение на phone, если пришли живые кадры
    if pipeline.source != "phone":
        print("[cam] переключение на iPhone Safari stream", flush=True)
        if pipeline._cap is not None and pipeline.source != "phone":
            try:
                pipeline._cap.release()
            except Exception:
                pass
        pipeline.source = "phone"
        pipeline.camera_label = "iPhone Safari"
        pipeline._cap = phone_source
        pipeline.black_frames = False
        pipeline._black_streak = 0
    return jsonify({"ok": True})


@app.route("/side")
def side_page():
    return render_template(
        "phone.html",
        title="Боковая камера (высота)",
        hint="Поставь телефон сбоку конвейера: линия ленты — у нижней границы кадра.",
        post_url="/api/side_frame",
        cam="side",
    )


@app.route("/api/side_frame", methods=["POST"])
def side_frame():
    data = request.get_data(cache=False)
    if not data:
        return jsonify({"ok": False, "error": "empty"}), 400
    if not side_phone_source.push_jpeg(data):
        return jsonify({"ok": False, "error": "bad jpeg"}), 400
    if pipeline.side_source is None:
        print("[side] включаю боковую камеру (phone)", flush=True)
        pipeline.enable_side("phone")
    return jsonify({"ok": True})


def _torch_cam(value: str | None) -> str:
    return "side" if str(value or "top").lower() == "side" else "top"


@app.route("/api/torch", methods=["GET", "POST"])
def api_torch():
    """GET — телефон спрашивает, что включить; POST — Mac задаёт состояние."""
    if request.method == "GET":
        cam = _torch_cam(request.args.get("cam"))
        with pipeline.lock:
            return jsonify(
                {"cam": cam, "on": bool(pipeline.torch_want[cam]), "seq": pipeline.torch_seq[cam]}
            )
    data = request.get_json(force=True, silent=True) or {}
    cam = _torch_cam(data.get("cam"))
    want = bool(data.get("on"))
    with pipeline.lock:
        if pipeline.torch_want[cam] != want:
            pipeline.torch_seq[cam] += 1
        pipeline.torch_want[cam] = want
    print(f"[torch] {cam} → {'вкл' if want else 'выкл'}", flush=True)
    return jsonify(pipeline.status()["torch"])


@app.route("/api/torch_state", methods=["POST"])
def api_torch_state():
    """Телефон сообщает, что фактически получилось."""
    data = request.get_json(force=True, silent=True) or {}
    cam = _torch_cam(data.get("cam"))
    with pipeline.lock:
        pipeline.torch_state[cam] = {
            "on": bool(data.get("on")),
            "supported": data.get("supported"),
            "ts": time.time(),
        }
    return jsonify({"ok": True})


@app.route("/side_video")
def side_video():
    def gen():
        while True:
            with pipeline.lock:
                chunk = pipeline.side_jpeg
            if chunk is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
            time.sleep(0.05)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/side.jpg")
def api_side_raw():
    with pipeline.lock:
        chunk = pipeline.side_raw_jpeg
    if chunk is None:
        return jsonify({"ok": False, "error": "нет кадра"}), 404
    return Response(chunk, mimetype="image/jpeg")


@app.route("/api/side_config", methods=["POST"])
def api_side_config():
    data = request.get_json(force=True, silent=True) or {}
    if "side_mm_per_px" in data:
        pipeline.side_mm_per_px = max(0.01, float(data["side_mm_per_px"]))
    if "side_min_area" in data:
        pipeline.side_min_area = max(50.0, float(data["side_min_area"]))
    if "side_rot" in data:
        pipeline.side_rot = int(data["side_rot"]) % 360
        pipeline.side_view.reset()
    if "side_roi_quad" in data:
        with pipeline.lock:
            pipeline.side_roi_quad = [list(p) for p in normalize_quad(data["side_roi_quad"])]
        pipeline.side_view.reset()
    if data.get("side_source") is not None:
        pipeline._side_cap = None
        pipeline.enable_side(data["side_source"])
    save_calibration(
        pipeline.mm_per_px,
        pipeline.roi_quad,
        side_mm_per_px=pipeline.side_mm_per_px,
        side_roi_quad=pipeline.side_roi_quad,
        side_rot=pipeline.side_rot,
    )
    return jsonify(pipeline.status())


@app.route("/api/side_scale", methods=["POST"])
def api_side_scale():
    """Масштаб сбоку по эталону известной высоты: mm/px = H_эталона / H_пикселей."""
    data = request.get_json(force=True, silent=True) or {}
    ref_mm = float(data.get("height_mm", 200.0))
    if ref_mm < 10.0:
        return jsonify({"ok": False, "error": "эталон слишком мал"}), 400
    with pipeline.lock:
        measure = pipeline.side_latest
        fresh = measure is not None and time.time() - pipeline.side_ts < 2.0
    if not fresh or measure is None:
        return jsonify({"ok": False, "error": "эталон не виден в боковом ROI"}), 400
    px = float(measure["bbox_roi"][3])
    if px < 8.0:
        return jsonify({"ok": False, "error": f"силуэт слишком мал: {px:.0f}px"}), 400
    pipeline.side_mm_per_px = ref_mm / px
    save_calibration(
        pipeline.mm_per_px,
        pipeline.roi_quad,
        side_mm_per_px=pipeline.side_mm_per_px,
        side_roi_quad=pipeline.side_roi_quad,
        side_rot=pipeline.side_rot,
    )
    print(
        f"[side] масштаб по эталону: {ref_mm:.0f}мм / {px:.0f}px "
        f"= {pipeline.side_mm_per_px:.4f} mm/px",
        flush=True,
    )
    return jsonify(
        {
            "ok": True,
            "side_mm_per_px": round(pipeline.side_mm_per_px, 4),
            "height_px": round(px, 1),
            "height_mm": round(ref_mm, 1),
        }
    )


@app.route("/api/side_calibrate", methods=["POST"])
def api_side_calibrate():
    """mm/px боковой камеры по вертикально стоящей линейке."""
    data = request.get_json(force=True, silent=True) or {}
    ruler_mm = float(data.get("ruler_mm", 200.0))
    ok, frame = (False, None)
    if pipeline._side_cap is not None:
        ok, frame = pipeline._side_cap.read()
    if not ok or frame is None:
        return jsonify({"ok": False, "error": "нет кадра боковой камеры"}), 400
    frame = pipeline._rotate_side(frame)
    with pipeline.lock:
        quad = [list(p) for p in pipeline.side_roi_quad]
    result = calibrate_mm_per_px(frame, roi_quad=quad, ruler_mm=ruler_mm)
    if result is None:
        return jsonify({"ok": False, "error": "линейка не найдена в боковом ROI"}), 400
    pipeline.side_mm_per_px = float(result["mm_per_px"])
    pipeline.side_view.reset()
    save_calibration(
        pipeline.mm_per_px,
        pipeline.roi_quad,
        side_mm_per_px=pipeline.side_mm_per_px,
        side_roi_quad=pipeline.side_roi_quad,
        side_rot=pipeline.side_rot,
    )
    return jsonify(
        {
            "ok": True,
            "side_mm_per_px": round(pipeline.side_mm_per_px, 4),
            "ruler_mm": round(float(result["ruler_mm"]), 1),
            "scale_px": round(float(result["scale_px"]), 1),
            "ticks": int(result["ticks"]),
            "method": result["method"],
        }
    )


@app.route("/video")
def video():
    def gen():
        while True:
            with pipeline.lock:
                chunk = pipeline.frame_jpeg
            if chunk is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
            time.sleep(0.033)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(pipeline.status())


@app.route("/api/raw.jpg")
def api_raw():
    with pipeline.lock:
        chunk = pipeline.raw_jpeg
    if chunk is None:
        return jsonify({"ok": False, "error": "нет кадра"}), 404
    return Response(chunk, mimetype="image/jpeg")


@app.route("/api/overlay.jpg")
def api_overlay():
    with pipeline.lock:
        chunk = pipeline.frame_jpeg
    if chunk is None:
        return jsonify({"ok": False, "error": "нет кадра"}), 404
    return Response(chunk, mimetype="image/jpeg")


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json(force=True, silent=True) or {}
    if "mm_per_px" in data:
        pipeline.mm_per_px = max(0.01, float(data["mm_per_px"]))
        save_calibration(pipeline.mm_per_px, pipeline.roi_quad)
    if "min_area" in data:
        pipeline.min_area = max(100.0, float(data["min_area"]))
    if "roi_quad" in data:
        with pipeline.lock:
            pipeline.roi_quad = [list(p) for p in normalize_quad(data["roi_quad"])]
            pipeline.tracker.reset()
            save_calibration(pipeline.mm_per_px, pipeline.roi_quad)
    if "source" in data:
        src = data["source"]
        if pipeline._cap is not None and pipeline.source != "phone":
            try:
                pipeline._cap.release()
            except Exception:
                pass
        pipeline._cap = None
        pipeline.source = int(src) if isinstance(src, str) and src.isdigit() else src
        pipeline.black_frames = False
        pipeline._black_streak = 0
    return jsonify(pipeline.status())


@app.route("/api/moments/clear", methods=["POST"])
def clear_moments():
    with pipeline.lock:
        pipeline.moments.clear()
    return jsonify({"ok": True})


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    """Калибровка мм/px по шкале зелёной линейки (0 → последнее деление)."""
    data = request.get_json(force=True, silent=True) or {}
    ruler_mm = float(data.get("ruler_mm", 200.0))
    if ruler_mm < 10.0:
        return jsonify({"ok": False, "error": "длина слишком мала"}), 400
    ok, frame = phone_source.read() if pipeline.source == "phone" else (False, None)
    if not ok or frame is None:
        if pipeline._cap is not None:
            ok, frame = pipeline._cap.read()
    if not ok or frame is None:
        return jsonify({"ok": False, "error": "нет кадра"}), 400
    with pipeline.lock:
        quad = [list(p) for p in pipeline.roi_quad]
    result = calibrate_mm_per_px(frame, roi_quad=quad, ruler_mm=ruler_mm)
    if result is None:
        return jsonify({"ok": False, "error": "линейка не найдена в ROI"}), 400
    pipeline.mm_per_px = float(result["mm_per_px"])
    save_calibration(pipeline.mm_per_px, pipeline.roi_quad)
    pipeline.tracker.reset()
    return jsonify(
        {
            "ok": True,
            "mm_per_px": round(pipeline.mm_per_px, 4),
            "ruler_mm": round(float(result["ruler_mm"]), 1),
            "scale_px": round(float(result["scale_px"]), 1),
            "body_px": round(float(result["body_px"]), 1),
            "ticks": int(result["ticks"]),
            "method": result["method"],
        }
    )


def make_ssl_context() -> ssl.SSLContext | None:
    if not (CERT_FILE.is_file() and KEY_FILE.is_file()):
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(CERT_FILE), str(KEY_FILE))
    return ctx


def start_http_redirect(host: str, http_port: int, https_port: int, ip: str) -> None:
    """HTTP → подсказка/редирект на HTTPS (иначе кажется, что сервер «не запущен»)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    https_base = f"https://{ip}:{https_port}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            target = https_base + (self.path or "/")
            body = (
                "<!doctype html><meta charset=utf-8>"
                "<title>OZON CV</title>"
                "<body style='font-family:system-ui;padding:24px'>"
                "<h1>Нужен HTTPS</h1>"
                f"<p>Открой: <a href='{target}'>{target}</a></p>"
                "<p>На iPhone: предупреждение сертификата → Показать подробности → Посетить сайт.</p>"
                "</body>"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        srv = ThreadingHTTPServer((host, http_port), Handler)
    except OSError as exc:
        print(f"[http] порт {http_port} занят, редирект не поднят: {exc}", flush=True)
        return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"HTTP-подсказка: http://127.0.0.1:{http_port}  →  {https_base}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        default="phone",
        help="phone (рекомендуется) | auto (Continuity) | индекс | URL",
    )
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 чтобы iPhone достучался по Wi‑Fi")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--mm-per-px", type=float, default=None, help="масштаб; иначе calibration.json / 0.5")
    p.add_argument(
        "--side",
        default=None,
        help="боковая камера для высоты: phone (второй телефон, /side) | индекс | URL",
    )
    p.add_argument("--http", action="store_true", help="без HTTPS (камера на iPhone не заработает)")
    args = p.parse_args()

    ssl_ctx = None if args.http else make_ssl_context()
    scheme = "https" if ssl_ctx else "http"
    ip = lan_ip()
    pipeline.phone_url = f"{scheme}://{ip}:{args.port}/phone"
    pipeline.side_url = f"{scheme}://{ip}:{args.port}/side"
    calib = load_calibration(0.5)
    if args.mm_per_px is not None:
        pipeline.mm_per_px = float(args.mm_per_px)
        saved_roi = calib["roi_quad"]
    else:
        pipeline.mm_per_px = calib["mm_per_px"]
        saved_roi = calib["roi_quad"]
    if "side_mm_per_px" in calib:
        pipeline.side_mm_per_px = calib["side_mm_per_px"]
    if "side_roi_quad" in calib:
        pipeline.side_roi_quad = calib["side_roi_quad"]
    if "side_rot" in calib:
        pipeline.side_rot = calib["side_rot"]
    if saved_roi is not None:
        pipeline.roi_quad = saved_roi
    else:
        pipeline.roi_quad = [list(p) for p in DEFAULT_ROI_QUAD]
        save_calibration(pipeline.mm_per_px, pipeline.roi_quad)
    if args.side is not None:
        pipeline.side_source = args.side if not args.side.isdigit() else int(args.side)

    if args.source == "phone":
        pipeline.source = "phone"
        pipeline.camera_label = "iPhone Safari"
    elif args.source == "auto":
        pipeline.source = "auto"
    elif args.source.isdigit():
        pipeline.source = int(args.source)
        pipeline.camera_label = f"Camera {args.source}"
    else:
        pipeline.source = args.source
        pipeline.camera_label = args.source

    pipeline.start()
    print(f"UI (Mac):    {scheme}://127.0.0.1:{args.port}", flush=True)
    print(f"Камера iPhone (Safari): {pipeline.phone_url}", flush=True)
    print(f"Боковая камера (высота): {pipeline.side_url}", flush=True)
    if ssl_ctx:
        print(
            "iOS: при предупреждении сертификата → «Показать подробности» → «Посетить сайт». "
            "Иначе getUserMedia не покажет запрос камеры.",
            flush=True,
        )
        start_http_redirect(args.host, args.port + 1, args.port, ip)
    else:
        print("WARN: нет certs/*.pem — iPhone Safari не даст камеру по обычному HTTP", flush=True)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        use_reloader=False,
        ssl_context=ssl_ctx,
    )


if __name__ == "__main__":
    main()
