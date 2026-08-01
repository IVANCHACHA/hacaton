"""Автопоиск и открытие iPhone Continuity Camera / IP-стрима."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass

import cv2


@dataclass
class CameraInfo:
    index: int
    name: str
    kind: str  # iphone | desk | facetime | other


def list_avfoundation_devices() -> list[CameraInfo]:
    """Имена устройств через ffmpeg avfoundation (на macOS)."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=8,
        )
        text = (proc.stderr or "") + (proc.stdout or "")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    devices: list[CameraInfo] = []
    in_video = False
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if not in_video:
            continue
        m = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2).strip()
        lower = name.lower()
        if "iphone" in lower and "desk" in lower:
            kind = "desk"
        elif "iphone" in lower:
            kind = "iphone"
        elif "facetime" in lower:
            kind = "facetime"
        else:
            kind = "other"
        devices.append(CameraInfo(index=idx, name=name, kind=kind))
    return devices


def probe_index(index: int, tries: int = 25) -> bool:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok = False
    for _ in range(tries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            ok = True
            break
        time.sleep(0.08)
    cap.release()
    return ok


def prefer_iphone(devices: list[CameraInfo]) -> CameraInfo | None:
    for kind in ("iphone", "desk"):
        for d in devices:
            if d.kind == kind:
                return d
    return None


def find_iphone_source() -> tuple[int | str, str]:
    """
    Возвращает (source, label).
    Приоритет: iPhone Camera → Desk View → первый рабочий индекс ≠ FaceTime.
    """
    devices = list_avfoundation_devices()
    preferred = prefer_iphone(devices)

    # Сначала пробуем известный iPhone-индекс (без лишних probe всех камер)
    candidates: list[CameraInfo] = []
    if preferred:
        candidates.append(preferred)
    for d in devices:
        if d.kind in ("iphone", "desk") and (not preferred or d.index != preferred.index):
            candidates.append(d)

    for d in candidates:
        print(f"[cam] пробую {d.name} [{d.index}]…", flush=True)
        if probe_index(d.index, tries=20):
            return d.index, d.name

    # Без ffmpeg или iPhone недоступен — любой индекс, кроме FaceTime
    for d in devices:
        if d.kind == "facetime":
            continue
        if any(c.index == d.index for c in candidates):
            continue
        if probe_index(d.index, tries=12):
            return d.index, d.name

    for idx in (1, 0, 2, 3):
        if probe_index(idx, tries=12):
            label = next((x.name for x in devices if x.index == idx), f"Camera {idx}")
            return idx, label

    raise RuntimeError(
        "iPhone-камера не найдена. Разблокируй iPhone рядом с Mac "
        "(Continuity Camera) или укажи --source URL."
    )


def open_capture(source: str | int, warmup: int = 40) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру/стрим: {source}")

    # Continuity стабильнее на 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    got = False
    for _ in range(warmup):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            got = True
            break
        time.sleep(0.1)
    if not got:
        # Не валим сразу — Continuity часто «просыпается» через 1–2 с
        print(f"[cam] warmup без кадра ({source}), оставляю открытой…", flush=True)
    return cap
