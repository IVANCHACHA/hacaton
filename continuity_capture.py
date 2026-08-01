"""Захват Continuity Camera через AVFoundation (не OpenCV).

OpenCV/ffmpeg часто получают чёрные кадры с iPhone Continuity.
Нужны AVCaptureDeviceTypeContinuityCamera + NSCameraUseContinuityCameraDeviceType.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Optional

import cv2
import numpy as np
import objc
from AVFoundation import (
    AVCaptureDevice,
    AVCaptureDeviceDiscoverySession,
    AVCaptureDeviceInput,
    AVCaptureDeviceTypeContinuityCamera,
    AVCaptureDeviceTypeDeskViewCamera,
    AVCaptureDeviceTypeExternal,
    AVCaptureSession,
    AVCaptureVideoDataOutput,
    AVMediaTypeVideo,
)
from Cocoa import NSObject
from CoreMedia import CMSampleBufferGetImageBuffer
from Foundation import NSRunLoop, NSDate, NSDefaultRunLoopMode
from Quartz import (
    CGBitmapContextCreate,
    CGBitmapContextCreateImage,
    CGColorSpaceCreateDeviceRGB,
    CGDataProviderCopyData,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CIContext,
    CIImage,
    CVPixelBufferGetHeight,
    CVPixelBufferGetWidth,
    kCGBitmapByteOrder32Little,
    kCGImageAlphaPremultipliedFirst,
)

BGRA = 1111970369  # kCVPixelFormatType_32BGRA


def _dispatch_queue(name: bytes):
    lib = ctypes.cdll.LoadLibrary("/usr/lib/system/libdispatch.dylib")
    lib.dispatch_queue_create.restype = ctypes.c_void_p
    lib.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    return objc.objc_object(c_void_p=lib.dispatch_queue_create(name, None))


def _pixel_buffer_to_bgr(pb) -> Optional[np.ndarray]:
    """CVPixelBuffer → BGR uint8 через CIImage (стабильно для Continuity)."""
    try:
        w = int(CVPixelBufferGetWidth(pb))
        h = int(CVPixelBufferGetHeight(pb))
        if w < 2 or h < 2:
            return None
        ci = CIImage.imageWithCVImageBuffer_(pb)
        if ci is None:
            ci = CIImage.imageWithCVPixelBuffer_(pb)
        if ci is None:
            return None
        ctx = CIContext.contextWithOptions_(None)
        cg = ctx.createCGImage_fromRect_(ci, ci.extent())
        if cg is None:
            return None
        # Draw into BGRA buffer
        bpr = w * 4
        buf = (ctypes.c_uint8 * (bpr * h))()
        color = CGColorSpaceCreateDeviceRGB()
        bitmap = CGBitmapContextCreate(
            buf,
            w,
            h,
            8,
            bpr,
            color,
            kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst,
        )
        if bitmap is None:
            # fallback: copy provider bytes
            provider = CGImageGetDataProvider(cg)
            data = CGDataProviderCopyData(provider)
            raw = bytes(data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            # assume BGRA
            if arr.size >= w * h * 4:
                bgra = arr[: w * h * 4].reshape((h, w, 4))
                return np.ascontiguousarray(bgra[:, :, :3])
            return None

        from Quartz import CGContextDrawImage, CGRectMake

        CGContextDrawImage(bitmap, CGRectMake(0, 0, w, h), cg)
        bgra = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4)).copy()
        return np.ascontiguousarray(bgra[:, :, :3])
    except Exception as exc:  # noqa: BLE001
        print(f"[continuity] convert error: {exc}", flush=True)
        return None


class _Sink(NSObject):
    def init(self):
        self = objc.super(_Sink, self).init()
        if self is None:
            return None
        self.lock = threading.Lock()
        self.latest: Optional[np.ndarray] = None
        self.count = 0
        self.nonzero = 0
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(
        self, _output, sample_buffer, _connection
    ):
        pb = CMSampleBufferGetImageBuffer(sample_buffer)
        if pb is None:
            return
        frame = _pixel_buffer_to_bgr(pb)
        if frame is None:
            return
        with self.lock:
            self.latest = frame
            self.count += 1
            if float(frame.mean()) > 1.5:
                self.nonzero += 1


def discover_continuity_device():
    types = [
        AVCaptureDeviceTypeContinuityCamera,
        AVCaptureDeviceTypeDeskViewCamera,
        AVCaptureDeviceTypeExternal,
    ]
    session = AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        types, AVMediaTypeVideo, 0  # AVCaptureDevicePositionUnspecified
    )
    devices = list(session.devices() or [])
    for d in devices:
        name = str(d.localizedName())
        print(f"[continuity] device: {name} type={d.deviceType()}", flush=True)
    for prefer in ("iPhone Camera", "iPhone", "Desk"):
        for d in devices:
            if prefer.lower() in str(d.localizedName()).lower():
                return d
    return devices[0] if devices else None


class ContinuityCapture:
    """Потоковый захват iPhone Continuity → read() как у cv2.VideoCapture."""

    def __init__(self):
        self._session: Optional[AVCaptureSession] = None
        self._sink: Optional[_Sink] = None
        self._runloop_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.label = "iPhone Continuity"

    def open(self) -> bool:
        device = discover_continuity_device()
        if device is None:
            # fallback: any video device named iPhone
            all_dev = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo) or []
            for d in all_dev:
                if "iphone" in str(d.localizedName()).lower():
                    device = d
                    break
        if device is None:
            return False

        self.label = str(device.localizedName())
        session = AVCaptureSession.alloc().init()
        session.beginConfiguration()
        inp, err = AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if inp is None:
            print(f"[continuity] input error: {err}", flush=True)
            return False
        if session.canAddInput_(inp):
            session.addInput_(inp)
        out = AVCaptureVideoDataOutput.alloc().init()
        out.setAlwaysDiscardsLateVideoFrames_(True)
        out.setVideoSettings_({"PixelFormatType": BGRA})
        sink = _Sink.alloc().init()
        out.setSampleBufferDelegate_queue_(sink, _dispatch_queue(b"ozon.continuity"))
        if session.canAddOutput_(out):
            session.addOutput_(out)
        session.commitConfiguration()
        session.startRunning()
        self._session = session
        self._sink = sink
        self._stop.clear()
        self._runloop_thread = threading.Thread(target=self._pump, daemon=True)
        self._runloop_thread.start()

        # ждём первый ненулевой кадр до 6 с
        t0 = time.time()
        while time.time() - t0 < 6.0:
            ok, frame = self.read()
            if ok and frame is not None and float(frame.mean()) > 1.5:
                print(
                    f"[continuity] OK {self.label} {frame.shape} mean={frame.mean():.1f}",
                    flush=True,
                )
                return True
            time.sleep(0.05)
        print(
            f"[continuity] кадры есть, но чёрные (count={sink.count}). "
            "Проверь Continuity / освещение / разблокировку iPhone.",
            flush=True,
        )
        return sink.count > 0

    def _pump(self) -> None:
        while not self._stop.is_set():
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05)
            )

    def read(self):
        if self._sink is None:
            return False, None
        with self._sink.lock:
            frame = None if self._sink.latest is None else self._sink.latest.copy()
        return frame is not None, frame

    def isOpened(self) -> bool:
        return self._session is not None and bool(self._session.isRunning())

    def release(self) -> None:
        self._stop.set()
        if self._session is not None:
            self._session.stopRunning()
            self._session = None
        self._sink = None


def open_continuity_or_none() -> Optional[ContinuityCapture]:
    cap = ContinuityCapture()
    if cap.open():
        return cap
    cap.release()
    return None
