"""Offline выпуклая декомпозиция (VHACD) реальных STL организаторов.

Динамическое тело в Bullet из одного GEOM_MESH схлопывает в выпуклую оболочку —
для вогнутых товаров (шлем, мешок, ручка со сложной геометрией) это даёт грубо
неверную форму столкновений, а на больших/сложных мешах (Мешок.stl, 5.4 МБ) сама
попытка на лету построить выпуклую оболочку у Bullet может уронить процесс
(наблюдалось на реальном STL организаторов). VHACD режет меш на несколько выпуклых
кусков offline, результат кэшируется по хешу входного файла — гоняем его один раз,
а не при каждом запуске симуляции.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pybullet as p
import trimesh

CACHE_DIR = Path(__file__).resolve().parent.parent / "assets" / "meshes" / "vhacd_cache"


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_convex_decomposition(mesh_path: str) -> str:
    """Возвращает путь к выпукло-декомпозированному .obj для mesh_path, считая offline
    один раз и кэшируя по хешу содержимого файла (не по имени — переименование не
    должно требовать пересчёта, а подмена файла под тем же именем не должна
    использовать устаревший кэш)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _file_hash(mesh_path)
    out_path = CACHE_DIR / f"{key}.obj"
    log_path = CACHE_DIR / f"{key}.log"
    if out_path.exists():
        return str(out_path)

    # Для уже выпуклых тел (короб, цилиндр) VHACD не нужен и не бесплатен: он
    # аппроксимирует форму вокселизацией и на грубом разрешении слегка "съедает"
    # точные размеры/грани — а Bullet и без того даёт ТОЧНУЮ выпуклую оболочку для
    # одного GEOM_MESH. Сравнивать объём меша с объёмом хула для этого не годится —
    # часть STL организаторов представляют собой полые оболочки (стенка бутылки,
    # короба), у них объём "тела" маленький даже при выпуклом наружном силуэте.
    # Вместо этого меряем максимальную глубину вогнутости наружного контура
    # относительно габарита — вот это уже не зависит от того, полый меш или нет.
    mesh = trimesh.load(mesh_path, force="mesh")
    hull = mesh.convex_hull
    _, concavity_mm, _ = hull.nearest.on_surface(mesh.vertices)
    relative_concavity = concavity_mm.max() / mesh.extents.max()
    if relative_concavity < 0.05:
        return mesh_path

    # встроенный p.vhacd понимает только Wavefront .obj — STL организаторов надо
    # сконвертировать (без потери геометрии, просто смена контейнера)
    vhacd_input = mesh_path
    if Path(mesh_path).suffix.lower() != ".obj":
        obj_path = CACHE_DIR / f"{key}_src.obj"
        trimesh.load(mesh_path, force="mesh").export(obj_path)
        vhacd_input = str(obj_path)

    p.vhacd(vhacd_input, str(out_path), str(log_path),
            resolution=200000, depth=20, concavity=0.0015,
            planeDownsampling=4, convexhullDownsampling=4,
            alpha=0.05, beta=0.05, gamma=0.00125, pca=0,
            mode=0, maxNumVerticesPerCH=64, minVolumePerCH=0.0001)
    return str(out_path)
