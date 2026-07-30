# Python 3.11 зафиксирован сознательно: под 3.12 у pybullet на момент написания нет
# готового wheel на части платформ — тянет сборку из исходников (~80 МБ, долго).
FROM python:3.11-slim

# build-essential/cmake — подстраховка на случай отсутствия готового wheel pybullet
# под конкретную платформу сборки; libspatialindex-dev — на случай, если wheel
# rtree для целевой платформы не несёт с собой бинарник libspatialindex.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ER_TINY_RENDERER (используется во всех sim/cameras.py и tools/make_video.py
# захватах кадров) — программный рендер, GPU/X11 не нужен, headless-запуск не требует
# отдельного виртуального дисплея (см. известные ограничения в README про GUI-режим).
ENTRYPOINT ["python", "run.py"]
CMD ["--backend", "sim", "--scenario", "typical", "--headless"]
