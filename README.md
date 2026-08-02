# Ozon CV Sorter

Классификация товаров для **физпрототипа**: камера (часто iPhone) + веб со стримом и оверлеем. Те же зоны, что в треке 3:

| Зона | Категория |
|------|-----------|
| **B** | Подходит для сортировки |
| **C** | Не подходит по габаритам |
| **D** | Не подходит без доупаковки (круг в сечении) |

Габариты: от \(10\times10\times10\) мм до \(450\times320\times320\) мм. Круг: \(K = r_\text{вп}/R_\text{оп} \ge 0.8\). Если конфликт — C важнее D.

Симуляция линии и шиберов — в ветке [`sim`](https://github.com/IVANCHACHA/hacaton/tree/sim).

## Запуск UI

```bash
git checkout cv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

1. На Mac: [https://127.0.0.1:5050](https://127.0.0.1:5050) (самоподписанный сертификат — «продолжить»)
2. На iPhone (та же Wi‑Fi): `https://<IP-мака>:5050/phone`
   - Safari: **Показать подробности → Посетить сайт**
   - затем **Включить камеру**

Без HTTPS iOS Safari не вызывает диалог камеры (`getUserMedia` только в secure context).

По умолчанию камера — **Safari на iPhone**. Continuity Camera на многих Mac отдаёт чёрный кадр — phone-режим обычно надёжнее.

```bash
python app.py --source phone     # iPhone Safari (по умолчанию)
python app.py --source auto      # Continuity (часто чёрный)
python app.py --source 0         # FaceTime / локальная
python app.py --source http://IP:8080/video
```

## CLI без браузера

```bash
python classify.py --list
python classify.py --source auto
```

## Что внутри

| Файл | Зачем |
|------|--------|
| `app.py` | Flask UI, HTTPS, стрим, история проездов |
| `classify.py` | OpenCV: ROI, контур, габариты, K, B/C/D |
| `side_view.py` | боковая камера: высота + K по вертикали |
| `segment_refine.py` | FastSAM подчищает маску (если есть веса/torch) |
| `camera.py` | источники: Continuity / индекс / HTTP |
| `templates/`, `static/` | страница оператора и `/phone` |

Сверху видны две стороны основания и \(K_{xy}\). Лежащий цилиндр сверху похож на прямоугольник — без бокового вида его легко отправить в B. Боковая камера даёт третье измерение и сечение по вертикали.

## Тесты

```bash
python -m pytest tests/ -q
# или
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Связка с sim

Правила классов общие. Здесь — восприятие с реальной камеры и UI для стола. Ветка `sim` — Gazebo/ROS 2 с двумя шиберами и executive. На хакатоне обычно показывают оба трека: стол с CV и симуляцию линии.
