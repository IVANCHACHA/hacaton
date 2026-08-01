# OZON CV Sorter

Классификация товаров с **iPhone Continuity Camera** и веб-интерфейсом со стримом.

| Зона | Категория |
|------|-----------|
| B | Подходит для сортировки |
| C | Не подходит по габаритам |
| D | Не подходит без доупаковки (круг) |

## Запуск UI

```bash
cd /Users/irwkc/Developer/ozon
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

1. На Mac: [https://127.0.0.1:5050](https://127.0.0.1:5050) (самоподписанный сертификат — «продолжить»)
2. На iPhone (та же Wi‑Fi): `https://<IP-мака>:5050/phone`
   - Safari покажет предупреждение → **Показать подробности → Посетить сайт**
   - затем **Включить камеру** → появится системный запрос

Без HTTPS iOS Safari **не вызывает** диалог камеры (`getUserMedia` только в secure context).

По умолчанию используется **Safari на iPhone** как камера. Continuity Camera на многих Mac отдаёт чёрный кадр — поэтому phone-режим надёжнее.

```bash
python app.py --source phone     # iPhone Safari (по умолчанию)
python app.py --source auto      # Continuity (часто чёрный)
python app.py --source 0         # FaceTime
python app.py --source http://IP:8080/video
```

## CLI без браузера

```bash
python classify.py --list
python classify.py --source auto
```
