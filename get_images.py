"""Скачивает из открытого архива СПУТНИКС снимки, перечисленные в data/annotations.json.

Запуск: python get_images.py
Снимки сохраняются в data/images/ в уменьшенном размере 1024 x 769 (как при обучении).
"""
import json
import os
import urllib.parse
import urllib.request

SHARE = "https://disk.sputnix.ru/index.php/apps/files_sharing/publicpreview/HbkKc34batcjkSX"

ann = json.load(open("data/annotations.json", encoding="utf-8"))["labels"]
os.makedirs("data/images", exist_ok=True)
for i, name in enumerate(sorted(ann), 1):
    out = os.path.join("data", "images", name)
    if os.path.exists(out):
        continue
    url = f"{SHARE}?file=/{urllib.parse.quote(name)}&x=1024&y=1024&a=true"
    urllib.request.urlretrieve(url, out)
    print(f"[{i}/{len(ann)}] {name}")
print("готово")
