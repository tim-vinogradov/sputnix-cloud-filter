"""Отсев облачных спутниковых снимков.

Программа обрабатывает папку со снимками: нейросеть CloudNet оценивает процент
облачности каждого снимка, снимки с облачностью выше порога (по умолчанию 75 %)
удаляются вместе с парным KML-файлом, остальные сохраняются. По окончании
оператор получает сообщение с результатом обработки.

Примеры:
    python cloud_filter.py --input incoming/
    python cloud_filter.py --input incoming/ --threshold 75 --masks masks/ --gui
    python cloud_filter.py --input incoming/ --dry-run      # без удаления, только отчёт
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

from cloudnet import CloudEstimator

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def save_mask_preview(arr, mask, percent, path):
    """Сохраняет снимок с подсвеченной маской облаков (для контроля оператором)."""
    over = arr.astype(np.float32).copy()
    over[mask] = over[mask] * 0.45 + np.array([255, 70, 40]) * 0.55
    Image.fromarray(over.astype(np.uint8)).save(path, quality=90)


def pct(v, digits=1):
    return f"{v:.{digits}f}".replace(".", ",")


def build_message(saved, deleted, failed, threshold, dry_run):
    total = len(saved) + len(deleted) + len(failed)
    head = (f"Всего снимков: {total}; сохранено: {len(saved)}; "
            f"{'помечено к удалению' if dry_run else 'удалено'}: {len(deleted)}")
    if failed:
        head += f"; не обработано: {len(failed)}"
    lines = [f"Обработка снимков завершена: {dt.datetime.now():%d.%m.%Y %H:%M}",
             f"Порог отсева: облачность выше {pct(threshold, 0)} %", head, ""]
    lines.append("СОХРАНЁННЫЕ СНИМКИ")
    for r in saved:
        lines.append(f"  {r['file']}   облачность {pct(r['cloud'])} %")
    if not saved:
        lines.append("  нет")
    lines.append("")
    lines.append("УДАЛЁННЫЕ СНИМКИ" if not dry_run else "СНИМКИ К УДАЛЕНИЮ (пробный запуск)")
    for r in deleted:
        lines.append(f"  {r['file']}   облачность {pct(r['cloud'])} %")
    if not deleted:
        lines.append("  нет")
    if failed:
        lines.append("")
        lines.append("НЕ ОБРАБОТАНЫ (файлы оставлены без изменений)")
        for r in failed:
            lines.append(f"  {r['file']}   {r['error']}")
    return "\n".join(lines)


def find_kml(image_path):
    """Парный KML-файл с тем же именем (регистр расширения не важен)."""
    stem = os.path.splitext(image_path)[0]
    for ext in (".kml", ".KML", ".Kml"):
        if os.path.exists(stem + ext):
            return stem + ext
    return None


def show_gui(message):
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except ImportError:
        print("Окно не может быть открыто (в Python нет модуля tkinter), сообщение выведено выше.")
        return
    try:
        root = tk.Tk()
    except tk.TclError:
        print("Окно не может быть открыто (нет графической среды), сообщение выведено выше.")
        return
    root.title("Результат обработки снимков")
    box = scrolledtext.ScrolledText(root, width=90, height=30, font=("Consolas", 10))
    box.insert("1.0", message)
    box.configure(state="disabled")
    box.pack(padx=10, pady=10)
    tk.Button(root, text="OK", width=12, command=root.destroy).pack(pady=(0, 10))
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Отсев спутниковых снимков по облачности")
    ap.add_argument("--input", required=True, help="папка со снимками")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "weights", "cloudnet.pt"))
    ap.add_argument("--threshold", type=float, default=75.0, help="порог облачности, %%")
    ap.add_argument("--masks", default=None, help="папка для сохранения масок облаков")
    ap.add_argument("--report", default=None, help="куда сохранить отчёт (по умолчанию в папку снимков)")
    ap.add_argument("--dry-run", action="store_true", help="не удалять файлы, только отчёт")
    ap.add_argument("--gui", action="store_true", help="показать сообщение в окне")
    a = ap.parse_args()

    if not os.path.isdir(a.input):
        print("Папка не найдена:", a.input)
        sys.exit(2)
    if not 0 <= a.threshold <= 100:
        print("Порог облачности должен быть от 0 до 100 %")
        sys.exit(2)
    files = sorted(f for f in glob.glob(os.path.join(a.input, "*"))
                   if os.path.isfile(f) and f.lower().endswith(IMAGE_EXT))
    if not files:
        print("В папке нет снимков:", a.input)
        sys.exit(1)
    if not os.path.exists(a.weights):
        print("Не найден файл весов модели:", a.weights)
        sys.exit(2)

    est = CloudEstimator(a.weights)
    if a.masks:
        os.makedirs(a.masks, exist_ok=True)

    # 1. Оценка облачности всех снимков. Ошибка чтения одного файла не прерывает обработку.
    results, failed = [], []
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        try:
            percent, mask, _, arr = est.estimate(f)
        except Exception as e:  # битый, пустой или нечитаемый файл
            failed.append({"file": name, "error": f"не удалось прочитать снимок ({type(e).__name__})", "action": "ошибка"})
            print(f"[{i}/{len(files)}] {name}: ошибка чтения, файл пропущен")
            continue
        print(f"[{i}/{len(files)}] {name}: облачность {pct(percent)} %")
        if a.masks:
            save_mask_preview(arr, mask, percent, os.path.join(a.masks, os.path.splitext(name)[0] + "_mask.jpg"))
        results.append((f, {"file": name, "cloud": round(float(percent), 1)}))

    # 2. Удаление снимков с облачностью выше порога вместе с KML.
    saved, deleted = [], []
    for f, rec in results:
        if rec["cloud"] > a.threshold:
            if not a.dry_run:
                try:
                    kml = find_kml(f)
                    os.remove(f)
                    if kml:
                        os.remove(kml)
                        rec["kml"] = os.path.basename(kml)
                except OSError as e:
                    failed.append({"file": rec["file"], "error": f"облачность {pct(rec['cloud'])} %, но удалить не удалось ({e.strerror})", "action": "ошибка удаления"})
                    continue
            rec["action"] = "удалён" if not a.dry_run else "к удалению"
            deleted.append(rec)
        else:
            rec["action"] = "сохранён"
            saved.append(rec)

    message = build_message(saved, deleted, failed, a.threshold, a.dry_run)
    print("\n" + message)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = a.report or os.path.join(a.input, f"cloud_report_{stamp}")
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(message)
    with open(base + ".csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["Файл", "Облачность, %", "Решение"])
        for r in saved + deleted:
            w.writerow([r["file"], pct(r["cloud"]), r["action"]])
        for r in failed:
            w.writerow([r["file"], "", r["error"]])
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump({"threshold": a.threshold, "dry_run": a.dry_run, "saved": saved, "deleted": deleted, "failed": failed}, fh, ensure_ascii=False, indent=1)
    print(f"\nОтчёт сохранён: {base}.txt / .csv / .json")

    if a.gui:
        show_gui(message)
    sys.exit(3 if failed else 0)


if __name__ == "__main__":
    main()
