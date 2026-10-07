#!/usr/bin/env bash
# Демонстрация: копирует снимки из demo в demo_run и обрабатывает копию.
set -e
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
rm -rf demo_run demo_run_masks
cp -r demo demo_run
python3 cloud_filter.py --input demo_run --masks demo_run_masks
