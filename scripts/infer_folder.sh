#!/usr/bin/env bash
set -e
python inference/predict.py --images "$1" --checkpoint runs/best.pt --out runs/preds
