#!/usr/bin/env bash
set -e
python data/roboflow_download.py "$@"
python data/prepare_masks.py --root data_store/OCT/train --classes config/classes.yaml || true
python data/split_patientwise.py --root data_store/OCT --ratio 0.8 0.1 0.1
