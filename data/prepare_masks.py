import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np

"""Унифицирует маски: собирает индексные PNG в masks/<stem>.png.
Если уже есть index-PNG, просто проверяет многоклассовую консистентность.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='папка сплита: .../train или .../val')
    ap.add_argument('--classes', required=True, help='config/classes.yaml')
    args = ap.parse_args()

    with open(args.classes, 'r') as f:
        classes = yaml.safe_load(f)['classes']
    n = len(classes)

    images = Path(args.root)/'images'
    masks_dir = Path(args.root)/'masks'
    masks_dir.mkdir(exist_ok=True, parents=True)

    for img_path in images.glob('*.*'):
        stem = img_path.stem
        # ожидаем уже существующий masks/<stem>.png, иначе пропускаем
        src = Path(args.root)/'masks'/f'{stem}.png'
        if not src.exists():
            # если разметка есть только в COCO — сгенерировать можно отдельно; для простоты пропустим
            continue
        m = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise RuntimeError(f'bad mask {src}')
        # проверим диапазон
        if m.max() > n:
            print(f'[warn] mask {src} has index>{n}, проверь соответствие classes.yaml')
        cv2.imwrite(str(masks_dir/f'{stem}.png'), m)

if __name__ == '__main__':
    main()
