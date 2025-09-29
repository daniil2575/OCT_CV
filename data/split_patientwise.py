import argparse, json, random
from pathlib import Path

"""Разбивает по пациентам (patient_id должен быть в имени файла: <patient>_<...>.png).
Пример: ratio 0.8 0.1 0.1 -> train/val/test.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='data_store/OCT')
    ap.add_argument('--ratio', nargs=3, type=float, default=[0.8,0.1,0.1])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    images = list((Path(args.root)/'images').glob('*.*'))
    if not images:
        # поддержим структуру Roboflow (<split>/images)
        images = []
        for sp in ['train','val','test']:
            images += list((Path(args.root)/sp/'images').glob('*.*'))
        if images:
            print('найдены уже готовые сплиты — ничего не делаем')
            return
        raise SystemExit('нет изображений для разбиения')

    # собрать patient_id
    patients = {}
    for p in images:
        pid = p.stem.split('_')[0]
        patients.setdefault(pid, []).append(p)

    pids = list(patients.keys()); random.shuffle(pids)
    n = len(pids)
    n_train = int(n*args.ratio[0]); n_val = int(n*args.ratio[1])
    splits = {
        'train': set(pids[:n_train]),
        'val': set(pids[n_train:n_train+n_val]),
        'test': set(pids[n_train+n_val:])
    }

    for sp in ['train','val','test']:
        (Path(args.root)/sp/'images').mkdir(parents=True, exist_ok=True)
        (Path(args.root)/sp/'masks').mkdir(parents=True, exist_ok=True)

    for pid, files in patients.items():
        # переносим в соответствующий сплит
        sp = 'train' if pid in splits['train'] else 'val' if pid in splits['val'] else 'test'
        for f in files:
            dst = Path(args.root)/sp/'images'/f.name
            if not dst.exists():
                dst.write_bytes(Path(f).read_bytes())
            # маску скопируем, если есть
            m = Path(args.root)/'masks'/f'{f.stem}.png'
            if m.exists():
                (Path(args.root)/sp/'masks'/m.name).write_bytes(m.read_bytes())

    print(json.dumps({k: len(v) for k,v in splits.items()}))

if __name__ == '__main__':
    main()
