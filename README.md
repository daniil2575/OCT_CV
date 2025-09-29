# octmultitask

Мультитаск-модель для OCT/FUNDUS: сегментация классов (PNG-маски, индексная разметка), бинарная классификация наличия признаков, постпроцессинг, API-сервис.

## Быстрый старт
```bash
# 1) установка
pip install -U pip
pip install -e .
cp .env.example .env

# 2) подготовка данных (для OCT)
python data/roboflow_download.py --workspace '***' --project '***'--version 3 --modality OCT
python data/prepare_masks.py --root data_store/OCT/train --classes config/classes.yaml
python data/split_patientwise.py --root data_store/OCT --ratio 0.8 0.1 0.1

# 3) обучение
python training/train.py --config config/project.yaml

# 4) инференс на папке
python inference/predict.py --images data_store/OCT/test/images --checkpoint runs/best.pt --out runs/preds

# 5) API
uvicorn service.app:app --host '***' --port '***'
```

## Формат масок
`masks/<stem>.png` — однослойная PNG с индексами классов: 0=фон, 1..N — классы в порядке из `config/classes.yaml`.

## Задачи
- **Seg**: per-pixel многоклассовая (sigmoid per-class, one-vs-all) или softmax — по конфигу.
- **Cls**: мульти-лейбл по глобальному пулу признаков.

## Лицензия
MIT
