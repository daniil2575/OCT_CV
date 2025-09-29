import argparse, os, yaml, math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from datasets.coco_multimodal import CocoMultimodalDataset
from models.multitask_unet import MultiTaskUNet
from utils.seed import set_seed
from utils.losses import build_loss
from utils.metrics import dice_from_logits


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["classes"]


def make_datasets(cfg: dict, classes_file: str, img_size, use_masks=True, max_items_per_split=None):
    ds_train, ds_val = [], []
    for mod in cfg["modalities"]:
        base = Path(cfg["paths"]["data_store"]) / mod
        tr = CocoMultimodalDataset(str(base / "train"), classes_file, img_size, is_train=True, use_masks=use_masks)
        va = CocoMultimodalDataset(str(base / "val"),   classes_file, img_size, is_train=False, use_masks=use_masks)
        if max_items_per_split:
            tr = Subset(tr, range(min(len(tr), max_items_per_split)))
            va = Subset(va, range(min(len(va), max_items_per_split)))
        ds_train.append(tr)
        ds_val.append(va)
    return ConcatDataset(ds_train), ConcatDataset(ds_val)


def train_one_trial(trial: optuna.Trial, base_cfg: dict, classes: list, device: torch.device, max_train_batches: int):
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy через yaml

    #  поиск гиперов 
    # архитектура
    cfg["model"]["base_channels"] = trial.suggest_categorical("base_channels", [32, 48, 64, 96])
    cfg["model"]["depth"]        = trial.suggest_int("depth", 3, 5)
    cfg["model"]["dropout"]      = trial.suggest_float("dropout", 0.0, 0.3)

    # оптимизация
    cfg["training"]["lr"]         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    cfg["training"]["batch_size"] = trial.suggest_categorical("batch_size", [2, 4, 8])
    cfg["training"]["epochs"]     = trial.suggest_int("epochs", 4, 10)
    cfg["losses"]["seg"]["type"]  = trial.suggest_categorical("seg_loss", ["dice", "dice_bce"])
    cfg["losses"]["seg"]["pos_weight"] = trial.suggest_float("seg_pos_weight", 0.5, 2.0)

    # размер входа (уменьшить можно)
    side = trial.suggest_categorical("image_side", [384, 448, 512])
    cfg["segmentation"]["image_size"] = [side, side]

    #  данные 
    use_masks = bool(cfg["tasks"].get("segmentation", True))
    # для скорости ограничим число сэмплов на сплит (можно поднять)
    max_items = 600 if side <= 448 else 400
    train_ds, val_ds = make_datasets(cfg, cfg["classes_file"], cfg["segmentation"]["image_size"],
                                     use_masks=use_masks, max_items_per_split=max_items)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=cfg["training"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=max(2, cfg["training"]["batch_size"]),
                              shuffle=False, num_workers=cfg["training"]["num_workers"], pin_memory=True)

    # ---- модель/лоссы ----
    n_cls_labels = len(cfg.get("cls_head", {}).get("labels", [])) if cfg["tasks"].get("classification", False) else 0
    model = MultiTaskUNet(
        in_ch=3,
        base=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
        dropout=cfg["model"]["dropout"],
        n_seg_classes=len(classes) if use_masks else 0,
        n_cls_labels=n_cls_labels
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"])

    seg_loss = build_loss(cfg["losses"]["seg"]) if use_masks else None
    cls_loss = None  # при желании добавьте bce и метки для классификации

    # ---- цикл обучения ----
    best_dice = -1.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, total=min(len(train_loader), max_train_batches), leave=False, desc=f"trn e{epoch}")
        for i, batch in enumerate(pbar):
            if i >= max_train_batches:
                break
            x = batch["image"].to(device)
            y_mask = batch["mask"].to(device) if use_masks else None
            opt.zero_grad()
            with torch.cuda.amp.autocast(cfg["training"]["amp"]):
                out = model(x)
                loss = 0.0
                if use_masks and "segmentation" in out:
                    loss = loss + seg_loss(out["segmentation"], y_mask)
                if "classification" in out and cls_loss is not None:
                    # пример: y_cls = batch['multilabel'][:, :n_cls_labels].to(device)
                    # loss += cls_loss(out['classification'], y_cls)
                    pass
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss))

        # ---- валидация ----
        model.eval(); dice_sum = 0.0; n = 0
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                if j >= math.ceil(max_train_batches / 2):  # быстрая валидация
                    break
                x = batch["image"].to(device)
                y_mask = batch["mask"].to(device) if use_masks else None
                out = model(x)
                if use_masks and "segmentation" in out:
                    dice_sum += dice_from_logits(out["segmentation"], y_mask); n += 1
        dice = dice_sum / max(n, 1)
        trial.report(dice, epoch)

        if dice > best_dice:
            best_dice = dice
        # прунинг, если метрика плохая
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/project.yaml")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--max-train-batches", type=int, default=100, help="сколько батчей обучать на эпоху (ускорение)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    base_cfg = load_cfg(args.config)
    classes = load_classes(base_cfg["classes_file"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=args.seed),
                                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    study.optimize(lambda t: train_one_trial(t, base_cfg, classes, device, args.max_train_batches),
                   n_trials=args.n_trials, show_progress_bar=True)

    print("Best value (Dice):", study.best_value)
    print("Best params:", study.best_trial.params)


if __name__ == "__main__":
    main()
