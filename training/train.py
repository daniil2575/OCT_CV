import argparse, yaml, os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from datasets.coco_multimodal import CocoMultimodalDataset
from models.multitask_unet import MultiTaskUNet
from utils.seed import set_seed
from utils.losses import build_loss
from utils.metrics import dice_from_logits, auc_roc
from training.callbacks import EarlyStopping, Checkpoint

def load_classes(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)['classes']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/project.yaml')
    args = ap.parse_args()

    with open(args.config,'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed',42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = load_classes(cfg['classes_file'])

    ds_list_train, ds_list_val = [], []
    for mod in cfg['modalities']:
        base = Path(cfg['paths']['data_store'])/mod
        ds_list_train.append(CocoMultimodalDataset(str(base/'train'), cfg['classes_file'], cfg['segmentation']['image_size'], True, cfg['tasks']['segmentation']))
        ds_list_val.append(CocoMultimodalDataset(str(base/'val'),   cfg['classes_file'], cfg['segmentation']['image_size'], False, cfg['tasks']['segmentation']))
    train_ds = ConcatDataset(ds_list_train)
    val_ds   = ConcatDataset(ds_list_val)

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training']['num_workers'], pin_memory=True)

    n_cls_labels = len(cfg['cls_head']['labels']) if cfg['tasks']['classification'] else 0
    model = MultiTaskUNet(in_ch=3, base=cfg['model']['base_channels'], depth=cfg['model']['depth'], dropout=cfg['model']['dropout'], n_seg_classes=len(classes) if cfg['tasks']['segmentation'] else 0, n_cls_labels=n_cls_labels).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['training']['amp'])

    seg_loss = build_loss(cfg['losses']['seg']) if cfg['tasks']['segmentation'] else None
    cls_loss = build_loss(cfg['losses']['cls']) if cfg['tasks']['classification'] else None

    ckpt_path = Path(cfg['paths']['run_dir'])/'best.pt'
    ckpt = Checkpoint(str(ckpt_path))
    stopper = EarlyStopping(patience=7)

    for epoch in range(1, cfg['training']['epochs']+1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            x = batch['image'].to(device)
            y_mask = batch['mask'].to(device)
            y_cls = batch['multilabel'][:, :n_cls_labels].to(device) if n_cls_labels>0 else None
            opt.zero_grad()
            with torch.cuda.amp.autocast(cfg['training']['amp']):
                out = model(x)
                loss = 0.0
                if 'segmentation' in out and seg_loss is not None:
                    loss = loss + seg_loss(out['segmentation'], y_mask)
                if 'classification' in out and cls_loss is not None:
                    loss = loss + cls_loss(out['classification'], y_cls)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss))

        # validation
        model.eval(); dice_sum=0.0; n=0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['image'].to(device)
                y_mask = batch['mask'].to(device)
                out = model(x)
                if 'segmentation' in out:
                    dice_sum += dice_from_logits(out['segmentation'], y_mask)
                    n+=1
        dice = dice_sum/max(n,1)
        print(f'val/dice={dice:.4f}')

        if stopper.best is None or dice>= stopper.best:
            ckpt.save(model, cfg)
        stopper.step(dice)
        if stopper.should_stop:
            break

if __name__ == '__main__':
    main()
