from pathlib import Path
from typing import List, Dict, Optional
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from data.augmentations import get_train_augs, get_val_augs

class CocoMultimodalDataset(Dataset):
    """Читает изображения и индексные PNG-маски из структуры <split>/{images,masks}.
    Если use_masks=False — возвращает пустую маску.
    """
    def __init__(self, split_dir: str, classes_yaml: str, image_size=(512,512), is_train=True, use_masks=True):
        self.split_dir = Path(split_dir)
        with open(classes_yaml,'r') as f:
            self.classes = yaml.safe_load(f)['classes']
        self.n_cls = len(self.classes)
        self.image_size = image_size
        self.use_masks = use_masks
        self.augs = get_train_augs(image_size) if is_train else get_val_augs(image_size)
        self.images = sorted((self.split_dir/'images').glob('*.*'))

    def __len__(self):
        return len(self.images)

    def _read_mask(self, stem: str) -> np.ndarray:
        if not self.use_masks:
            return np.zeros((self.n_cls, *self.image_size), dtype=np.float32)
        mpath = self.split_dir/'masks'/f'{stem}.png'
        m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if m is None:
            m = np.zeros(self.image_size, dtype=np.uint8)
        if m.ndim==3:
            m = m[...,0]
        # one-hot по индексам 0..N
        onehot = np.zeros((self.n_cls, m.shape[0], m.shape[1]), dtype=np.float32)
        for i in range(self.n_cls):
            onehot[i] = (m==(i+1)).astype(np.float32)  # 0=фон, 1..N классы
        return onehot

    def __getitem__(self, i: int):
        ip = self.images[i]
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask = self._read_mask(ip.stem)
        tf = self.augs(image=img, mask=mask.transpose(1,2,0))
        img_t = tf['image']
        mask_t = tf['mask'].permute(2,0,1)
        multilabel = (mask_t.view(self.n_cls,-1).sum(dim=1)>0).float()
        return {
            'image': img_t,
            'mask': mask_t,
            'multilabel': multilabel,
            'name': ip.name
        }
