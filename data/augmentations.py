import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augs(size=(512,512)):
    return A.Compose([
        A.LongestMaxSize(max_size=max(size)),
        A.PadIfNeeded(min_height=size[0], min_width=size[1], border_mode=0, value=0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=7, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_val_augs(size=(512,512)):
    return A.Compose([
        A.LongestMaxSize(max_size=max(size)),
        A.PadIfNeeded(min_height=size[0], min_width=size[1], border_mode=0, value=0),
        A.Normalize(),
        ToTensorV2(),
    ])
