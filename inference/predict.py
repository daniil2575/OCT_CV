import argparse, yaml
from pathlib import Path
import torch, cv2, numpy as np, json
from models.multitask_unet import MultiTaskUNet
from utils.io import load_checkpoint
from inference.postprocess import clean_mask

def preprocess(img_bgr, size=(512,512)):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    x = torch.from_numpy((img.astype(np.float32)/255.).transpose(2,0,1)).unsqueeze(0)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt['cfg']
    n_classes = len(yaml.safe_load(open(cfg['classes_file']))['classes']) if isinstance(cfg, dict) else 1
    model = MultiTaskUNet(base=cfg['model']['base_channels'], depth=cfg['model']['depth'], dropout=cfg['model']['dropout'], n_seg_classes=n_classes, n_cls_labels=len(cfg['cls_head']['labels'])).eval()
    model.load_state_dict(ckpt['model'])

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for p in Path(args.images).glob('*.*'):
        img0 = cv2.imread(str(p))
        x = preprocess(img0, size=tuple(cfg['segmentation']['image_size']))
        with torch.no_grad():
            out = model(x)
        if 'segmentation' in out:
            prob = torch.sigmoid(out['segmentation']).cpu().numpy()[0]
            for ci in range(prob.shape[0]):
                m = (prob[ci]>0.5).astype(np.uint8)*255
                m = clean_mask(m)
                cv2.imwrite(str(out_dir/f"{p.stem}__c{ci+1}.png"), m)
        if 'classification' in out:
            cls = torch.sigmoid(out['classification']).cpu().numpy()[0].tolist()
            json.dump({'name': p.name, 'cls_proba': cls}, open(out_dir/f'{p.stem}.json','w'))

if __name__ == '__main__':
    main()
