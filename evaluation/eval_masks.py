import argparse, yaml, json
from pathlib import Path
import cv2
import numpy as np
from utils.metrics import dice_iou_from_binary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True, help='.../test/masks')
    ap.add_argument('--pred', required=True, help='папка с предсказанными *_cX.png')
    ap.add_argument('--classes', default='config/classes.yaml')
    ap.add_argument('--out_csv', default='runs/seg_report.csv')
    args = ap.parse_args()

    classes = yaml.safe_load(open(args.classes))['classes']
    rows = []
    for gt_mask in Path(args.gt).glob('*.png'):
        stem = gt_mask.stem
        gt = cv2.imread(str(gt_mask), cv2.IMREAD_UNCHANGED)
        for i, cls in enumerate(classes, start=1):
            pm = Path(args.pred)/f'{stem}__c{i}.png'
            if not pm.exists():
                continue
            pr = cv2.imread(str(pm), cv2.IMREAD_GRAYSCALE)
            pr = (pr>127).astype(np.uint8)
            dice, iou = dice_iou_from_binary((gt==i).astype(np.uint8), pr)
            rows.append((stem, cls, float(dice), float(iou)))
    # простая запись CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv,'w') as f:
        f.write('image,class,dice,iou\n')
        for r in rows:
            f.write(','.join(map(str,r))+'\n')
    print(f'wrote {args.out_csv}')

if __name__ == '__main__':
    main()
