import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True, help='папка с *.json от инференса')
    ap.add_argument('--labels_json', required=True, help='файл с GT: {name: [0/1,...]}')
    args = ap.parse_args()

    y_true_map = json.load(open(args.labels_json))
    y_true, y_prob = [], []
    for jp in Path(args.pred_dir).glob('*.json'):
        rec = json.load(open(jp))
        y_prob.append(rec['cls_proba'])
        y_true.append(y_true_map.get(rec['name']))
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    auc = roc_auc_score(y_true, y_prob, average='macro')
    y_pred = (y_prob>0.5).astype(int)
    f1 = f1_score(y_true, y_pred, average='macro')
    print({'AUC': float(auc), 'F1': float(f1)})

if __name__ == '__main__':
    main()
