from fastapi import FastAPI, UploadFile, File
import torch, cv2, numpy as np, yaml
from models.multitask_unet import MultiTaskUNet
from utils.io import load_checkpoint

app = FastAPI(title='octmultitask')

_model = None
_cfg = None

@app.get('/health')
def health():
    return {"status": "ok"}

@app.post('/load')
def load(checkpoint_path: str):
    global _model, _cfg
    ckpt = load_checkpoint(checkpoint_path)
    _cfg = ckpt['cfg']
    _model = MultiTaskUNet(base=_cfg['model']['base_channels'], depth=_cfg['model']['depth'], dropout=_cfg['model']['dropout'], n_seg_classes=len(yaml.safe_load(open(_cfg['classes_file']))['classes']), n_cls_labels=len(_cfg['cls_head']['labels']))
    _model.load_state_dict(ckpt['model']); _model.eval()
    return {"status": "loaded"}

@app.post('/analyze')
def analyze(file: UploadFile = File(...)):
    assert _model is not None, 'model not loaded'
    data = np.frombuffer(file.file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    size = tuple(_cfg['segmentation']['image_size'])
    img = cv2.resize(img, size)
    x = torch.from_numpy((img.astype(np.float32)/255.).transpose(2,0,1)).unsqueeze(0)
    with torch.no_grad():
        out = _model(x)
    resp = {}
    if 'classification' in out:
        prob = torch.sigmoid(out['classification']).squeeze(0).tolist()
        resp['multilabel'] = {k: float(v) for k,v in zip(_cfg['cls_head']['labels'], prob)}
    return resp
