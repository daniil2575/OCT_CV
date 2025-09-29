import os, argparse, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

"""Скачивает датасет Roboflow в data_store/<MODALITY>/{train,val,test}/ ...
Ожидается формат coco-segmentation. Требуется RF_API_KEY.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', required=True)
    ap.add_argument('--project', required=True)
    ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--modality', choices=['OCT','FUNDUS'], required=True)
    args = ap.parse_args()

    api_key = os.getenv('RF_API_KEY')
    assert api_key, 'RF_API_KEY не задан в .env'

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(args.workspace).project(args.project).version(args.version)
    out = Path(os.getenv('DATA_STORE', 'data_store')) / args.modality
    out.mkdir(parents=True, exist_ok=True)
    path = ds.download('coco-segmentation', location=str(out))
    print(json.dumps({"downloaded_to": str(path)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
