import json, cv2, numpy as np, yaml
from pathlib import Path

def rle_decode_uncompressed(counts, h, w):
    # counts: список чисел; порядок заполнения — колонками (Fortran order), как в COCO
    arr = np.zeros(h*w, dtype=np.uint8)
    idx = 0; val = 0
    for c in counts:
        if val == 1:
            arr[idx:idx+c] = 1
        idx += c
        val ^= 1
    return arr.reshape((h, w), order="F")

def coco_to_masks(split_dir, classes_yaml):
    split = Path(split_dir)
    # ищем JSON: annotations.coco.json или любой *annotations*.json
    cand = list(split.glob("annotations.coco.json")) + list(split.glob("*annotations*.json"))
    if not cand:
        raise SystemExit(f"No COCO json in {split}")
    coco_json = cand[0]
    coco = json.load(open(coco_json, "r", encoding="utf-8"))
    classes = yaml.safe_load(open(classes_yaml, "r", encoding="utf-8"))["classes"]

    # map category_id -> индекс класса (1..N), 0 = фон
    id2idx = {}
    for c in coco.get("categories", []):
        if c.get("name") in classes:
            id2idx[c["id"]] = classes.index(c["name"]) + 1

    (split / "masks").mkdir(parents=True, exist_ok=True)

    # сгруппируем аннотации по image_id
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for im in coco.get("images", []):
        h, w = int(im["height"]), int(im["width"])
        mask = np.zeros((h, w), np.uint8)
        for ann in anns_by_img.get(im["id"], []):
            cls_idx = id2idx.get(ann["category_id"])
            if not cls_idx:  # неизвестная категория
                continue
            seg = ann.get("segmentation")
            if isinstance(seg, list):  # polygons
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    pts = np.round(pts).astype(np.int32)
                    cv2.fillPoly(mask, [pts], int(cls_idx))
            elif isinstance(seg, dict) and "counts" in seg and "size" in seg:
                counts = seg["counts"]
                if isinstance(counts, list):  # uncompressed RLE
                    m = rle_decode_uncompressed(counts, h, w)
                    mask = np.where(m > 0, int(cls_idx), mask).astype(np.uint8)
                else:
                    raise SystemExit(
                        "Found compressed RLE (string 'counts'). "
                        "Переэкспортируй из Roboflow как COCO (polygons) или PNG masks."
                    )
            else:
                # неизвестный формат сегментации
                continue

        out = split / "masks" / (Path(im["file_name"]).stem + ".png")
        cv2.imwrite(str(out), mask)
    print(f"done -> {split/'masks'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: py data\\coco_to_masks.py <split_dir> <classes_yaml>")
        raise SystemExit(1)
    coco_to_masks(sys.argv[1], sys.argv[2])