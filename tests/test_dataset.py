from datasets.coco_multimodal import CocoMultimodalDataset

def test_ds_instantiation(tmp_path):
    d = tmp_path/"split"/"images"; d.mkdir(parents=True)
    (tmp_path/"split"/"masks").mkdir(parents=True)
    import numpy as np, cv2, yaml
    img = (np.random.rand(64,64)*255).astype('uint8')
    cv2.imwrite(str(d/"a.png"), img)
    cv2.imwrite(str(tmp_path/"split"/"masks"/"a.png"), (img>128).astype('uint8'))
    (tmp_path/"classes.yaml").write_text(yaml.dump({'classes': ['X']}))
    ds = CocoMultimodalDataset(str(tmp_path/"split"), str(tmp_path/"classes.yaml"), image_size=(64,64))
    s = ds[0]; assert s['image'].shape[-2:] == (64,64)
