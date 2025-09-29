# Roboflow COCO Segmentation (кратко)

* `annotations.coco.json` содержит `images`, `annotations`, `categories`.
* Поля `annotations`: `image_id`, `category_id`, `segmentation` (полилинии), `bbox`, `iscrowd`.
* Мы используем PNG-маски `masks/<stem>.png` с индексами классов согласно `config/classes.yaml`.
