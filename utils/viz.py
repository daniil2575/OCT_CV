import cv2
import numpy as np

def overlay_mask(img_rgb: np.ndarray, mask_bin: np.ndarray, alpha=0.5):
    color = np.zeros_like(img_rgb); color[...,1]=255
    overlay = cv2.addWeighted(img_rgb, 1-alpha, color, alpha, 0)
    out = np.where(mask_bin[...,None]>0, overlay, img_rgb)
    return out
