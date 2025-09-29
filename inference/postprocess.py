import cv2
import numpy as np

def clean_mask(mask_bin: np.ndarray, kernel=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel,kernel))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k)
    return mask_bin
