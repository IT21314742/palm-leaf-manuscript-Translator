import cv2
import numpy as np
from typing import List

def segment_lines(img: np.ndarray) -> List[np.ndarray]:
    """
    Segment image into lines using horizontal projection profile.
    Returns list of line images.
    """
    # Binarize if not already
    if img.max() > 1:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Invert: text=1, bg=0
    img_inv = 255-img
    h_proj = np.sum(img_inv, axis=1)
    threshold = np.max(h_proj) * 0.1
    lines = []
    in_line = False
    start = 0
    for i, val in enumerate(h_proj):
        if val > threshold and not in_line:
            in_line = True
            start = i
        elif val <= threshold and in_line:
            in_line = False
            end = i
            if end-start > 5:
                lines.append(img[start:end, :])
    if in_line:
        lines.append(img[start:, :])
    return lines

def segment_characters(line_img: np.ndarray) -> List[np.ndarray]:
    """
    Segment a line image into character images using vertical projection profile.
    Returns list of character images.
    """
    img_inv = 255-line_img
    v_proj = np.sum(img_inv, axis=0)
    threshold = np.max(v_proj) * 0.1
    chars = []
    in_char = False
    start = 0
    for i, val in enumerate(v_proj):
        if val > threshold and not in_char:
            in_char = True
            start = i
        elif val <= threshold and in_char:
            in_char = False
            end = i
            if end-start > 2:
                chars.append(line_img[:, start:end])
    if in_char:
        chars.append(line_img[:, start:])
    return chars
