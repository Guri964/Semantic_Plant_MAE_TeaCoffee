import cv2
import numpy as np
from config import COLOR_RANGES, COLOR_THRESHOLDS, COLOR_CAP_RATIOS


def get_leaf_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([20, 30, 30])
    upper_green = np.array([90, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((7, 7), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    leaf_pixel_count = np.sum(leaf_mask > 0)
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    if leaf_pixel_count < total_pixels * 0.1:
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, leaf_mask = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY)
    return leaf_mask


def detect_colors(img_bgr, leaf_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    leaf_pixel_count = np.sum(leaf_mask > 0)

    detected_colors = []
    disease_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for color_name, ranges in COLOR_RANGES.items():
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined = cv2.bitwise_or(combined, m)
        mask = cv2.bitwise_and(combined, leaf_mask)
        pixel_count = np.sum(mask > 0)
        ratio = pixel_count / leaf_pixel_count if leaf_pixel_count > 0 else 0

        if ratio > COLOR_THRESHOLDS[color_name]:
            detected_colors.append(color_name)
            if color_name in COLOR_CAP_RATIOS:
                cap = COLOR_CAP_RATIOS[color_name]
                capped_pixels = min(pixel_count, int(leaf_pixel_count * cap))
                cap_mask = np.zeros_like(mask)
                indices = np.where(mask > 0)
                cap_mask[indices[0][:capped_pixels], indices[1][:capped_pixels]] = 255
                disease_mask = cv2.bitwise_or(disease_mask, cap_mask)
            else:
                disease_mask = cv2.bitwise_or(disease_mask, mask)

    return detected_colors, disease_mask, leaf_pixel_count