# data_preprocessing.py

import cv2
import numpy as np

# 归一化函数
def normalize_image(image):
    return image / 255.0

# 使用光流技术进行图像对齐
def align_images(images):
    aligned_images = [images[0]]  # 保留第一个图像
    for i in range(1, len(images)):
        prev_img = images[i-1]
        next_img = images[i]
        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        aligned_img = cv2.remap(next_img, flow, None, cv2.INTER_LINEAR)
        aligned_images.append(aligned_img)
    return aligned_images

# 加载和预处理数据
def preprocess_data(image_paths):
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    images = [normalize_image(img) for img in images]
    aligned_images = align_images(images)
    return aligned_images
