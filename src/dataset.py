import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from config import *

DATA_DIR = "data"

def load_data():
    images, labels = [], []

    for label, class_name in enumerate(CLASSES):
        class_folder = os.path.join(DATA_DIR, class_name)

        if not os.path.isdir(class_folder):
            raise FileNotFoundError(f"{class_folder} not found")

        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)

            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: could not read {image_path}")
                continue

            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0
            images.append(img)
            labels.append(label)

    x = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp,
        test_size=VAL_SPLIT,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return x_train, x_val, x_test, y_train, y_val, y_test