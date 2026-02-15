import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


DATA_DIR = "data"
IMAGE_SIZE = [224,224]
IMAGE_SIZE_1 = [128,128]
CLASSES = ["with_mask", "without_mask"]
labels = []
images = []

for label in range(len(CLASSES)):
    class_name = CLASSES[label]
    class_folder = os.path.join(DATA_DIR, class_name)
    
    
    for image in os.listdir(class_folder):
        image_path = os.path.join(class_folder,image)
        
        img = cv2.imread(image_path)
        
        if img is None:
            continue
        
        img = cv2.resize(img, IMAGE_SIZE)
        img = img/255.0
        img = img.astype(np.float32)
        
        images.append(img)
        labels.append(label)
        
        
print("Images len", len(images))
print("Labels len", len(labels))

x = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.int32)


print("X shape", x.shape)
print("Y shape", y.shape)


x_temp, x_test, y_temp,y_test = train_test_split(x,y, test_size = 0.15, random_state=42, stratify = y)

x_train, x_val, y_train,y_val = train_test_split(x_temp, y_temp, random_state=42, test_size=0.176, stratify=y_temp)

print("Train:", x_train.shape)
print("Val:", x_val.shape)
print("Test:", x_test.shape)        