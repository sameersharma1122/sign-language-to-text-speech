import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(data_path="data", img_size=64):
    categories = os.listdir(data_path)
    data, labels = [], []

    for idx, category in enumerate(categories):
        folder_path = os.path.join(data_path, category)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (img_size, img_size))
            data.append(img_resized)
            labels.append(idx)

    data = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
    labels = np.array(labels)
    return train_test_split(data, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
