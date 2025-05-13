import kagglehub
import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


import tensorflow as tf
# Download latest version
path = kagglehub.dataset_download("niten19/face-shape-dataset")
print("Path to dataset files:", path)


# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_and_crop_face(image_path, target_size=(86, 86)):
    """Detects face in an image, crops and resizes it."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None  # Skip if no face is detected

    x, y, w, h = faces[0]
    face_img = img[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, target_size)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL compatibility
    return face_img


# Base directory
base_path = path+"/FaceShape Dataset/training_set"  # Replace 'path' with your actual base path

# Resize all images to a consistent shape
image_size = (86, 86)

data = []

# Loop through each labeled subdirectory
for label in ["Oval", "Heart", "Oblong", "Square", "Round"]:
    dir_path = os.path.join(base_path, label)
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(dir_path, filename)
            try:
                face = detect_and_crop_face(filepath, target_size=image_size)
                if face is None:
                    print(f"No face detected in {filepath}")
                    continue
                img_array = np.array(face).astype('float32') / 255.0
                img_array = img_array.flatten()
                row = [label] + img_array.tolist()
                data.append(row)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

# Prepare column names: first is 'label', rest are 'pixel_0', ..., 'pixel_n'
column_names = ['label'] + [f'pixel_{i}' for i in range(image_size[0] * image_size[1] * 3)]
df = pd.DataFrame(data, columns=column_names)
df.to_csv('training_set.csv', index=False)


# Base directory
base_path = path+"/FaceShape Dataset/testing_set"  # Replace 'path' with your actual base path

# Resize all images to a consistent shape
image_size = (86, 86)

data2 = []

# Loop through each labeled subdirectory
for label in ["Oval", "Heart", "Oblong", "Square", "Round"]:
    dir_path = os.path.join(base_path, label)
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(dir_path, filename)
            try:
                img = Image.open(filepath).convert("RGB").resize(image_size)
                img_array = np.array(img).astype('float32') / 255.0 #normalizes the data
                img_array = img_array.flatten()
                row = [label] + img_array.tolist()
                data2.append(row)
            except Exception as e:
                #insert fix for images
                #loop back to try
                print(f"Error processing {filepath}: {e}")

# Prepare column names: first is 'label', rest are 'pixel_0', ..., 'pixel_n'
column_names = ['label'] + [f'pixel_{i}' for i in range(image_size[0] * image_size[1] * 3)]
testing_df = pd.DataFrame(data2, columns=column_names)
testing_df.to_csv('testing.csv', index=False)
