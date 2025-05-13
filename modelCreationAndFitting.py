import kagglehub
import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator

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

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))




# dataset: https://www.kaggle.com/datasets/niten19/face-shape-dataset/data

# Download latest version
path = kagglehub.dataset_download("niten19/face-shape-dataset")
print("Path to dataset files:", path)

import cv2

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


# Define the CNN model
def create_head_shape_model(input_shape=(86, 86, 3), num_classes=5):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening the output from convolutional layers
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # To prevent overfitting

    # Output layer for head shape classification (e.g., 5 head shapes)
    model.add(Dense(num_classes, activation='softmax'))

    return model

opt = keras.optimizers.Adam(learning_rate=0.0008764)
# Create and compile the model
head_shape_model = create_head_shape_model(input_shape=(86, 86, 3), num_classes=5)
head_shape_model.compile(
    optimizer=opt,
    loss='categorical_crossentropy', metrics=['accuracy','precision', 'recall'])

# Summary of the model
head_shape_model.summary()

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

# Save as CSV
# df.to_csv("training_set.csv", index=False)
# print("CSV file 'training_set.csv' has been created successfully.")


# Error processing /kaggle/input/face-shape-dataset/FaceShape Dataset/training_set/Heart/heart (633).jpg: image file is truncated (8 bytes not processed)
# Error processing /kaggle/input/face-shape-dataset/FaceShape Dataset/training_set/Square/square (84).jpg: image file is truncated (0 bytes not processed)


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

# Save as CSV for testing
# df.to_csv("testing_set.csv", index=False)
# print("CSV file 'testing_set.csv' has been created successfully.")

#training the data with the training df

#split data into 80,20
train, test = train_test_split(df, test_size=0.2)

#separate the label
label = 'label'
y_train = train[label]
y_test = test[label]
x_train = train.drop(label, axis=1)
x_test = test.drop(label, axis=1)

# Assuming images were flattened to 2352 = 28 x 28 x 3
x_train = x_train.values.reshape(-1, 86, 86, 3)
x_test = x_test.values.reshape(-1, 86, 86, 3)

# Step 1: Convert string labels to integer labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

#categorize head shapes
y_train = keras.utils.to_categorical(y_train_encoded, num_classes=5)
y_test = keras.utils.to_categorical(y_test_encoded, num_classes=5)

# Create the callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=40,  # Let it run longer before giving up
    mode='min',
    restore_best_weights=True
)



lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,  # Triggers before early stopping
    verbose=1,
    min_lr=1e-6
)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)



# Train the model with the callback
head_shape_model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=500,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
)


head_shape_model.save('/res/last_model1.keras')



