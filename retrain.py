import pandas as pd
import matplotlib.pyplot as plt
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

df = pd.read_csv('res/training_set.csv')
df2 = pd.read_csv('res/testing.csv')

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


model = load_model('last_model1.keras') #change model name here

# run if want to change optimizer learning rate and such
from tensorflow.keras.optimizers import Adam, SGD
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,  # Let it run longer before giving up
    mode='min',
    restore_best_weights=True
)



lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=5,  # Triggers before early stopping
    verbose=1,
    min_lr=1e-6
)

import numpy as np
from sklearn.utils import class_weight

# Convert one-hot encoded y_train to class labels
y_train_labels = np.argmax(y_train, axis=1)

# Now compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

# Convert to dictionary format for Keras
class_weights = dict(enumerate(class_weights))


# Train the model with class weights
history = model.fit(
    x_train, y_train,
    epochs=1000,
    validation_data=(x_train, y_train),
    class_weight=class_weights,
    callbacks=[lr_schedule, early_stop]
)

#save model
model.save('/res/last_model1.keras')
print("model saved")


# Plot Loss
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Recall
plt.subplot(1, 3, 3)
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Val Recall')
plt.title('Recall Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()
plt.show()