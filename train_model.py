import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = r"D:\feb 2013"

img_size = 48
batch_size = 64

# ✅ Data Augmentation (IMPORTANT)
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    DATASET_PATH + r"\train",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    DATASET_PATH + r"\test",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

# ✅ Improved CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🔥 Train longer
model.fit(train_data, epochs=30, validation_data=test_data)

# ✅ Save model
model.save("emotion_model.h5")
print("✅ Improved model saved!")
