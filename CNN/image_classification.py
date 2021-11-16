import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 180
img_width = 180
batch_size = 32

ds_train = tf.keras.utils.image_dataset_from_directory(
	'/home/marija/tensorflow/CNN/data/vehicles', #path
	batch_size=batch_size,
	image_size = (img_height, img_width), 
	shuffle=True,
	seed=123,
	validation_split=0.2,
	subset="training",		
)

ds_validation = tf.keras.utils.image_dataset_from_directory(
	'/home/marija/tensorflow/CNN/data/vehicles', #path
	batch_size=batch_size,
	image_size = (img_height, img_width), 
	shuffle=True,
	seed=123,
	validation_split=0.2,
	subset="validation",		
)

class_names = ds_train.class_names
print(class_names)


AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
ds_validation = ds_validation.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
	layers.RandomFlip("horizontal", input_shape = (img_height, img_width,3)),
	tf.keras.layers.RandomRotation(0.1),
	tf.keras.layers.RandomZoom(0.1),
])	

num_classes = 5

model = tf.keras.Sequential([
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
	tf.keras.layers.Conv2D(32, 3, activation='relu'),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(64, 3, activation='relu'),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(128, 3, activation='relu'),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(num_classes),
])
		
model. compile(
	optimizer='adam',
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=["accuracy"],
)

model.fit(ds_train, validation_data=ds_validation, epochs=4)

loss, acc = model.evaluate(ds_validation)
print("Accuracy", acc)

