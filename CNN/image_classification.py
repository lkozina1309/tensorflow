import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


img_height = 180
img_width = 180
batch_size = 32

ds_train = tf.keras.utils.image_dataset_from_directory(
	'/data/vehicles', #path
	batch_size=batch_size,
	image_size = (img_height, img_width), 
	shuffle=True,
	seed=123,
	validation_split=0.2,
	subset="training",		
)

ds_validation = tf.keras.utils.image_dataset_from_directory(
	'data/vehicles', #path
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

datagen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')	

num_classes = 5

model = tf.keras.Sequential([
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
	tf.keras.layers.Conv2D(8, 3, activation='relu'),
	tf.keras.layers.MaxPooling2D((2,2)),
	tf.keras.layers.Conv2D(16, 3, activation='relu'),
	tf.keras.layers.MaxPooling2D((2,2)),
	tf.keras.layers.Conv2D(32, 3, activation='relu'),
	tf.keras.layers.MaxPooling2D((2,2)),
	tf.keras.layers.Conv2D(64, 3, activation='relu'),
	tf.keras.layers.MaxPooling2D((2,2)),
	tf.keras.layers.Conv2D(64, 3, activation='relu'),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(num_classes),
])
		
model. compile(
	optimizer='adam',
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=["accuracy"],
)

model.fit(ds_train, validation_data=ds_validation, epochs=15)

loss, acc = model.evaluate(ds_validation)
print("Accuracy", acc)
