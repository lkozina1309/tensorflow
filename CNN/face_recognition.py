# Script for facial recognition

import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, datasets, models

img_height = 224
img_width = 224
batch_size = 16

ds_train = tf.keras.utils.image_dataset_from_directory(
	'data/faces', #path
	batch_size=batch_size,
	image_size = (img_height, img_width), 
	shuffle=True,
	seed=123,
	validation_split=0.2,
	subset="training",		
)

ds_validation = tf.keras.utils.image_dataset_from_directory(
	'data/faces', #path
	batch_size=batch_size,
	image_size = (img_height, img_width), 
	shuffle=True,
	seed=123,
	validation_split=0.2,
	subset="validation",		
)

names = ds_train.class_names
print(names)

normalization_layer = layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
	layers.RandomFlip("horizontal", input_shape = (img_height, img_width,3)),
	tf.keras.layers.RandomRotation(0.1),
	tf.keras.layers.RandomZoom(0.1),
])

num_classes = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False,  weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
	base_model,
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
	tf.keras.layers.Conv2D(32, 3, activation='relu'),
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(32, activation='relu'),
	tf.keras.layers.Dense(num_classes)
])

model. compile(
	optimizer='adam',
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=["accuracy"],
)

model.fit(ds_train, validation_data=ds_validation, epochs=24)

loss, acc = model.evaluate(ds_validation)
print("Accuracy", acc) 

