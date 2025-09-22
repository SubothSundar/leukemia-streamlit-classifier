import os
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def create_datasets(
	data_dir: str,
	img_size: Tuple[int, int] = (224, 224),
	batch_size: int = 32,
	val_split: float = 0.2,
	seed: int = 1337,
):
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		validation_split=val_split,
		subset="training",
		seed=seed,
		image_size=img_size,
		batch_size=batch_size,
	)
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		validation_split=val_split,
		subset="validation",
		seed=seed,
		image_size=img_size,
		batch_size=batch_size,
	)

	class_names = train_ds.class_names
	# Cache and prefetch for performance
	autotune = tf.data.AUTOTUNE
	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
	val_ds = val_ds.cache().prefetch(buffer_size=autotune)

	return train_ds, val_ds, class_names


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
	data_augmentation = tf.keras.Sequential(
		[
			layers.RandomFlip("horizontal"),
			layers.RandomRotation(0.1),
			layers.RandomZoom(0.1),
		]
	)

	inputs = layers.Input(shape=input_shape)
	x = data_augmentation(inputs)
	x = layers.Rescaling(1.0 / 255)(x)

	# Simple CNN
	x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
	x = layers.MaxPooling2D()(x)
	x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
	x = layers.MaxPooling2D()(x)
	x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
	x = layers.MaxPooling2D()(x)
	x = layers.Dropout(0.3)(x)
	x = layers.Flatten()(x)
	x = layers.Dense(128, activation="relu")(x)
	x = layers.Dropout(0.3)(x)
	outputs = layers.Dense(num_classes, activation="softmax")(x)

	model = models.Model(inputs, outputs)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def train_and_save(
	data_dir: str = "MyData",
	img_size: Tuple[int, int] = (224, 224),
	batch_size: int = 32,
	epochs: int = 10,
	model_path: str = "model2.keras",
	labels_path: str = "labels.txt",
):
	train_ds, val_ds, class_names = create_datasets(
		data_dir=data_dir, img_size=img_size, batch_size=batch_size
	)
	model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=len(class_names))

	callbacks = [
		tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
	]

	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs,
		callbacks=callbacks,
	)

	model.save(model_path)
	with open(labels_path, "w", encoding="utf-8") as f:
		for name in class_names:
			f.write(f"{name}\n")

	print(f"Saved model to {model_path} and labels to {labels_path}")


if __name__ == "__main__":
	data_dir = os.environ.get("DATA_DIR", "MyData")
	img_height = int(os.environ.get("IMG_HEIGHT", "224"))
	img_width = int(os.environ.get("IMG_WIDTH", "224"))
	batch_size = int(os.environ.get("BATCH_SIZE", "32"))
	epochs = int(os.environ.get("EPOCHS", "10"))
	model_path = os.environ.get("MODEL_PATH", "model2.keras")
	labels_path = os.environ.get("LABELS_PATH", "labels.txt")

	train_and_save(
		data_dir=data_dir,
		img_size=(img_height, img_width),
		batch_size=batch_size,
		epochs=epochs,
		model_path=model_path,
		labels_path=labels_path,
	)


