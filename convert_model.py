import os
import tensorflow as tf


def main():
	# Source and targets
	src_h5 = os.environ.get("SRC_MODEL", "model2.keras")
	keras_out = os.environ.get("KERAS_OUT", "model2.keras")
	tflite_out = os.environ.get("TFLITE_OUT", "model_fp16.tflite")

	if not os.path.exists(src_h5):
		raise FileNotFoundError(f"Source model not found: {src_h5}")

	print(f"Loading model from {src_h5} ...")
	model = tf.keras.models.load_model(src_h5)

	# Save in Keras v3 native format (.keras)
	print(f"Saving Keras v3 model to {keras_out} ...")
	model.save(keras_out, save_format="keras")
	print(f"Saved: {keras_out} (exists={os.path.exists(keras_out)})")

	# Create a smaller TFLite (float16) variant as well
	print(f"Converting to TFLite (float16) -> {tflite_out} ...")
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_types = [tf.float16]
	tfl = converter.convert()
	with open(tflite_out, "wb") as f:
		f.write(tfl)
	print(f"Saved: {tflite_out} (exists={os.path.exists(tflite_out)})")

	print("Done.")


if __name__ == "__main__":
	main()


