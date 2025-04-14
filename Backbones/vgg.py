import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19

def build_vgg(variant = 'vgg16', input_shape = (224, 224, 3), weights = 'imagenet'):
	vgg_map = {
		'vgg16': VGG16,
		'vgg19': VGG19
	}

	if variant not in vgg_map:
		raise ValueError(f"Unsupported VGG variant '{variant}'.")

	return vgg_map[variant](
		include_top = False,
		weights = weights,
		input_shape = input_shape
	)		