import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152

def build_resnet(variant = 'resnet50', input_shape = (224, 224, 3), weights = 'imagenet'):
	resnet_map = {
		'resnet50': ResNet50,
		'resnet101': ResNet101,
		'resnet152': ResNet152,
	}

	if variant not in resnet_map:
		raise ValueError(f"Unsupported ResNet variant '{variant}'.")

	return resnet_map[variant](
		include_top = False,
		weights = weights,
		input_shape = input_shape
	)		