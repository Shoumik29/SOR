from omegaconf import OmegaConf
from .resnet import build_resnet
from .vgg import build_vgg




def load_config(path = 'Configs/model_config.yaml'):
	return OmegaConf.load(path)


def get_backbone(config = 'config_1'):
	# Loading model configuration
	cfg_yaml = load_config()

	if config not in cfg_yaml:
		raise KeyError(f"Config key '{config}' not found in Configs/model_config.yaml")

	cfg = cfg_yaml[config]
	cfg.backbone_name = cfg.backbone_name.lower()	

	if cfg.backbone_name.startswith("resnet"):
		backbone = build_resnet(
			variant = cfg.backbone_name,
			input_shape = cfg.input_shape
		)
	elif cfg.backbone_name.startswith("vgg"):
		backbone = build_vgg(
			variant = cfg.backbone_name,
			input_shape = cfg.input_shape
		)
	else:
		raise ValueError(f"Backbone '{cfg.backbone_name}' not implemented.")	

	# Freeze the backbone if trainable is false
	if not cfg.trainable:
		for layer in backbone.layers:
			layer.trainable = False

	return backbone