import tensorflow as tf
from Configs.config import cfg
from Backbones import backbone 
from tensorflow.keras import layers
from ROI.roi_pooling_layer import roi_pooling
from Region_Proposals.region_proposal_network import RPN


class SOR(tf.keras.Model):
	def __init__(self, is_training):
		super().__init__()
		self.is_training = is_training	
		self.num_classes = cfg.num_classes

		# Initialize the backbone
		self.model_backbone = backbone.get_backbone('config_1')
		
		# Initializing the layers
		self.rpn = RPN(self.is_training)
		self.roi_pooling = roi_pooling()


	def call(self, inputs):

		# Extracting feature map
		feature_map = self.model_backbone(inputs)

		img_size = tf.constant([224, 224], dtype=tf.float32)

		# Getting region proposals
		rois = self.rpn(feature_map, img_size)

		# ROI pooling
		crops = self.roi_pooling(feature_map, rois, img_size)

		return crops



inputs = tf.ones((8, 224, 224, 3)) 
model = SOR(True)(inputs)
model.summary()