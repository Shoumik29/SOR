import tensorflow as tf
from tensorflow.keras import layers, Model, applications
from Utils.Custom_Layers import ProposalFeatureExtractor, DetectionBranch, SORBranch
from Utils.Custom_Losses import total_loss



class SORModel(Model):
    """Full SOR model with position-preserved attention"""
    def __init__(self, backbone_name='resnet50', num_classes=80):
        super(SORModel, self).__init__()
        
        # Initialize backbone
        if backbone_name == 'resnet50':
            self.backbone = applications.ResNet50(include_top=False, weights='imagenet')
        elif backbone_name == 'resnet101':
            self.backbone = applications.ResNet101(include_top=False, weights='imagenet')
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Initialize other components
        self.position_encoder = layers.Lambda(
            lambda x: tf.concat([
                x,
                tf.tile(
                    tf.reshape(
                        tf.linspace(0.0, 1.0, tf.shape(x)[1]),
                        [1, tf.shape(x)[1], 1, 1]
                    ),
                    [tf.shape(x)[0], 1, tf.shape(x)[2], 1]
                ),  # X-axis position
                tf.tile(
                    tf.reshape(
                        tf.linspace(0.0, 1.0, tf.shape(x)[2]),
                        [1, 1, tf.shape(x)[2], 1]
                    ),
                    [tf.shape(x)[0], tf.shape(x)[1], 1, 1]
                )   # Y-axis position
            ], axis=-1)
        )
        
        self.proposal_feature_extractor = ProposalFeatureExtractor()
        self.detection_branch = DetectionBranch(num_classes)
        self.sor_branch = SORBranch()
        
    def call(self, inputs, training=False):
        image, proposal_boxes = inputs
        
        # Extract features using backbone
        feature_map = self.backbone(image, training=training)
        
        # Add positional information to feature map
        feature_pos_map = self.position_encoder(feature_map)
        
        # Extract proposal features
        position_map = feature_pos_map[:, :, :, -2:]  # Get positional channels
        feature_map_only = feature_pos_map[:, :, :, :-2]  # Get feature channels
        proposal_features = self.proposal_feature_extractor(feature_map_only, position_map, proposal_boxes)
        
        # Detection branch (without positional information)
        detection_features = proposal_features[:, :, :, :-2]  # Remove position channels
        detection_outputs = self.detection_branch(detection_features)
        
        # SOR branch (with positional information)
        ranking_scores = self.sor_branch(proposal_features, training)
        
        return {
            'detection': detection_outputs,
            'ranking': ranking_scores
        }



# Example usage:
def build_model(input_shape=(224, 224, 3), num_classes=80):
    """Build the SOR model with PPA module"""
    # Input placeholders
    image_input = tf.keras.Input(shape=input_shape)
    proposal_boxes_input = tf.keras.Input(shape=(None, 4))  # [y1, x1, y2, x2] format
    
    # Create model
    model = SORModel(backbone_name='resnet50', num_classes=num_classes)
    
    # Compile model
    model.compile(optimizer='adam', loss=total_loss)
    
    return model



model = build_model()
model.summary()