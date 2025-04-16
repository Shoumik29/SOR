import tensorflow as tf
from tensorflow.keras import layers



class MultiHeadSelfAttention(layers.Layer):
    """Multi-head Self-Attention layer"""
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embed_dim} should be divisible by number of heads {num_heads}")
        
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output
        
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        
        output = self.combine_heads(concat_attention)
        return output



class TransformerBlock(layers.Layer):
    """Transformer block with self-attention and feed-forward network"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



class PositionEmbeddingStage(layers.Layer):
    """Position Embedding Stage of PPA module"""
    def __init__(self, filters=64):
        super(PositionEmbeddingStage, self).__init__()
        self.pos_conv = layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')
        self.fusion_convs = [
            layers.Conv2D(256, kernel_size=3, padding='same', activation='relu') for _ in range(4)
        ]
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(2048, activation='relu')
        self.fc2 = layers.Dense(1024)
        
    def call(self, inputs):
        # Inputs shape: [batch_size, roi_height, roi_width, feat_dim+2]
        semantic_feat = inputs[:, :, :, :-2]  # Semantic part
        position_feat = inputs[:, :, :, -2:]  # Positional part
        
        # Extract low-level position features
        pos_feat = self.pos_conv(position_feat)
        
        # Concatenate original position and low-level position feature
        pos_embedding = tf.concat([position_feat, pos_feat], axis=-1)
        
        # Concatenate semantic feature and position embedding
        x = tf.concat([semantic_feat, pos_embedding], axis=-1)
        
        # Apply convolution layers
        for conv in self.fusion_convs:
            x = conv(x)
        
        # Transform to visual token
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x



class FeatureInteractionStage(layers.Layer):
    """Feature Interaction Stage of PPA module using Transformer Encoder"""
    def __init__(self, embed_dim=1024, num_heads=8, ff_dim=2048, num_layers=3):
        super(FeatureInteractionStage, self).__init__()
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) 
            for _ in range(num_layers)
        ]
    
    def call(self, inputs, training=False):
        x = inputs
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training)
        return x



class PPAModule(layers.Layer):
    """Position-Preserved Attention Module"""
    def __init__(self):
        super(PPAModule, self).__init__()
        self.position_embedding = PositionEmbeddingStage()
        self.feature_interaction = FeatureInteractionStage()
        
    def call(self, inputs, training=False):
        # Position embedding stage
        visual_tokens = self.position_embedding(inputs)
        
        # Visual tokens shape: [batch_size, num_proposals, 1024]
        visual_tokens = tf.expand_dims(visual_tokens, axis=0)
        
        # Feature interaction stage
        contextualized_representations = self.feature_interaction(visual_tokens, training)
        
        return contextualized_representations
    
    
    
class ProposalFeatureExtractor(layers.Layer):
    """Extract and process proposal features with position information"""
    def __init__(self, roi_size=14):
        super(ProposalFeatureExtractor, self).__init__()
        self.roi_size = roi_size
        
    def call(self, feature_map, position_map, proposal_boxes):
        # Concatenate feature map with position map
        combined_map = tf.concat([feature_map, position_map], axis=-1)
        
        # Apply ROI pooling for each proposal
        proposal_features = []
        
        # This is a simplified ROI pooling implementation
        # In a real implementation, you would use tf.image.crop_and_resize or a proper ROI pooling op
        for bbox in proposal_boxes:
            y1, x1, y2, x2 = bbox
            
            # Crop the region from combined_map
            roi = tf.image.crop_to_bounding_box(
                combined_map,
                tf.cast(y1, tf.int32),
                tf.cast(x1, tf.int32),
                tf.cast(y2 - y1, tf.int32),
                tf.cast(x2 - x1, tf.int32)
            )
            
            # Resize to fixed size
            roi_resized = tf.image.resize(roi, [self.roi_size, self.roi_size])
            proposal_features.append(roi_resized)
            
        return tf.stack(proposal_features, axis=0)



class SORBranch(layers.Layer):
    """Salient Object Ranking branch with PPA module"""
    def __init__(self):
        super(SORBranch, self).__init__()
        self.ppa_module = PPAModule()
        self.ranking_fc = layers.Dense(1)  # For ranking score prediction
        
    def call(self, features, training=False):
        # Features shape: [batch_size, roi_height, roi_width, channels+2]
        contextualized_features = self.ppa_module(features, training)
        
        # Predict ranking order
        ranking_scores = self.ranking_fc(contextualized_features)
        
        return ranking_scores
    
    
    
class DetectionBranch(layers.Layer):
    """Detection branch based on MaskRCNN-like architecture"""
    def __init__(self, num_classes):
        super(DetectionBranch, self).__init__()
        # This is a simplified version. In practice, you would implement
        # a full detection branch similar to MaskRCNN or other detectors
        self.bbox_fc = layers.Dense(4 * num_classes)  # Box regression
        self.cls_fc = layers.Dense(num_classes, activation='softmax')  # Classification
        self.mask_conv = layers.Conv2DTranspose(num_classes, kernel_size=2, strides=2, activation='sigmoid')  # Mask
        
    def call(self, features):
        # Features shape: [batch_size, roi_height, roi_width, channels]
        pooled_features = tf.reduce_mean(features, axis=[1, 2])  # Global average pooling
        
        # Box prediction
        bbox_pred = self.bbox_fc(pooled_features)
        
        # Class prediction
        cls_pred = self.cls_fc(pooled_features)
        
        # Mask prediction (simplified)
        mask_pred = self.mask_conv(features)
        
        return {
            'boxes': bbox_pred,
            'classes': cls_pred,
            'masks': mask_pred
        }