# Got help from https://github.com/hxuaj/tf2-faster-rcnn/blob/main/config/config.py

import numpy as np


class Config:
    """
    Construct a class to hold default configs for dataset, model, training and testing.
    """
    
    

    # ------------------------- Data Related ------------------------- #
    # The number of classes in the dataset
    num_classes = 21 



    # -------------------------     NMS     ------------------------- #
    # Number of top-score boxes before NMS when training
    train_pre_nms_top_n = 12000
    
    # Number of top-score boxes after NMS when training
    train_post_nms_top_n = 2000
    
    # Number of top-score boxes before NMS when testing
    test_pre_nms_top_n = 6000
    
    # Number of top-score boxes after NMS when testing
    test_post_nms_top_n = 300
    
    # NMS threshold applied to RPN proposals
    nms_thresh = 0.7



    # ------------------------- RPN Training ------------------------- #
    # Total feature stride from base net
    feat_stride = 16 # VGG16
    
    # Threshold for anchor_to_target 
    # (call them pos and neg to tell apart from roi training)
    pos_thresh = 0.7
    neg_thresh = 0.3
    
    # The fraction of positive samples
    pos_sample_ratio = 0.5
    
    # The total number of RPN training sample
    num_sample_rpn = 256
    


    # ------------------------- ROI Training ------------------------- #
    # Whether to include ground truth boxes into proposal sampling, incase no fg
    proposal_sample_use_gt = False
    
    # (call them fg and bg to tell apart from rpn training)
    # Foreground threshold
    fg_thresh = 0.5
    # Background threshold
    bg_thresh_hi = 0.5
    # the origin thresh is 0.1, change to 0 in case no bg and no fg sampled
    bg_thresh_lo = 0.
    
    # Number of sampling rois per image for training
    num_sample_rois = 128
    
    # Foreground rois ratio per image
    fg_sample_ratio = 0.25
    
    # Max foreground rois samples per image for training
    max_fg_rois = np.round(num_sample_rois * fg_sample_ratio)
    
    # Whether to normalize the roi_bbox_targets
    norm_bbox = True
    norm_mean = np.array([0., 0., 0., 0.], dtype=np.float32)
    norm_std = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)
    


    # ------------------------- ROI Pooling ------------------------- #
    # pool and resize to fix size
    pool_size = 7
    
    # If use max pooling after crop_and_resize
    if_max_pool = True
    


    


cfg = Config()