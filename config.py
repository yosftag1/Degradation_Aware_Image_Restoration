"""Configuration for degradation-aware image restoration model."""

class Config:
    """Main configuration class."""
    
    # Model architecture
    MODEL_NAME = "degradation_aware_restoration"
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 3
    BASE_CHANNELS = 32
    NUM_BLOCKS = 4  # Legacy lightweight setting
    ATTENTION_REDUCTION = 16

    # NAFNet backbone (stronger model)
    NAFNET_WIDTH = 48
    NAFNET_ENC_BLOCKS = [2, 2, 2, 4]
    NAFNET_DEC_BLOCKS = [2, 2, 2, 2]
    NAFNET_MIDDLE_BLOCKS = 8
    
    # Degradation estimator
    DEGRADATION_TYPES = [
        "gaussian_noise",
        "motion_blur",
        "jpeg_compression",
        "color_degradation",
        "exposure_issue",
        "lens_distortion",
        "downsample_upsample",
        "defocus_blur",
    ]
    DEGRADATION_EMBEDDING_DIM = 16
    SEVERITY_LEVELS = 5
    MIN_SEVERITY = 0.4
    
    # Data
    DATASET_NAME = "DIV2K"
    DATASET_PATH = "./data/DIV2K"
    TRAIN_IMAGES_COUNT = 0  # 0 means use all available images
    VAL_IMAGES_COUNT = 0    # 0 means use all available images
    OUTPUT_DIR = "./outputs"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"
    VAL_SAMPLE_INTERVAL = 1  # Save sample images every N epochs (0 to disable)
    VAL_SAMPLE_COUNT = 10      # Number of validation samples to save
    
    # Image processing
    PATCH_SIZE = 512
    BATCH_SIZE = 2
    NUM_WORKERS = 4
    
    # Degradation pipeline
    GAUSSIAN_NOISE_STD_RANGE = (10.0, 70.0)
    MOTION_BLUR_KERNEL_RANGE = (7, 35)
    MOTION_BLUR_ANGLE_RANGE = (0, 180)
    JPEG_QUALITY_RANGE = (3, 45)
    JPEG_MAX_SEVERITY_STEP = 3
    COLOR_SHIFT_MAX = 0.8
    SATURATION_LOSS_MAX = 0.9
    HUE_SHIFT_MAX = 45
    BRIGHTNESS_MAX = 0.7
    CONTRAST_MAX = 0.9
    GAMMA_RANGE = (0.4, 2.4)
    DOWNSAMPLE_SCALE_RANGE = (0.2, 0.95)
    DEFOCUS_KERNEL_RANGE = (3, 21)
    LENS_DISTORTION_K1_MAX = 0.7
    LENS_DISTORTION_K2_MAX = 0.3

    DEGRADATION_SEVERITY_BOOSTS = {
        "gaussian_noise": 1.8,
        "motion_blur": 1.4,
        "jpeg_compression": 1.0,
        "color_degradation": 1.5,
        "exposure_issue": 1.0,
        "lens_distortion": 1.6,
        "downsample_upsample": 1.5,
        "defocus_blur": 1.4,
    }
    DEGRADATION_MIN_SEVERITY = {
        "gaussian_noise": 0.7,
        "motion_blur": 0.6,
        "color_degradation": 0.65,
        "lens_distortion": 0.6,
        "downsample_upsample": 0.3,
        "defocus_blur": 0.3,
    }
    
    # Training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    WARMUP_EPOCHS = 5
    CHECKPOINT_EVERY_N_EPOCHS = 5
    DEVICE = "cuda"
    
    # Loss weights
    LOSS_L1_WEIGHT = 1.0
    LOSS_PERCEPTUAL_WEIGHT = 0.1
    LOSS_FREQUENCY_WEIGHT = 0.05
    
    # Perceptual loss
    PERCEPTUAL_LAYERS = ["relu3_4"]  # VGG layer for perceptual loss
    
    # Logging
    LOG_INTERVAL = 100
    VAL_INTERVAL = 500
    VAL_EPOCH_INTERVAL = 1
    SEED = 42

    # Evaluation image saving
    EVAL_IMAGE_COUNT = 12
    EVAL_SEVERITY_STEPS = 3
    

class DebugConfig(Config):
    """Debug configuration with smaller dataset and faster training."""
    EPOCHS = 3
    BATCH_SIZE = 1
    TRAIN_IMAGES_COUNT = 10
    VAL_IMAGES_COUNT = 2
    LOG_INTERVAL = 5
    VAL_INTERVAL = 20
