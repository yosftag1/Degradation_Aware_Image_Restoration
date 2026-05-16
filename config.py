"""Configuration for degradation-aware image restoration model."""

class Config:
    """Main configuration class."""
    
    # Model architecture
    MODEL_NAME = "degradation_aware_restoration"
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 3
    BASE_CHANNELS = 32
    NUM_BLOCKS = 4  # Lightweight: 4 blocks instead of full NAFNet
    ATTENTION_REDUCTION = 16
    
    # Degradation estimator
    DEGRADATION_TYPES = [
        "gaussian_noise",
        "motion_blur",
        "jpeg_compression",
        "color_degradation",
        "exposure_issue",
    ]
    DEGRADATION_EMBEDDING_DIM = 16
    SEVERITY_LEVELS = 5  # 0.2, 0.4, 0.6, 0.8, 1.0
    
    # Data
    DATASET_NAME = "DIV2K"
    DATASET_PATH = "./data/DIV2K"
    TRAIN_IMAGES_COUNT = 0  # 0 means use all available images
    VAL_IMAGES_COUNT = 0    # 0 means use all available images
    OUTPUT_DIR = "./outputs"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"
    VAL_SAMPLE_INTERVAL = 1  # Save sample images every N epochs (0 to disable)
    VAL_SAMPLE_COUNT = 5      # Number of validation samples to save
    
    # Image processing
    PATCH_SIZE = 256
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    
    # Degradation pipeline
    GAUSSIAN_NOISE_RANGE = (0.01, 0.1)
    MOTION_BLUR_KERNEL_RANGE = (3, 15)
    MOTION_BLUR_ANGLE_RANGE = (0, 180)
    JPEG_QUALITY_RANGE = (10, 80)
    COLOR_SHIFT_RANGE = (0.0, 0.25)
    SATURATION_LOSS_RANGE = (0.0, 0.5)
    HUE_SHIFT_RANGE = (0, 20)
    BRIGHTNESS_RANGE = (0.0, 0.35)
    CONTRAST_RANGE = (0.5, 1.5)
    GAMMA_RANGE = (0.7, 1.5)
    
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
    SEED = 42
    

class DebugConfig(Config):
    """Debug configuration with smaller dataset and faster training."""
    EPOCHS = 3
    BATCH_SIZE = 2
    TRAIN_IMAGES_COUNT = 10
    VAL_IMAGES_COUNT = 2
    LOG_INTERVAL = 5
    VAL_INTERVAL = 20
