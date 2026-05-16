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
    PATCH_SIZE = 256
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    
    # Degradation pipeline
    GAUSSIAN_NOISE_RANGE = (0.02, 0.18)
    MOTION_BLUR_KERNEL_RANGE = (3, 25)
    MOTION_BLUR_ANGLE_RANGE = (0, 180)
    JPEG_QUALITY_RANGE = (5, 60)
    COLOR_SHIFT_RANGE = (0.0, 0.4)
    SATURATION_LOSS_RANGE = (0.0, 0.7)
    HUE_SHIFT_RANGE = (0, 30)
    BRIGHTNESS_RANGE = (0.0, 0.5)
    CONTRAST_RANGE = (0.3, 1.8)
    GAMMA_RANGE = (0.5, 2.0)
    
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
    

class DebugConfig(Config):
    """Debug configuration with smaller dataset and faster training."""
    EPOCHS = 3
    BATCH_SIZE = 2
    TRAIN_IMAGES_COUNT = 10
    VAL_IMAGES_COUNT = 2
    LOG_INTERVAL = 5
    VAL_INTERVAL = 20
