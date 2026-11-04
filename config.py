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
        "color_banding",
        "downsample_upsample",
        "defocus_blur",
    ]
    DEGRADATION_EMBEDDING_DIM = 32
    SEVERITY_LEVELS = 5
    MIN_SEVERITY = 0.2
    MAX_COMPOUND_DEGRADATIONS = 3  # Chain 1 to 3 degradations in physical camera order
    COMPOUND_PROB = 0.6            # 60% of training samples combine 2-3 degradations, 40% single

    # Physics / Stochastic Bridge settings (I2SB, Brownian Bridge, Flow Matching)
    BRIDGE_TYPE = "i2sb"          # Options: "i2sb", "brownian", "flow"
    BRIDGE_STEPS = 4              # Number of reverse bridge steps during inference (t: 1 -> 0)
    BRIDGE_SIGMA_MAX = 0.05       # Peak noise bubble std (0.05 avoids grainy noise in 4-step pixel walks)
    BRIDGE_OT_ODE = False         # If True, use deterministic OT-ODE sampler at inference
    TIME_EMBEDDING_DIM = 64       # Sinusoidal time embedding dimension for bridge network
    CONCAT_DEGRADED_INPUT = False # Pure x_t input to bridge backbone (prevents linear (x_t, x_1) shortcut)
    
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
    BATCH_SIZE = 16             # 8 per T4 GPU on Kaggle T4 x2 (auto-scales if single GPU)
    NUM_WORKERS = 4
    USE_AMP = True              # Mixed precision (FP16) for T4 Tensor Cores
    MULTI_GPU = True            # Use all available GPUs via DataParallel (e.g. Kaggle T4 x2)
    
    # Degradation pipeline (physically recoverable ranges across severity in [0.2, 1.0])
    GAUSSIAN_NOISE_STD_RANGE = (8.0, 50.0)
    MOTION_BLUR_KERNEL_RANGE = (5, 27)
    MOTION_BLUR_ANGLE_RANGE = (0, 180)
    JPEG_QUALITY_RANGE = (10, 50)
    JPEG_MAX_SEVERITY_STEP = 5
    COLOR_SHIFT_MAX = 0.35      # Physical white-balance / illuminant channel cast
    SATURATION_LOSS_MAX = 0.55  # Dye fading without destroying chroma identity
    HUE_SHIFT_MAX = 10          # Subtle tint drift (avoids unidentifiable hue roulette)
    BRIGHTNESS_MAX = 0.25       # Avoids hard-clipping entire regions to 0 or 255
    CONTRAST_MAX = 0.35
    GAMMA_RANGE = (0.55, 1.85)  # Scaled by severity in dataset.py
    DOWNSAMPLE_SCALE_RANGE = (0.25, 0.75)
    DEFOCUS_KERNEL_RANGE = (3, 17)
    BANDING_LEVELS_RANGE = (6, 26)      # Quantization levels (26=mild contour steps, 6=severe banding without neon hue shifts)

    # Keep boosts at 1.0 so all 5 severity levels [0.2, 0.4, 0.6, 0.8, 1.0] stay distinct and learnable
    DEGRADATION_SEVERITY_BOOSTS = {
        "gaussian_noise": 1.0,
        "motion_blur": 1.0,
        "jpeg_compression": 1.0,
        "color_degradation": 1.0,
        "exposure_issue": 1.0,
        "color_banding": 1.0,
        "downsample_upsample": 1.0,
        "defocus_blur": 1.0,
    }
    DEGRADATION_MIN_SEVERITY = {}
    
    # Training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    WARMUP_EPOCHS = 5
    CHECKPOINT_EVERY_N_EPOCHS = 5
    DEVICE = "cuda"
    
    # Loss weights
    LOSS_L1_WEIGHT = 1.0
    LOSS_GRADIENT_WEIGHT = 0.15
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
    EVAL_IMAGE_COUNT = 4
    EVAL_SEVERITY_STEPS = 3
    

class DebugConfig(Config):
    """Debug configuration with smaller dataset and faster training."""
    EPOCHS = 3
    BATCH_SIZE = 1
    TRAIN_IMAGES_COUNT = 10
    VAL_IMAGES_COUNT = 2
    LOG_INTERVAL = 5
    VAL_INTERVAL = 20
