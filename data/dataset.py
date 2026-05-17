"""Degradation pipeline for synthesizing degraded images."""

import cv2
import numpy as np
import torch
import random


class DegradationPipeline:
    """Applies random combinations of degradations to images."""
    
    def __init__(self, config):
        self.config = config
        self.degradation_types = config.DEGRADATION_TYPES
        min_severity = getattr(config, "MIN_SEVERITY", 0.2)
        self.severity_levels = np.linspace(min_severity, 1.0, config.SEVERITY_LEVELS)

    def _adjust_severity(self, deg_type_name, severity):
        boost = self.config.DEGRADATION_SEVERITY_BOOSTS.get(deg_type_name, 1.0)
        min_override = self.config.DEGRADATION_MIN_SEVERITY.get(deg_type_name)
        adjusted = min(1.0, severity * boost)
        if min_override is not None:
            adjusted = max(adjusted, min_override)
        if deg_type_name == "jpeg_compression":
            max_step = min(
                max(1, int(self.config.JPEG_MAX_SEVERITY_STEP)), len(self.severity_levels)
            )
            max_severity = float(self.severity_levels[max_step - 1])
            adjusted = min(adjusted, max_severity)
        return adjusted
    
    def apply_gaussian_noise(self, image, severity):
        """Apply Gaussian noise with given severity."""
        min_std, max_std = self.config.GAUSSIAN_NOISE_STD_RANGE
        noise_std = min_std + severity * (max_std - min_std)
        noise = np.random.normal(0.0, noise_std, image.shape)
        degraded = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return degraded, "gaussian_noise"
    
    def apply_motion_blur(self, image, severity):
        """Apply motion blur with given severity."""
        min_size, max_size = self.config.MOTION_BLUR_KERNEL_RANGE
        kernel_size = int(min_size + severity * (max_size - min_size))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        angle = random.uniform(*self.config.MOTION_BLUR_ANGLE_RANGE)
        kernel = self._get_motion_blur_kernel(kernel_size, angle)
        degraded = cv2.filter2D(image, -1, kernel)
        return degraded, "motion_blur"
    
    def apply_jpeg_compression(self, image, severity):
        """Apply JPEG compression with given severity."""
        min_q, max_q = self.config.JPEG_QUALITY_RANGE
        quality = int(max_q - severity * (max_q - min_q))
        quality = int(np.clip(quality, 1, 95))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
        if not success:
            return image, "jpeg_compression"
        decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        degraded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return degraded, "jpeg_compression"

    def apply_color_degradation(self, image, severity):
        """Apply color-related degradations such as white balance, saturation, and hue shifts."""
        image_float = image.astype(np.float32) / 255.0

        # White balance style channel shifts in RGB space.
        max_shift = self.config.COLOR_SHIFT_MAX * severity
        channel_gains = np.array([
            1.0 + random.uniform(-max_shift, max_shift),
            1.0 + random.uniform(-max_shift, max_shift),
            1.0 + random.uniform(-max_shift, max_shift),
        ], dtype=np.float32)
        image_float = np.clip(image_float * channel_gains.reshape(1, 1, 3), 0.0, 1.0)

        # Convert to HSV for saturation and hue changes.
        hsv = cv2.cvtColor((image_float * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = int(random.uniform(-self.config.HUE_SHIFT_MAX, self.config.HUE_SHIFT_MAX) * severity)
        saturation_scale = 1.0 - random.uniform(0.3, self.config.SATURATION_LOSS_MAX) * severity
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)

        degraded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return degraded, "color_degradation"

    def apply_exposure_issue(self, image, severity):
        """Apply exposure and contrast issues."""
        image_float = image.astype(np.float32) / 255.0

        brightness_delta = random.uniform(-self.config.BRIGHTNESS_MAX, self.config.BRIGHTNESS_MAX) * severity
        contrast_scale = 1.0 + random.uniform(-self.config.CONTRAST_MAX, self.config.CONTRAST_MAX) * severity
        gamma = random.uniform(*self.config.GAMMA_RANGE)

        degraded = image_float * contrast_scale + brightness_delta
        degraded = np.clip(degraded, 0.0, 1.0)

        degraded = np.power(np.clip(degraded, 0.0, 1.0), gamma)
        degraded = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
        return degraded, "exposure_issue"

    def apply_lens_distortion(self, image, severity):
        """Apply stronger radial lens distortion."""
        h, w = image.shape[:2]
        fx = 0.9 * w
        fy = 0.9 * h
        cx = w / 2.0
        cy = h / 2.0

        k1 = self.config.LENS_DISTORTION_K1_MAX * severity
        k2 = self.config.LENS_DISTORTION_K2_MAX * severity

        xs = (np.arange(w, dtype=np.float32) - cx) / fx
        ys = (np.arange(h, dtype=np.float32) - cy) / fy
        grid_x, grid_y = np.meshgrid(xs, ys)
        r2 = grid_x * grid_x + grid_y * grid_y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2

        map_x = ((grid_x * radial) * fx + cx).astype(np.float32)
        map_y = ((grid_y * radial) * fy + cy).astype(np.float32)

        warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return warped, "lens_distortion"

    def apply_downsample_upsample(self, image, severity):
        """Downsample then upsample to simulate resolution loss."""
        h, w = image.shape[:2]
        min_scale, max_scale = self.config.DOWNSAMPLE_SCALE_RANGE
        scale = max_scale - severity * (max_scale - min_scale)
        new_w = max(4, int(w * scale))
        new_h = max(4, int(h * scale))
        down = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
        return up, "downsample_upsample"

    def apply_defocus_blur(self, image, severity):
        """Apply defocus blur via Gaussian blur."""
        min_k, max_k = self.config.DEFOCUS_KERNEL_RANGE
        ksize = int(min_k + severity * (max_k - min_k))
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return blurred, "defocus_blur"
    
    def apply_degradation(self, image):
        """Apply random degradation to image.
        
        Returns:
            degraded_image: Degraded image array
            deg_type: Degradation type (0, 1, or 2)
            severity_level: Severity level index (0-4)
        """
        # Random degradation type and severity
        deg_type_idx = random.randint(0, len(self.degradation_types) - 1)
        severity_idx = random.randint(0, len(self.severity_levels) - 1)
        severity = self.severity_levels[severity_idx]
        
        deg_type_name = self.degradation_types[deg_type_idx]
        severity = self._adjust_severity(deg_type_name, severity)
        
        if deg_type_name == "gaussian_noise":
            degraded, _ = self.apply_gaussian_noise(image, severity)
        elif deg_type_name == "motion_blur":
            degraded, _ = self.apply_motion_blur(image, severity)
        elif deg_type_name == "jpeg_compression":
            degraded, _ = self.apply_jpeg_compression(image, severity)
        elif deg_type_name == "color_degradation":
            degraded, _ = self.apply_color_degradation(image, severity)
        elif deg_type_name == "exposure_issue":
            degraded, _ = self.apply_exposure_issue(image, severity)
        elif deg_type_name == "lens_distortion":
            degraded, _ = self.apply_lens_distortion(image, severity)
        elif deg_type_name == "downsample_upsample":
            degraded, _ = self.apply_downsample_upsample(image, severity)
        elif deg_type_name == "defocus_blur":
            degraded, _ = self.apply_defocus_blur(image, severity)
        else:
            raise ValueError(f"Unknown degradation type: {deg_type_name}")
        
        return degraded, deg_type_idx, severity_idx

    def apply_named_degradation(self, image, deg_type_name, severity):
        """Apply a specific degradation with adjusted severity."""
        severity = self._adjust_severity(deg_type_name, severity)

        if deg_type_name == "gaussian_noise":
            degraded, _ = self.apply_gaussian_noise(image, severity)
        elif deg_type_name == "motion_blur":
            degraded, _ = self.apply_motion_blur(image, severity)
        elif deg_type_name == "jpeg_compression":
            degraded, _ = self.apply_jpeg_compression(image, severity)
        elif deg_type_name == "color_degradation":
            degraded, _ = self.apply_color_degradation(image, severity)
        elif deg_type_name == "exposure_issue":
            degraded, _ = self.apply_exposure_issue(image, severity)
        elif deg_type_name == "lens_distortion":
            degraded, _ = self.apply_lens_distortion(image, severity)
        elif deg_type_name == "downsample_upsample":
            degraded, _ = self.apply_downsample_upsample(image, severity)
        elif deg_type_name == "defocus_blur":
            degraded, _ = self.apply_defocus_blur(image, severity)
        else:
            raise ValueError(f"Unknown degradation type: {deg_type_name}")

        return degraded
    
    @staticmethod
    def _get_motion_blur_kernel(size, angle):
        """Generate motion blur kernel."""
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0
        rotation_matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        return kernel


class ImageRestorationDataset(torch.utils.data.Dataset):
    """Dataset for image restoration with synthetic degradation."""
    
    def __init__(self, image_paths, config, is_train=True):
        """
        Args:
            image_paths: List of paths to clean images
            config: Configuration object
            is_train: Whether this is training or validation set
        """
        self.image_paths = image_paths
        self.config = config
        self.is_train = is_train
        self.degradation_pipeline = DegradationPipeline(config)
        self.patch_size = config.PATCH_SIZE
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        clean_image = cv2.imread(img_path)
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
        
        # Random crop during training
        if self.is_train:
            h, w = clean_image.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                y = random.randint(0, h - self.patch_size)
                x = random.randint(0, w - self.patch_size)
                clean_image = clean_image[y:y+self.patch_size, x:x+self.patch_size]
            else:
                # Resize if too small
                clean_image = cv2.resize(clean_image, (self.patch_size, self.patch_size))
        else:
            # Center crop or resize for validation
            h, w = clean_image.shape[:2]
            size = min(h, w, self.patch_size)
            y = (h - size) // 2
            x = (w - size) // 2
            clean_image = clean_image[y:y+size, x:x+size]
        
        # Apply degradation
        degraded_image, deg_type, severity_level = self.degradation_pipeline.apply_degradation(clean_image)
        
        # Convert to tensors [0, 1] range
        clean_tensor = torch.from_numpy(clean_image.astype(np.float32) / 255.0).permute(2, 0, 1)
        degraded_tensor = torch.from_numpy(degraded_image.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        return {
            'degraded': degraded_tensor,
            'clean': clean_tensor,
            'deg_type': torch.tensor(deg_type, dtype=torch.long),
            'severity': torch.tensor(severity_level, dtype=torch.long),
        }


def create_div2k_dataset(dataset_path, config, is_train=True):
    """Create dataset from DIV2K images.
    
    Expected structure:
    - dataset_path/DIV2K_train_HR/*.png (800 images)
    - dataset_path/DIV2K_valid_HR/*.png (100 images)
    """
    import os
    from glob import glob
    
    if is_train:
        image_dir = os.path.join(dataset_path, "DIV2K_train_HR")
        expected_count = config.TRAIN_IMAGES_COUNT
    else:
        image_dir = os.path.join(dataset_path, "DIV2K_valid_HR")
        expected_count = config.VAL_IMAGES_COUNT
    
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))

    # If DIV2K structure not found, fall back to generic recursive image collection from dataset_path
    if not image_paths:
        print(f"No DIV2K folder found at {image_dir}, falling back to recursive image search in {dataset_path}.")
        # Collect common image extensions recursively
        patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp"]
        all_paths = []
        for pat in patterns:
            all_paths.extend(glob(os.path.join(dataset_path, pat), recursive=True))

        all_paths = sorted(list(set(all_paths)))
        print(f"Found {len(all_paths)} total images under {dataset_path}")
        if not all_paths:
            raise FileNotFoundError(f"No images found in {dataset_path}. Provide a valid dataset folder.")

        # Shuffle for random split
        random.shuffle(all_paths)

        # Determine split counts
        # If count is 0 or >= dataset size, use all images.
        if is_train:
            count = config.TRAIN_IMAGES_COUNT
        else:
            count = config.VAL_IMAGES_COUNT

        if count is None:
            split_idx = int(len(all_paths) * 0.9)
            selected = all_paths[:split_idx] if is_train else all_paths[split_idx:]
        elif count <= 0 or count >= len(all_paths):
            selected = all_paths
        else:
            selected = all_paths[:count] if is_train else all_paths[-count:]

        print(f"Using {len(selected)} images from {dataset_path} for {'train' if is_train else 'val'} set")
        return ImageRestorationDataset(selected, config, is_train=is_train)

    print(f"Found {len(image_paths)} images in {image_dir} (expected ~{expected_count})")
    return ImageRestorationDataset(image_paths, config, is_train=is_train)


if __name__ == "__main__":
    from config import Config
    import matplotlib.pyplot as plt
    
    # Test degradation pipeline
    config = Config()
    pipeline = DegradationPipeline(config)
    
    # Create a test image
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Apply degradations
    for i in range(3):
        degraded, deg_type, severity = pipeline.apply_degradation(test_image)
        print(f"Degradation type: {config.DEGRADATION_TYPES[deg_type]}, Severity level: {severity}")
