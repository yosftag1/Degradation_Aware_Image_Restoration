"""Generate degradation study images for a single input."""

import argparse
import os
import random

import cv2
import numpy as np

from config import Config


def apply_gaussian_noise(image, severity, min_std=10.0, max_std=70.0):
    """Apply stronger Gaussian noise in pixel space."""
    noise_std = min_std + severity * (max_std - min_std)
    noise = np.random.normal(0.0, noise_std, image.shape)
    degraded = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return degraded


def _motion_blur_kernel(size, angle):
    """Create a directional motion blur kernel."""
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    rotation_matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def apply_motion_blur(image, severity, min_ksize=7, max_ksize=35):
    """Apply stronger directional motion blur."""
    ksize = int(min_ksize + severity * (max_ksize - min_ksize))
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    angle = random.uniform(0, 180)
    kernel = _motion_blur_kernel(ksize, angle)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def apply_jpeg_compression(image, severity, min_quality=3, max_quality=45):
    """Apply more aggressive JPEG compression using OpenCV encode/decode."""
    quality = int(max_quality - severity * (max_quality - min_quality))
    quality = int(np.clip(quality, 1, 95))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
    if not success:
        return image
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def apply_color_degradation(image, severity, max_shift=0.35, max_hue_shift=10, max_sat_loss=0.55):
    """Apply physical white-balance illuminant cast and dye saturation fading."""
    image_float = image.astype(np.float32) / 255.0

    shift = max_shift * severity
    channel_gains = np.array(
        [
            1.0 + random.uniform(-shift, shift),
            1.0 + random.uniform(-shift * 0.6, shift * 0.6),
            1.0 + random.uniform(-shift, shift),
        ],
        dtype=np.float32,
    )
    channel_gains /= np.mean(channel_gains)
    image_float = np.clip(image_float * channel_gains.reshape(1, 1, 3), 0.0, 1.0)

    hsv = cv2.cvtColor((image_float * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_shift = int(random.uniform(-max_hue_shift, max_hue_shift) * severity)
    sat_scale = 1.0 - random.uniform(0.15, max_sat_loss) * severity
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)

    degraded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return degraded


def apply_exposure_issue(image, severity, max_brightness=0.25, max_contrast=0.35, gamma_range=(0.55, 1.85)):
    """Apply physical exposure and gamma curve shifts scaled by severity without hard clipping."""
    image_float = image.astype(np.float32) / 255.0

    min_g, max_g = gamma_range
    if random.random() < 0.5:
        gamma = 1.0 + severity * (max_g - 1.0)
        brightness_delta = -random.uniform(0.02, max_brightness) * severity
    else:
        gamma = 1.0 - severity * (1.0 - min_g)
        brightness_delta = random.uniform(0.02, max_brightness) * severity

    contrast_scale = 1.0 - random.uniform(0.05, max_contrast) * severity
    degraded = (image_float - 0.5) * contrast_scale + 0.5 + brightness_delta
    degraded = np.clip(degraded, 1e-4, 1.0)
    degraded = np.power(degraded, gamma)
    degraded = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    return degraded


def apply_defocus_blur(image, severity, min_ksize=3, max_ksize=17):
    """Apply Gaussian defocus blur."""
    ksize = int(min_ksize + severity * (max_ksize - min_ksize))
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred


def apply_down_up_sample(image, severity, min_scale=0.25, max_scale=0.75):
    """Downsample then upsample to simulate resolution loss."""
    h, w = image.shape[:2]
    scale = max_scale - severity * (max_scale - min_scale)
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))
    down = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
    return up


def apply_atmospheric_haze(image, severity, beta_range=(0.6, 2.2), airlight_range=(0.75, 0.98)):
    """Apply Koschmieder physical atmospheric scattering haze: I(x) = J(x)*t(x) + A*(1 - t(x))."""
    h, w = image.shape[:2]
    image_float = image.astype(np.float32) / 255.0

    min_beta, max_beta = beta_range
    min_A, max_A = airlight_range
    beta = min_beta + severity * (max_beta - min_beta)

    y_grad = np.linspace(0.95, 0.30, h, dtype=np.float32)[:, None]
    x_freq = random.uniform(1.0, 2.5)
    x_phase = random.uniform(0.0, 2.0 * np.pi)
    x_wave = 0.12 * np.sin(np.linspace(0.0, x_freq * np.pi, w, dtype=np.float32) + x_phase)[None, :]
    depth = np.clip(y_grad + x_wave, 0.2, 1.0)[:, :, None]

    transmission = np.exp(-beta * depth)
    base_A = random.uniform(min_A, max_A)
    airlight = np.array(
        [
            np.clip(base_A + random.uniform(-0.04, 0.04), 0.7, 1.0),
            np.clip(base_A + random.uniform(-0.03, 0.03), 0.7, 1.0),
            np.clip(base_A + random.uniform(-0.04, 0.05), 0.7, 1.0),
        ],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    hazy = image_float * transmission + airlight * (1.0 - transmission)
    return np.clip(hazy * 255.0, 0, 255).astype(np.uint8)


def save_image(path, image_rgb):
    """Save RGB image to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def main():
    parser = argparse.ArgumentParser(description="Generate degradation study images.")
    parser.add_argument("--image", default="./test.jpg", help="Path to input image")
    parser.add_argument("--output-dir", default="./outputs/degradation_study", help="Output directory")
    parser.add_argument("--steps", type=int, default=5, help="Number of severity steps")
    parser.add_argument("--min-severity", type=float, default=None, help="Override minimum severity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    config = Config()
    min_severity = args.min_severity if args.min_severity is not None else config.MIN_SEVERITY
    severities = np.linspace(min_severity, 1.0, args.steps)

    image_bgr = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    degradations = {
        "gaussian_noise": lambda img, s: apply_gaussian_noise(img, s, *config.GAUSSIAN_NOISE_STD_RANGE),
        "motion_blur": lambda img, s: apply_motion_blur(img, s, *config.MOTION_BLUR_KERNEL_RANGE),
        "jpeg_compression": lambda img, s: apply_jpeg_compression(img, s, *config.JPEG_QUALITY_RANGE),
        "color_degradation": apply_color_degradation,
        "exposure_issue": apply_exposure_issue,
        "atmospheric_haze": apply_atmospheric_haze,
        "downsample_upsample": apply_down_up_sample,
        "defocus_blur": apply_defocus_blur,
    }

    for name, func in degradations.items():
        for idx, sev in enumerate(severities):
            degraded = func(image_rgb, float(sev))
            out_path = os.path.join(args.output_dir, name, f"step_{idx + 1:02d}.png")
            save_image(out_path, degraded)

    print(f"Saved study images to: {args.output_dir}")


if __name__ == "__main__":
    main()
