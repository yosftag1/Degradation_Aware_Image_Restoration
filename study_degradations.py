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


def apply_color_degradation(image, severity, max_shift=0.8, max_hue_shift=45, max_sat_loss=0.9):
    """Apply stronger color degradation for visibility."""
    image_float = image.astype(np.float32) / 255.0

    shift = max_shift * severity
    channel_gains = np.array(
        [
            1.0 + random.uniform(-shift, shift),
            1.0 + random.uniform(-shift, shift),
            1.0 + random.uniform(-shift, shift),
        ],
        dtype=np.float32,
    )
    image_float = np.clip(image_float * channel_gains.reshape(1, 1, 3), 0.0, 1.0)

    hsv = cv2.cvtColor((image_float * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_shift = int(random.uniform(-max_hue_shift, max_hue_shift) * severity)
    sat_scale = 1.0 - random.uniform(0.3, max_sat_loss) * severity
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)

    degraded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return degraded


def apply_exposure_issue(image, severity, max_brightness=0.7, max_contrast=0.9, gamma_range=(0.4, 2.4)):
    """Apply exposure and contrast issues."""
    image_float = image.astype(np.float32) / 255.0

    brightness_delta = random.uniform(-max_brightness, max_brightness) * severity
    contrast_scale = 1.0 + random.uniform(-max_contrast, max_contrast) * severity
    gamma = random.uniform(gamma_range[0], gamma_range[1])

    degraded = image_float * contrast_scale + brightness_delta
    degraded = np.clip(degraded, 0.0, 1.0)
    degraded = np.power(np.clip(degraded, 0.0, 1.0), gamma)
    degraded = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    return degraded


def apply_defocus_blur(image, severity, min_ksize=3, max_ksize=21):
    """Apply stronger Gaussian defocus blur."""
    ksize = int(min_ksize + severity * (max_ksize - min_ksize))
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred


def apply_down_up_sample(image, severity, min_scale=0.2, max_scale=0.95):
    """Downsample then upsample to simulate resolution loss."""
    h, w = image.shape[:2]
    scale = max_scale - severity * (max_scale - min_scale)
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))
    down = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
    return up


def apply_lens_distortion(image, severity, k1_max=0.7, k2_max=0.3):
    """Apply stronger radial lens distortion via a custom remap."""
    h, w = image.shape[:2]
    fx = 0.9 * w
    fy = 0.9 * h
    cx = w / 2.0
    cy = h / 2.0

    k1 = k1_max * severity
    k2 = k2_max * severity

    xs = (np.arange(w, dtype=np.float32) - cx) / fx
    ys = (np.arange(h, dtype=np.float32) - cy) / fy
    grid_x, grid_y = np.meshgrid(xs, ys)
    r2 = grid_x * grid_x + grid_y * grid_y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2

    map_x = (grid_x * radial) * fx + cx
    map_y = (grid_y * radial) * fy + cy

    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def save_image(path, image_rgb):
    """Save RGB image to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def main():
    parser = argparse.ArgumentParser(description="Generate degradation study images.")
    parser.add_argument("--image", default="./test.jpg", help="Path to input image")
    parser.add_argument("--output-dir", default="./outputs/degradation_study", help="Output directory")
    parser.add_argument("--steps", type=int, default=7, help="Number of severity steps")
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

    # Existing degradations
    existing = {
        "gaussian_noise": apply_gaussian_noise,
        "motion_blur": apply_motion_blur,
        "jpeg_compression": apply_jpeg_compression,
        "color_degradation": apply_color_degradation,
        "exposure_issue": apply_exposure_issue,
    }

    severity_boosts = {
        "gaussian_noise": (1.8, 0.7),
        "motion_blur": (1.4, 0.6),
        "jpeg_compression": (1.0, None),
        "color_degradation": (1.5, 0.65),
        "exposure_issue": (1.0, None),
        "lens_distortion": (1.6, 0.6),
        "downsample_upsample": (1.3, 0.3),
        "defocus_blur": (1.2, 0.3),
    }

    max_severity_caps = {
        "jpeg_compression": float(severities[min(2, len(severities) - 1)]),
    }

    def boost_severity(base_sev, boost, min_override, max_override):
        sev = min(1.0, base_sev * boost)
        if min_override is not None:
            sev = max(sev, min_override)
        if max_override is not None:
            sev = min(sev, max_override)
        return sev

    for name, func in existing.items():
        boost, min_override = severity_boosts.get(name, (1.0, None))
        max_override = max_severity_caps.get(name)
        for idx, sev in enumerate(severities):
            adjusted = boost_severity(float(sev), boost, min_override, max_override)
            degraded = func(image_rgb, adjusted)
            out_path = os.path.join(args.output_dir, name, f"step_{idx + 1:02d}.png")
            save_image(out_path, degraded)

    # New degradations for study
    new_degradations = {
        "lens_distortion": apply_lens_distortion,
        "downsample_upsample": apply_down_up_sample,
        "defocus_blur": apply_defocus_blur,
    }

    for name, func in new_degradations.items():
        boost, min_override = severity_boosts.get(name, (1.0, None))
        max_override = max_severity_caps.get(name)
        for idx, sev in enumerate(severities):
            adjusted = boost_severity(float(sev), boost, min_override, max_override)
            degraded = func(image_rgb, adjusted)
            out_path = os.path.join(args.output_dir, name, f"step_{idx + 1:02d}.png")
            save_image(out_path, degraded)

    print(f"Saved study images to: {args.output_dir}")


if __name__ == "__main__":
    main()
