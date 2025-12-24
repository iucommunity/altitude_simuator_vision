#!/usr/bin/env python3
"""
Test script for altitude_estimation_v2.py using DroneCaptures1 dataset.

Usage:
    conda run -n py39 python test_drone_captures.py --folder DroneCaptures1
    conda run -n py39 python test_drone_captures.py --folder DroneCaptures1 --init-frames 5 --plot
"""

import argparse
import json
import logging
import time
try:
    import cv2
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "OpenCV (cv2) is not installed in this Python environment. "
        "Install it via one of:\n"
        "  - pip install opencv-python\n"
        "  - conda install -c conda-forge opencv\n"
    ) from e
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from altitude_estimation_v2 import (
    create_altimeter
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOG = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration for Unity Physical Camera."""
    sensor_width_mm: float = 36.0
    focal_length_mm: float = 20.78461
    
    @classmethod
    def unity_default(cls) -> 'CameraConfig':
        """Default Unity Physical Camera (36x24mm sensor, 20.78mm focal)."""
        return cls(sensor_width_mm=36.0, focal_length_mm=20.78461)


@dataclass
class IMUNoiseConfig:
    """
    MEMS IMU noise configuration for realistic simulation.
    
    Typical consumer MEMS IMU characteristics:
    - Noise: 0.1-1.0° RMS per axis
    - Bias: 0.5-3.0° constant offset
    - Bias drift: 0.01-0.1°/s random walk
    - Yaw drift: 0.5-5.0°/min (gyro integration without magnetometer)
    """
    # White noise (degrees RMS)
    roll_noise_deg: float = 0.5
    pitch_noise_deg: float = 0.5
    yaw_noise_deg: float = 1.0
    
    # Initial bias (degrees) - constant offset
    roll_bias_deg: float = 0.0
    pitch_bias_deg: float = 0.0
    yaw_bias_deg: float = 0.0
    
    # Bias instability / drift rate (degrees per second)
    bias_drift_rate: float = 0.02  # Random walk coefficient
    
    # Yaw drift rate (degrees per minute) - gyro integration error
    yaw_drift_rate: float = 2.0  # Typical for consumer MEMS without magnetometer
    
    # Random seed for reproducibility
    seed: Optional[int] = None
    
    @classmethod
    def perfect(cls) -> 'IMUNoiseConfig':
        """No noise (perfect IMU)."""
        return cls(
            roll_noise_deg=0.0, pitch_noise_deg=0.0, yaw_noise_deg=0.0,
            roll_bias_deg=0.0, pitch_bias_deg=0.0, yaw_bias_deg=0.0,
            bias_drift_rate=0.0, yaw_drift_rate=0.0
        )
    
    @classmethod
    def consumer_mems(cls, seed: Optional[int] = None) -> 'IMUNoiseConfig':
        """Typical consumer-grade MEMS IMU (MPU6050, BMI160, etc.)."""
        return cls(
            roll_noise_deg=0.5, pitch_noise_deg=0.5, yaw_noise_deg=1.0,
            roll_bias_deg=1.0, pitch_bias_deg=1.0, yaw_bias_deg=2.0,
            bias_drift_rate=0.02, yaw_drift_rate=2.0,
            seed=seed
        )
    
    @classmethod
    def poor_mems(cls, seed: Optional[int] = None) -> 'IMUNoiseConfig':
        """Poor quality or uncalibrated MEMS IMU."""
        return cls(
            roll_noise_deg=1.0, pitch_noise_deg=1.0, yaw_noise_deg=2.0,
            roll_bias_deg=3.0, pitch_bias_deg=3.0, yaw_bias_deg=5.0,
            bias_drift_rate=0.05, yaw_drift_rate=5.0,
            seed=seed
        )


class IMUNoiseSimulator:
    """
    Simulates realistic MEMS IMU noise on RPY measurements.
    
    Noise components:
    1. White noise: Random Gaussian noise per sample
    2. Bias: Constant offset (simulates calibration error)
    3. Bias drift: Slowly varying bias (random walk)
    4. Yaw drift: Accumulated heading error (gyro integration)
    """
    
    def __init__(self, config: IMUNoiseConfig, fps: float = 30.0):
        self.cfg = config
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Initialize random generator
        self.rng = np.random.default_rng(config.seed)
        
        # Current bias state (starts at configured bias, then drifts)
        self.roll_bias = config.roll_bias_deg
        self.pitch_bias = config.pitch_bias_deg
        self.yaw_bias = config.yaw_bias_deg
        
        # Accumulated yaw drift
        self.yaw_drift_accumulated = 0.0
        
        # Frame counter
        self.frame_count = 0
    
    def add_noise(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[float, float, float]:
        """
        Add realistic IMU noise to RPY values.
        
        Args:
            roll_deg: True roll in degrees
            pitch_deg: True pitch in degrees
            yaw_deg: True yaw in degrees
            
        Returns:
            (noisy_roll, noisy_pitch, noisy_yaw) in degrees
        """
        # 1. Update bias drift (random walk)
        if self.cfg.bias_drift_rate > 0:
            drift_sigma = self.cfg.bias_drift_rate * np.sqrt(self.dt)
            self.roll_bias += self.rng.normal(0, drift_sigma)
            self.pitch_bias += self.rng.normal(0, drift_sigma)
            self.yaw_bias += self.rng.normal(0, drift_sigma)
        
        # 2. Update yaw drift (accumulates over time)
        if self.cfg.yaw_drift_rate > 0:
            # Convert deg/min to deg/frame
            drift_per_frame = self.cfg.yaw_drift_rate / 60.0 * self.dt
            self.yaw_drift_accumulated += self.rng.normal(0, drift_per_frame)
        
        # 3. Add white noise
        roll_noise = self.rng.normal(0, self.cfg.roll_noise_deg) if self.cfg.roll_noise_deg > 0 else 0
        pitch_noise = self.rng.normal(0, self.cfg.pitch_noise_deg) if self.cfg.pitch_noise_deg > 0 else 0
        yaw_noise = self.rng.normal(0, self.cfg.yaw_noise_deg) if self.cfg.yaw_noise_deg > 0 else 0
        
        # 4. Combine all noise components
        noisy_roll = roll_deg + self.roll_bias + roll_noise
        noisy_pitch = pitch_deg + self.pitch_bias + pitch_noise
        noisy_yaw = yaw_deg + self.yaw_bias + self.yaw_drift_accumulated + yaw_noise
        
        self.frame_count += 1
        
        return noisy_roll, noisy_pitch, noisy_yaw
    
    def get_stats(self) -> Dict[str, float]:
        """Get current noise statistics."""
        return {
            'roll_bias': self.roll_bias,
            'pitch_bias': self.pitch_bias,
            'yaw_bias': self.yaw_bias,
            'yaw_drift': self.yaw_drift_accumulated,
            'total_yaw_error': self.yaw_bias + self.yaw_drift_accumulated
        }


def load_metadata(folder: Path) -> List[Dict]:
    """Load metadata.json from folder."""
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {folder}")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_images(folder: Path) -> List[Path]:
    """Find all images in folder/images/."""
    img_folder = folder / "images"
    if not img_folder.exists():
        raise FileNotFoundError(f"images/ folder not found in {folder}")
    
    images = sorted(img_folder.glob("*.png")) + sorted(img_folder.glob("*.jpg"))
    return images


def match_images_to_metadata(
    images: List[Path], 
    metadata: List[Dict]
) -> List[tuple]:
    """
    Match images to metadata entries by filename.
    
    Args:
        images: List of image paths
        metadata: List of metadata dicts (must have 'filename' key)
    
    Returns:
        List of (image_path, metadata_dict) tuples, sorted by filename
    """
    # Build lookup from filename to metadata
    meta_by_filename = {}
    for entry in metadata:
        if 'filename' in entry:
            meta_by_filename[entry['filename']] = entry
    
    # Match images to metadata
    matched = []
    for img_path in images:
        filename = img_path.name
        if filename in meta_by_filename:
            matched.append((img_path, meta_by_filename[filename]))
    
    # Sort by filename for consistent ordering
    matched.sort(key=lambda x: x[0].name)
    return matched


def create_calibration_for_dataset(
    sample_image: np.ndarray, 
    camera_tilt_deg: float,
    camera_config: Optional[CameraConfig] = None
) -> "CalibrationData":
    """
    Create calibration for dataset using Unity Physical Camera model.
    
    Args:
        sample_image: Sample image to get dimensions
        camera_tilt_deg: Camera tilt angle below horizontal (degrees)
        camera_config: Camera configuration (defaults to Unity default)
    
    Returns:
        CalibrationData with computed intrinsics
    """
    h, w = sample_image.shape[:2]
    
    # Use default Unity camera if not specified
    if camera_config is None:
        camera_config = CameraConfig.unity_default()
    
    # Unity Physical Camera with Gate Fit = Horizontal
    # Step 1: Effective sensor height (crops vertically if aspect != sensor aspect)
    sensor_height_eff = camera_config.sensor_width_mm * h / w
    
    # Step 2: Both fx and fy from physical camera model (consistent)
    fx = camera_config.focal_length_mm * w / camera_config.sensor_width_mm
    fy = camera_config.focal_length_mm * h / sensor_height_eff
    
    cx, cy = w / 2.0, h / 2.0
    
    LOG.info(f"Camera Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    from altitude_estimation_v2 import (
        CameraIntrinsics, CameraExtrinsics, TimeSync, FrameConventions, CalibrationData
    )
 
    intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=w, height=h,
        dist_coeffs=np.zeros(5)  # Unity has no distortion
    )
    
    # Camera tilted down by pitch angle
    extrinsics = CameraExtrinsics.from_tilt_angle(camera_tilt_deg)
    
    time_sync = TimeSync(time_offset=0.0)
    
    return CalibrationData(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        time_sync=time_sync,
        conventions=FrameConventions(),
        camera_model="Drone Camera",
        notes=f"Auto-calibrated for {w}x{h} images, tilt={camera_tilt_deg}°"
    )


def run_test(
    folder: Path, 
    max_frames: Optional[int] = None, 
    init_frames: int = 10, 
    camera_config: Optional[CameraConfig] = None,
    imu_noise_config: Optional[IMUNoiseConfig] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Run altitude estimation on drone captures dataset.
    
    Args:
        folder: Path to dataset folder containing images/ and metadata.json
        max_frames: Maximum frames to process (None = all)
        init_frames: Number of initial frames with known altitude for initialization
        camera_config: Camera configuration (defaults to Unity default)
        imu_noise_config: IMU noise configuration (None = perfect IMU)
        verbose: Print detailed output
    
    Returns:
        List of result dictionaries with frame, gt, est, error, mode, sigma
    """
    # Input validation
    if init_frames < 1:
        raise ValueError(f"init_frames must be >= 1, got {init_frames}")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be >= 1 or None, got {max_frames}")
    
    print(f"\n{'='*60}")
    print(f"ALTITUDE ESTIMATION TEST - {folder.name}")
    print(f"{'='*60}\n")
    
    # Load data
    metadata = load_metadata(folder)
    images = find_images(folder)
    
    print(f"Found {len(metadata)} metadata entries")
    print(f"Found {len(images)} images")
    
    if len(images) == 0:
        LOG.error("No images found!")
        return []
    
    # Match images to metadata by filename
    matched_data = match_images_to_metadata(images, metadata)
    print(f"Matched {len(matched_data)} image-metadata pairs")
    
    if len(matched_data) == 0:
        LOG.error("No images matched to metadata! Check filenames.")
        return []
    
    n_frames = len(matched_data)
    if max_frames:
        n_frames = min(n_frames, max_frames)
    
    print(f"Processing {n_frames} frames\n")
    
    # Load first image to get dimensions
    sample_img = cv2.imread(str(matched_data[0][0]))
    if sample_img is None:
        LOG.error(f"Could not load {matched_data[0][0]}")
        return []
    
    print(f"Image size: {sample_img.shape[1]}x{sample_img.shape[0]}")
    
    # Camera tilt from metadata (pitch = 60° means camera tilted 60° down from horizontal)
    camera_tilt = matched_data[0][1]['pitch']
    print(f"Camera tilt: {camera_tilt}°")
    
    # Compute intrinsics for production API
    h, w = sample_img.shape[:2]
    if camera_config is None:
        camera_config = CameraConfig.unity_default()
 
    sensor_height_eff = camera_config.sensor_width_mm * h / w
    fx = camera_config.focal_length_mm * w / camera_config.sensor_width_mm
    fy = camera_config.focal_length_mm * h / sensor_height_eff
    cx, cy = w / 2.0, h / 2.0
 
    print(f"Init frames: {init_frames} (known altitude)")
    
    # Create IMU noise simulator
    if imu_noise_config is None:
        imu_noise_config = IMUNoiseConfig.perfect()
        print("IMU noise: OFF (perfect)")
    else:
        print(f"IMU noise: ON")
        print(f"  Roll/Pitch/Yaw noise: {imu_noise_config.roll_noise_deg:.1f}°/{imu_noise_config.pitch_noise_deg:.1f}°/{imu_noise_config.yaw_noise_deg:.1f}°")
        print(f"  Bias: {imu_noise_config.roll_bias_deg:.1f}°/{imu_noise_config.pitch_bias_deg:.1f}°/{imu_noise_config.yaw_bias_deg:.1f}°")
        print(f"  Yaw drift: {imu_noise_config.yaw_drift_rate:.1f}°/min")
    
    fps = 30.0
    imu_simulator = IMUNoiseSimulator(imu_noise_config, fps=fps)
     
    # Create production altimeter API
    altimeter = create_altimeter(
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        image_width=int(w), image_height=int(h),
        camera_tilt_deg=float(camera_tilt),
        init_frames=int(init_frames),
        fps=float(fps)
    )
    
    # Track results
    results = []
    errors = []
    processing_times = []
    
    print(f"\n{'Frame':<8} {'GT Alt':<10} {'Est Alt':<10} {'Error':<10} {'Mode':<8} {'σ':<8} {'ms':<8}")
    print("-" * 70)
    
    fps = float(fps)
    total_start_time = time.perf_counter()
    
    for i in range(n_frames):
        # Get matched image and metadata
        img_path, meta = matched_data[i]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            LOG.warning(f"Could not load {img_path}")
            continue
        
        # Get ground truth from metadata
        gt_altitude = meta['height']
        
        # DATASET-SPECIFIC: JSON contains partial camera info
        # - pitch: constant 60° = camera tilt mount angle (NOT varying pitch)
        # - yaw: UAV heading (body yaw = camera yaw for this mount)
        # - roll: CORRUPTED (equals yaw in data, not usable)
        #
        # For this level-flight dataset, body RPY is:
        # - roll = 0 (level flight)
        # - pitch = 0 (level flight)  
        # - yaw = heading from JSON
        #
        # The 60° camera tilt is handled by CameraExtrinsics.from_tilt_angle(60)
        # which is set in create_calibration_for_dataset()
        
        body_yaw_deg = meta['yaw']  # UAV heading
        roll_deg_gt = 0.0   # Level flight assumption
        pitch_deg_gt = 0.0  # Level flight assumption
        yaw_deg_gt = body_yaw_deg
        
        # Apply IMU noise to simulate real MEMS sensor
        roll_deg, pitch_deg, yaw_deg = imu_simulator.add_noise(
            roll_deg_gt, pitch_deg_gt, yaw_deg_gt
        )
        
        # Process frame with timing
        # During init phase (first init_frames frames), provide known altitude
        roll_rad = float(np.deg2rad(roll_deg))
        pitch_rad = float(np.deg2rad(pitch_deg))
        yaw_rad = float(np.deg2rad(yaw_deg))
        
        is_init = altimeter.frame_count < init_frames
        
        frame_start = time.perf_counter()
        try:
            result = altimeter.process(
                image=image,
                rpy=(roll_rad, pitch_rad, yaw_rad),
                known_altitude=gt_altitude if is_init else None
            )
        except Exception as e:
            LOG.warning(f"Altimeter failed on frame {i}: {e}")
            continue
        frame_time_ms = (time.perf_counter() - frame_start) * 1000
        processing_times.append(frame_time_ms)
        
        # Compute error
        if result.is_valid and gt_altitude > 0:
            error = result.altitude_m - gt_altitude
            errors.append(error)
        else:
            error = float('nan')

        results.append({
            'frame': i,
            'gt': gt_altitude,
            'est': result.altitude_m,
            'error': error,
            'mode': result.mode,
            'sigma': result.sigma_m,
            'time_ms': frame_time_ms
        })
        
        # Print progress with constraint info and timing
        if verbose and (i < 30 or i % 50 == 0 or i == n_frames - 1):
            constraint_info = ""
            status = altimeter.get_status()
            last_constraint = status.get('last_constraint')
            if last_constraint and last_constraint.get('s') is not None:
                constraint_info = f"s={last_constraint.get('s'):.3f}"
            print(f"{i:<8} {gt_altitude:<10.1f} {result.altitude_m:<10.1f} "
                  f"{error:+<10.1f} {result.mode:<8} {result.sigma_m:<8.1f} {frame_time_ms:<8.1f} {constraint_info}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    valid_errors = [e for e in errors if not np.isnan(e)]
    if valid_errors:
        mae = np.mean(np.abs(valid_errors))
        rmse = np.sqrt(np.mean(np.array(valid_errors)**2))
        mean_error = np.mean(valid_errors)
        std_error = np.std(valid_errors)
        
        # Compute percentage errors
        valid_results = [r for r in results if not np.isnan(r['error']) and r['gt'] > 0]
        pct_errors = [100 * r['error'] / r['gt'] for r in valid_results]
        mean_pct = np.mean(pct_errors) if pct_errors else 0
        
        print(f"Frames processed: {len(results)}")
        print(f"Valid estimates:  {len(valid_errors)}")
        print(f"Mean Error:       {mean_error:+.2f} m")
        print(f"Std Error:        {std_error:.2f} m")
        print(f"MAE:              {mae:.2f} m")
        print(f"RMSE:             {rmse:.2f} m")
        print(f"Mean % Error:     {mean_pct:+.1f}%")
    else:
        print("No valid estimates produced")
    
    # Performance statistics
    total_time = time.perf_counter() - total_start_time
    if processing_times:
        avg_time = np.mean(processing_times)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)
        std_time = np.std(processing_times)
        fps_achieved = 1000.0 / avg_time if avg_time > 0 else 0
        
        print(f"\nPerformance:")
        print(f"  Total time:     {total_time:.2f} s")
        print(f"  Avg per frame:  {avg_time:.2f} ms")
        print(f"  Min per frame:  {min_time:.2f} ms")
        print(f"  Max per frame:  {max_time:.2f} ms")
        print(f"  Std per frame:  {std_time:.2f} ms")
        print(f"  Throughput:     {fps_achieved:.1f} FPS")
    
    # IMU noise statistics
    imu_stats = imu_simulator.get_stats()
    if imu_stats['total_yaw_error'] != 0 or imu_stats['roll_bias'] != 0:
        print(f"\nIMU Noise (final state):")
        print(f"  Roll bias:      {imu_stats['roll_bias']:+.2f}°")
        print(f"  Pitch bias:     {imu_stats['pitch_bias']:+.2f}°")
        print(f"  Yaw bias:       {imu_stats['yaw_bias']:+.2f}°")
        print(f"  Yaw drift:      {imu_stats['yaw_drift']:+.2f}°")
        print(f"  Total yaw err:  {imu_stats['total_yaw_error']:+.2f}°")
    
    # Mode breakdown
    mode_counts = {}
    for r in results:
        mode = r['mode']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    print(f"\nMode breakdown:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count} frames ({100*count/len(results):.1f}%)")
    
    # Final status
    print(f"\nFinal system status:")
    status = altimeter.get_status()
    for key, val in status.items():
        if key != 'smoother' and key != 'last_constraint':
            print(f"  {key}: {val}")
    
    return results


def visualize_results(results: List[dict], save_path: Optional[Path] = None):
    """Visualize GT vs estimated altitude comparison."""
    # Extract data
    frames = [r['frame'] for r in results]
    gt_values = [r['gt'] for r in results]
    est_values = [r['est'] for r in results]
    errors = [r['error'] for r in results]
    modes = [r['mode'] for r in results]
    
    # Filter out INIT frames for plotting
    valid_idx = [i for i, m in enumerate(modes) if m != 'INIT']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Altitude Estimation Results', fontsize=14, fontweight='bold')
    
    # Plot 1: GT vs Estimated altitude
    ax1 = axes[0]
    ax1.plot(frames, gt_values, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(frames, est_values, 'r--', linewidth=2, label='Estimated', alpha=0.8)
    ax1.set_ylabel('Altitude (m)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Ground Truth vs Estimated Altitude')
    
    # Shade INIT region
    init_frames = [f for f, m in zip(frames, modes) if m == 'INIT']
    if init_frames:
        ax1.axvspan(min(init_frames), max(init_frames), alpha=0.2, color='gray', label='Init')
    
    # Plot 2: Error over time
    ax2 = axes[1]
    valid_frames = [frames[i] for i in valid_idx]
    valid_errors = [errors[i] for i in valid_idx if not np.isnan(errors[i])]
    valid_frames_clean = [frames[i] for i in valid_idx if not np.isnan(errors[i])]
    
    ax2.plot(valid_frames_clean, valid_errors, 'g-', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.fill_between(valid_frames_clean, valid_errors, 0, alpha=0.3, color='green')
    ax2.set_ylabel('Error (m)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Estimation Error (Estimated - GT)')
    
    # Add mean error line
    if valid_errors:
        mean_err = np.mean(valid_errors)
        ax2.axhline(y=mean_err, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_err:+.2f}m')
        ax2.legend(loc='upper right', fontsize=10)
    
    # Plot 3: Percentage error
    ax3 = axes[2]
    pct_errors = []
    pct_frames = []
    for i in valid_idx:
        if not np.isnan(errors[i]) and gt_values[i] > 0:
            pct_errors.append(100 * errors[i] / gt_values[i])
            pct_frames.append(frames[i])
    
    if pct_errors:
        ax3.plot(pct_frames, pct_errors, 'm-', linewidth=1.5, alpha=0.8)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax3.fill_between(pct_frames, pct_errors, 0, alpha=0.3, color='magenta')
        mean_pct = np.mean(pct_errors)
        ax3.axhline(y=mean_pct, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_pct:+.1f}%')
        ax3.legend(loc='upper right', fontsize=10)
    
    ax3.set_xlabel('Frame', fontsize=11)
    ax3.set_ylabel('Error (%)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Percentage Error')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def main() -> int:
    """Main entry point for test script."""
    parser = argparse.ArgumentParser(
        description='Test altitude estimation on drone captures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--folder', type=str, default='DroneCaptures1',
                        help='Folder containing images/ and metadata.json')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process (None = all)')
    parser.add_argument('--init-frames', type=int, default=10,
                        help='Number of initial frames with known altitude')
    parser.add_argument('--sensor-width', type=float, default=36.0,
                        help='Camera sensor width in mm')
    parser.add_argument('--focal-length', type=float, default=20.78461,
                        help='Camera focal length in mm')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--plot', action='store_true',
                        help='Show visualization plot')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Save plot to file (e.g., results.png)')
    
    # IMU noise simulation arguments
    parser.add_argument('--imu-noise', type=str, default='off',
                        choices=['off', 'consumer', 'poor', 'custom'],
                        help='IMU noise preset: off=perfect, consumer=typical MEMS, poor=bad MEMS')
    parser.add_argument('--imu-noise-deg', type=float, default=1.0,
                        help='IMU white noise (degrees RMS) for custom mode')
    parser.add_argument('--imu-bias-deg', type=float, default=2.0,
                        help='IMU bias (degrees) for custom mode')
    parser.add_argument('--imu-yaw-drift', type=float, default=2.0,
                        help='Yaw drift rate (degrees/minute) for custom mode')
    parser.add_argument('--imu-seed', type=int, default=None,
                        help='Random seed for IMU noise (for reproducibility)')
    args = parser.parse_args()
    
    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = Path(__file__).parent / folder
    
    if not folder.exists():
        LOG.error(f"Folder not found: {folder}")
        return 1
    
    # Create camera config from arguments
    camera_config = CameraConfig(
        sensor_width_mm=args.sensor_width,
        focal_length_mm=args.focal_length
    )
    
    # Create IMU noise config from arguments
    imu_noise_config = None
    if args.imu_noise == 'off':
        imu_noise_config = None  # Will use perfect IMU
    elif args.imu_noise == 'consumer':
        imu_noise_config = IMUNoiseConfig.consumer_mems(seed=args.imu_seed)
    elif args.imu_noise == 'poor':
        imu_noise_config = IMUNoiseConfig.poor_mems(seed=args.imu_seed)
    elif args.imu_noise == 'custom':
        imu_noise_config = IMUNoiseConfig(
            roll_noise_deg=args.imu_noise_deg,
            pitch_noise_deg=args.imu_noise_deg,
            yaw_noise_deg=args.imu_noise_deg * 2,  # Yaw typically noisier
            roll_bias_deg=args.imu_bias_deg,
            pitch_bias_deg=args.imu_bias_deg,
            yaw_bias_deg=args.imu_bias_deg * 1.5,
            bias_drift_rate=0.02,
            yaw_drift_rate=args.imu_yaw_drift,
            seed=args.imu_seed
        )
    
    results = run_test(
        folder, 
        max_frames=args.max_frames, 
        init_frames=args.init_frames,
        camera_config=camera_config,
        imu_noise_config=imu_noise_config,
        verbose=not args.quiet
    )
    
    # Visualization
    if results and (args.plot or args.save_plot):
        save_path = Path(args.save_plot) if args.save_plot else None
        visualize_results(results, save_path=save_path)
    
    return 0


if __name__ == '__main__':
    exit(main())
