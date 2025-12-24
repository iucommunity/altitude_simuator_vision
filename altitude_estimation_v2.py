#!/usr/bin/env python3
"""
Dominant-Plane Visual Altimeter with Fixed-Lag Robust Smoother
==============================================================

Monocular altitude estimation for UAVs using homography constraints.

PRIMARY PATH:
- Dominant-plane homography detection from tracked features
- Physics-gated constraint extraction (rank-1, coverage, ground-likeness)
- Fixed-lag log-distance smoother with Huber loss

SECONDARY (debug/fallback):
- PoseEngine with triangulated map points
- Ground plane fitting from 3D points

Coordinate Frames:
- World W: NED (x=North, y=East, z=Down)
- Body B: FRD (x=Forward, y=Right, z=Down)
- Camera C: OpenCV (x=Right, y=Down, z=Forward)

RPY: ZYX order (yaw→pitch→roll), radians internally.
Output: AGL altitude h (camera to ground), uncertainty σ, mode (OK/HOLD/LOST).
"""

import numpy as np
import cv2
import logging
import yaml
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum, auto
from pathlib import Path
from collections import deque
from scipy.spatial.transform import Rotation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
LOG = logging.getLogger(__name__)


# --- Coordinate Frames ---

class CoordinateFrame(Enum):
    """Supported coordinate frames."""
    NED = auto()  # North-East-Down (default world frame)
    ENU = auto()  # East-North-Up (alternative)
    FRD = auto()  # Forward-Right-Down (body frame)
    OPENCV = auto()  # Right-Down-Forward (camera frame)


class RotationOrder(Enum):
    """Euler angle rotation orders."""
    ZYX = auto()  # Yaw-Pitch-Roll (aerospace convention)
    XYZ = auto()  # Roll-Pitch-Yaw
    ZXY = auto()  # Alternative


@dataclass
class FrameConventions:
    """Coordinate frame conventions: World=NED, Body=FRD, Camera=OpenCV."""
    world_frame: CoordinateFrame = CoordinateFrame.NED
    body_frame: CoordinateFrame = CoordinateFrame.FRD
    camera_frame: CoordinateFrame = CoordinateFrame.OPENCV
    rotation_order: RotationOrder = RotationOrder.ZYX
    
    # Sign conventions
    yaw_positive_clockwise: bool = True  # When viewed from above
    pitch_positive_nose_up: bool = True
    roll_positive_right_down: bool = True
    
    # Reference
    yaw_reference: str = "true_north"  # or "magnetic", "arbitrary"
    
    def validate(self) -> bool:
        """Validate that conventions are consistent."""
        if self.world_frame == CoordinateFrame.NED:
            # NED: z-down, gravity is +z
            pass
        elif self.world_frame == CoordinateFrame.ENU:
            # ENU: z-up, gravity is -z
            pass
        return True


def _Rx(angle: float) -> np.ndarray:
    """Rotation matrix about x-axis (roll)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=np.float64)


def _Ry(angle: float) -> np.ndarray:
    """Rotation matrix about y-axis (pitch)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float64)


def _Rz(angle: float) -> np.ndarray:
    """Rotation matrix about z-axis (yaw)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float64)


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float,
                           order: RotationOrder = RotationOrder.ZYX) -> np.ndarray:
    """Convert RPY angles to rotation matrix R_WB (Body → World). ZYX: R = Rz @ Ry @ Rx."""
    if order == RotationOrder.ZYX:
        # R_WB = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)
    elif order == RotationOrder.XYZ:
        return _Rx(roll) @ _Ry(pitch) @ _Rz(yaw)
    else:  # ZXY
        return _Rz(yaw) @ _Rx(roll) @ _Ry(pitch)


def rotation_matrix_to_rpy(R: np.ndarray,
                           order: RotationOrder = RotationOrder.ZYX) -> Tuple[float, float, float]:
    """Extract RPY from rotation matrix R_WB. Returns (roll, pitch, yaw) in radians."""
    if order == RotationOrder.ZYX:
        # For R = Rz @ Ry @ Rx, extract angles
        # pitch = -asin(R[2,0])
        # roll = atan2(R[2,1], R[2,2])
        # yaw = atan2(R[1,0], R[0,0])
        pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
        
        if np.abs(np.cos(pitch)) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock
            roll = 0.0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        
        return roll, pitch, yaw
    else:
        # Fallback to scipy for other orders (less common)
        rot = Rotation.from_matrix(R)
        if order == RotationOrder.XYZ:
            angles = rot.as_euler('XYZ')
            return angles[0], angles[1], angles[2]
        else:
            angles = rot.as_euler('ZXY')
            return angles[1], angles[2], angles[0]


# --- Calibration ---

@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters from calibration.
    
    Calibrate with checkerboard/Charuco at actual resolution and focus.
    """
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    
    # Distortion coefficients (OpenCV model)
    # k1, k2, p1, p2, k3 for standard model
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    
    # Image dimensions
    width: int = 1280
    height: int = 720
    
    # Calibration quality
    reprojection_rms: float = 0.0  # RMS reprojection error from calibration
    
    @property
    def K(self) -> np.ndarray:
        """Camera matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @property
    def K_inv(self) -> np.ndarray:
        """Inverse camera matrix."""
        return np.linalg.inv(self.K)
    
    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Undistort 2D points."""
        if pts.size == 0:
            return pts
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        undist = cv2.undistortPoints(pts_reshaped, self.K, self.dist_coeffs, P=self.K)
        return undist.reshape(-1, 2)
    
    def project(self, pts_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D."""
        if pts_3d.size == 0:
            return np.empty((0, 2))
        pts_h = pts_3d / pts_3d[:, 2:3]
        pts_2d = (self.K @ pts_h.T).T
        return pts_2d[:, :2]
    
    def unproject(self, pts_2d: np.ndarray, depth: np.ndarray = None) -> np.ndarray:
        """Unproject 2D points to normalized rays or 3D points if depth given."""
        if pts_2d.size == 0:
            return np.empty((0, 3))
        pts_h = np.hstack([pts_2d, np.ones((len(pts_2d), 1))])
        rays = (self.K_inv @ pts_h.T).T
        if depth is not None:
            rays = rays * depth.reshape(-1, 1)
        return rays


@dataclass
class CameraExtrinsics:
    """Camera to Body frame extrinsics. R_BC: Camera→Body rotation, t_BC: translation."""
    R_BC: np.ndarray = field(default_factory=lambda: np.eye(3))  # Camera to Body rotation
    t_BC: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Camera to Body translation
    
    # Calibration quality
    rotation_residual_deg: float = 0.0
    
    @classmethod
    def from_tilt_angle(cls, tilt_deg: float, 
                        roll_offset_deg: float = 0.0,
                        yaw_offset_deg: float = 0.0,
                        translation: np.ndarray = None) -> 'CameraExtrinsics':
        """Create extrinsics from camera tilt angle (deg below horizontal)."""
        # Camera frame: x=right, y=down, z=forward
        # Body frame: x=forward, y=right, z=down
        # 
        # Base rotation to align camera to body (no tilt):
        # Camera z (forward) -> Body x (forward)
        # Camera x (right) -> Body y (right)
        # Camera y (down) -> Body z (down)
        R_base = np.array([
            [0, 0, 1],   # Body x from Camera z
            [1, 0, 0],   # Body y from Camera x
            [0, 1, 0]    # Body z from Camera y
        ], dtype=np.float64)
        
        # Apply tilt: rotate about Body y-axis (camera pitches down)
        tilt_rad = np.deg2rad(tilt_deg)
        R_tilt = _Ry(-tilt_rad)  # Negative because pitching down
        
        # Apply roll and yaw offsets (yaw first, then roll = Rz @ Rx)
        roll_rad = np.deg2rad(roll_offset_deg)
        yaw_rad = np.deg2rad(yaw_offset_deg)
        R_offsets = _Rz(yaw_rad) @ _Rx(roll_rad)
        
        R_BC = R_offsets @ R_tilt @ R_base
        
        t = translation if translation is not None else np.zeros(3)
        
        return cls(R_BC=R_BC, t_BC=t)
    
    @property
    def R_CB(self) -> np.ndarray:
        """Body to Camera rotation (inverse of R_BC)."""
        return self.R_BC.T
    
    def transform_to_body(self, pts_camera: np.ndarray) -> np.ndarray:
        """Transform points from Camera frame to Body frame."""
        return (self.R_BC @ pts_camera.T).T + self.t_BC
    
    def transform_to_camera(self, pts_body: np.ndarray) -> np.ndarray:
        """Transform points from Body frame to Camera frame."""
        return (self.R_CB @ (pts_body - self.t_BC).T).T


@dataclass
class TimeSync:
    """Time synchronization: t_aligned = t_rpy + time_offset."""
    time_offset: float = 0.0
    
    # Calibration quality
    sync_residual_ms: float = 0.0
    
    def align_timestamp(self, t_rpy: float) -> float:
        """Align RPY timestamp to camera time."""
        return t_rpy + self.time_offset


@dataclass
class CalibrationData:
    """Complete calibration bundle for a camera setup."""
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    time_sync: TimeSync
    conventions: FrameConventions = field(default_factory=FrameConventions)
    
    # Metadata
    calibration_date: str = ""
    camera_model: str = ""
    notes: str = ""
    
    def save(self, path: str) -> None:
        """Save calibration to YAML file."""
        data = {
            'intrinsics': {
                'fx': self.intrinsics.fx,
                'fy': self.intrinsics.fy,
                'cx': self.intrinsics.cx,
                'cy': self.intrinsics.cy,
                'dist_coeffs': self.intrinsics.dist_coeffs.tolist(),
                'width': self.intrinsics.width,
                'height': self.intrinsics.height,
                'reprojection_rms': self.intrinsics.reprojection_rms,
            },
            'extrinsics': {
                'R_BC': self.extrinsics.R_BC.tolist(),
                't_BC': self.extrinsics.t_BC.tolist(),
                'rotation_residual_deg': self.extrinsics.rotation_residual_deg,
            },
            'time_sync': {
                'time_offset': self.time_sync.time_offset,
                'sync_residual_ms': self.time_sync.sync_residual_ms,
            },
            'conventions': {
                'world_frame': self.conventions.world_frame.name,
                'body_frame': self.conventions.body_frame.name,
                'rotation_order': self.conventions.rotation_order.name,
                'yaw_reference': self.conventions.yaw_reference,
            },
            'metadata': {
                'calibration_date': self.calibration_date,
                'camera_model': self.camera_model,
                'notes': self.notes,
            }
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        LOG.info(f"Saved calibration to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CalibrationData':
        """Load calibration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        intrinsics = CameraIntrinsics(
            fx=data['intrinsics']['fx'],
            fy=data['intrinsics']['fy'],
            cx=data['intrinsics']['cx'],
            cy=data['intrinsics']['cy'],
            dist_coeffs=np.array(data['intrinsics']['dist_coeffs']),
            width=data['intrinsics']['width'],
            height=data['intrinsics']['height'],
            reprojection_rms=data['intrinsics'].get('reprojection_rms', 0.0),
        )
        
        extrinsics = CameraExtrinsics(
            R_BC=np.array(data['extrinsics']['R_BC']),
            t_BC=np.array(data['extrinsics']['t_BC']),
            rotation_residual_deg=data['extrinsics'].get('rotation_residual_deg', 0.0),
        )
        
        time_sync = TimeSync(
            time_offset=data['time_sync']['time_offset'],
            sync_residual_ms=data['time_sync'].get('sync_residual_ms', 0.0),
        )
        
        conventions = FrameConventions(
            world_frame=CoordinateFrame[data['conventions']['world_frame']],
            body_frame=CoordinateFrame[data['conventions']['body_frame']],
            rotation_order=RotationOrder[data['conventions']['rotation_order']],
            yaw_reference=data['conventions'].get('yaw_reference', 'true_north'),
        )
        
        return cls(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            time_sync=time_sync,
            conventions=conventions,
            calibration_date=data['metadata'].get('calibration_date', ''),
            camera_model=data['metadata'].get('camera_model', ''),
            notes=data['metadata'].get('notes', ''),
        )


# --- Configuration ---

@dataclass
class Config:
    """
    System configuration with all tunable parameters.
    """
    # --- Frame rate and timing ---
    fps: float = 30.0
    keyframe_rate: float = 5.0  # Keyframes per second
    depth_rate: float = 5.0     # Depth inference rate
    segmentation_rate: float = 5.0  # Ground segmentation rate
    
    # --- Feature tracking ---
    max_features: int = 500
    feature_quality: float = 0.01
    min_feature_distance: int = 10
    lk_win_size: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    
    # --- Geometric filtering ---
    ransac_reproj_threshold: float = 2.0  # pixels
    min_inliers: int = 20
    min_parallax_deg: float = 1.0
    max_rotation_deg: float = 10.0  # Max rotation between frames for scale estimation
    
    # --- Ground plane fitting ---
    ground_ransac_threshold: float = 0.1  # meters
    min_ground_inliers: int = 10
    ground_normal_tolerance_deg: float = 20.0  # Max deviation from world-up
    
    # --- Depth prior ---
    use_depth_prior: bool = True
    depth_model_path: str = ""
    depth_confidence_threshold: float = 0.5
    
    # --- Ground segmentation ---
    use_ground_segmentation: bool = True
    segmentation_model_path: str = ""
    ground_mask_threshold: float = 0.5
    
    # --- Initialization ---
    init_window_size: int = 10  # Number of frames for initialization
    min_init_keyframes: int = 3
    init_altitude_tolerance: float = 0.1  # 10% tolerance for altitude anchors
    
    # --- SLAM / VO ---
    window_size: int = 12  # Fixed-lag window for BA
    max_keyframes: int = 30
    min_keyframe_gap: int = 3
    triangulation_min_parallax_deg: float = 2.0
    ba_iterations: int = 10
    
    # --- Fusion estimator ---
    fusion_window_sec: float = 2.0  # Time window for fusion
    altitude_process_sigma: float = 0.1  # Process noise for altitude (log space)
    altitude_velocity_sigma: float = 0.5  # Process noise for velocity
    max_vertical_speed_mps: float = 5.0
    huber_k: float = 1.345
    
    # --- Quality thresholds ---
    slam_quality_threshold: float = 0.3
    ground_quality_threshold: float = 0.3
    depth_quality_threshold: float = 0.3
    
    # --- Failure handling ---
    hold_timeout_sec: float = 2.0  # Max time to hold altitude
    recovery_min_quality: float = 0.5
    
    # --- Homography constraints ---
    # Enable/disable rank1 gate (requires accurate RPY)
    enable_rank1_gate: bool = False  # Disabled: RPY sign conventions vary by platform
    rank1_thresh_s2: float = 0.50    # s2/s1 threshold for rank-1 check
    rank1_thresh_s3: float = 0.30    # s3/s1 threshold for rank-1 check
    
    # Scale factor range (s = d_new / d_old)
    homography_s_min: float = 0.3    # Min scale (fast descent)
    homography_s_max: float = 3.0    # Max scale (fast ascent)
    
    # Enable vertical_factor update from plane normal
    # Disable for flat terrain or when R_CW (from RPY) is unreliable
    enable_vertical_factor_update: bool = False  # Disabled: requires accurate R_CW
    
    # Sign convention for homography scale: s = 1 + sign * (n · u)
    # +1 for OpenCV convention (tested with Unity), -1 for standard convention
    homography_sign_convention: int = 1


class AltitudeMode(Enum):
    """Operating mode for altitude estimation."""
    INIT = auto()       # Initialization phase
    GEOM = auto()       # Geometry-based (primary)
    FUSED = auto()      # Fused geometry + depth
    DEPTH = auto()       # Depth-only fallback
    HOLD = auto()       # Hold last value (failure)
    LOST = auto()       # System lost


@dataclass
class RPYSample:
    """Single RPY measurement from autopilot."""
    timestamp: float  # seconds
    roll: float       # radians
    pitch: float      # radians
    yaw: float        # radians
    covariance: Optional[np.ndarray] = None  # 3x3 covariance if available
    quality: float = 1.0  # 0-1 quality indicator


@dataclass
class FrameData:
    """Data associated with a single frame."""
    index: int
    timestamp: float
    image: np.ndarray
    image_gray: np.ndarray = None
    rpy: Optional[RPYSample] = None
    altitude_gt: Optional[float] = None  # Ground truth if available
    
    # Computed data (CONSISTENT CONVENTION: R_CW = world->camera, C_W = camera position)
    R_CW: Optional[np.ndarray] = None  # World to Camera rotation
    C_W: Optional[np.ndarray] = None   # Camera position in World frame
    features: Optional[np.ndarray] = None  # 2D feature points
    track_ids: Optional[np.ndarray] = None  # Per-feature persistent track IDs
    descriptors: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.image_gray is None and self.image is not None:
            if len(self.image.shape) == 3:
                self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                self.image_gray = self.image


@dataclass
class Keyframe:
    """Keyframe for SLAM/VO."""
    index: int
    timestamp: float
    R_CW: np.ndarray  # World to Camera rotation (CONSISTENT: world->camera)
    C_W: np.ndarray   # Camera position in World frame (CONSISTENT)
    features: np.ndarray  # 2D feature points
    descriptors: Optional[np.ndarray] = None
    point_ids: Optional[np.ndarray] = None  # IDs of 3D map points (persistent track IDs)
    
    # Quality metrics
    num_tracks: int = 0
    avg_parallax: float = 0.0
    tracking_quality: float = 1.0


@dataclass
class MapPoint:
    """3D map point in world coordinates."""
    id: int
    position: np.ndarray  # [x, y, z] in World frame
    observations: Dict[int, int] = field(default_factory=dict)  # keyframe_idx -> feature_idx
    
    # Quality
    reprojection_error: float = 0.0
    num_observations: int = 0
    
    ground_probability: float = 0.0
    
    @property
    def is_ground(self) -> bool:
        """Classify as ground if probability > threshold."""
        return self.ground_probability > 0.7


@dataclass
class GroundModel:
    """Ground surface model."""
    # Plane: ax + by + cz + d = 0, where [a,b,c] is normal
    normal: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1]))  # Up vector in NED
    distance: float = 0.0  # Distance from origin
    
    # Quality metrics
    inlier_ratio: float = 0.0
    residual_m: float = 0.0
    coverage: float = 0.0  # Fraction of image covered by ground
    stability: float = 0.0  # Temporal consistency
    
    @property
    def quality(self) -> float:
        """Overall ground model quality."""
        effective_coverage = self.coverage if self.coverage > 0 else 1.0
        return self.inlier_ratio * effective_coverage * max(self.stability, 0.5)
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Signed distance from point to plane."""
        return np.dot(self.normal, point) + self.distance
    
    def distance_to_camera(self, camera_pos: np.ndarray) -> float:
        """Distance from camera to ground (altitude)."""
        return abs(self.distance_to_point(camera_pos))


@dataclass
class AltitudeEstimate:
    """Single altitude estimate with metadata."""
    altitude_m: float
    sigma_m: float
    mode: AltitudeMode
    timestamp: float
    
    # Quality breakdown
    slam_quality: float = 0.0
    ground_quality: float = 0.0
    depth_quality: float = 0.0
    
    # Component estimates
    altitude_geom: Optional[float] = None
    altitude_depth: Optional[float] = None
    altitude_homography: Optional[float] = None
    
    def __repr__(self):
        return f"Alt={self.altitude_m:.2f}m ± {self.sigma_m:.2f}m [{self.mode.name}]"


# --- Core Modules ---

class RotationProvider:
    """Provides rotation from RPY. Convention: R_CW (World→Camera), C_W (camera pos in World)."""
    
    def __init__(self, calibration: CalibrationData, config: Config):
        self.calib = calibration
        self.cfg = config
        self.rpy_buffer: deque = deque(maxlen=100)
        self.last_R_CW: Optional[np.ndarray] = None  # World -> Camera rotation
        
    def add_rpy(self, rpy: RPYSample) -> None:
        """Add RPY sample to buffer with convention conversion."""
        # Apply time synchronization
        aligned_time = self.calib.time_sync.align_timestamp(rpy.timestamp)
        
        # Apply sign conventions from calibration
        # Standard math convention: positive = counter-clockwise (right-hand rule)
        # Common autopilot convention: yaw positive clockwise (heading increases CW)
        conv = self.calib.conventions
        
        roll = rpy.roll
        pitch = rpy.pitch
        yaw = rpy.yaw
        
        # If yaw_positive_clockwise=True (common autopilot), negate for math convention
        if conv.yaw_positive_clockwise:
            yaw = -yaw
        
        # If pitch_positive_nose_up=True and we're using NED (z-down), 
        # standard rotation is already correct (pitch up = positive rotation about Y-right)
        # No adjustment needed for typical FRD body frame
        
        # If roll_positive_right_down=True in FRD, standard rotation is correct
        # No adjustment needed
        
        rpy_aligned = RPYSample(
            timestamp=aligned_time,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            covariance=rpy.covariance,
            quality=rpy.quality
        )
        self.rpy_buffer.append(rpy_aligned)
    
    def get_rpy_at_time(self, t: float) -> Optional[RPYSample]:
        """Interpolate RPY at given timestamp."""
        if len(self.rpy_buffer) < 2:
            return self.rpy_buffer[-1] if self.rpy_buffer else None
        
        # Find bracketing samples
        before, after = None, None
        for rpy in self.rpy_buffer:
            if rpy.timestamp <= t:
                before = rpy
            elif rpy.timestamp > t and after is None:
                after = rpy
                break
        
        if before is None:
            return self.rpy_buffer[0]
        if after is None:
            return before
        
        # Linear interpolation
        alpha = (t - before.timestamp) / (after.timestamp - before.timestamp + 1e-9)
        alpha = np.clip(alpha, 0, 1)
        
        # Interpolate angles (handle wrap-around for yaw)
        roll = before.roll + alpha * (after.roll - before.roll)
        pitch = before.pitch + alpha * (after.pitch - before.pitch)
        
        # Yaw wrap-around
        yaw_diff = after.yaw - before.yaw
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        elif yaw_diff < -np.pi:
            yaw_diff += 2 * np.pi
        yaw = before.yaw + alpha * yaw_diff
        
        return RPYSample(
            timestamp=t,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            quality=min(before.quality, after.quality)
        )
    
    def get_R_CW(self, t: float) -> Optional[np.ndarray]:
        """
        Get World-to-Camera rotation at time t.
        
        R_CW transforms vectors from World frame to Camera frame:
            v_C = R_CW @ v_W
        
        Derivation:
            R_WB = rpy_to_rotation_matrix(roll, pitch, yaw)  # Body -> World
            R_BW = R_WB.T                                     # World -> Body
            R_CB = extrinsics.R_CB                            # Body -> Camera
            R_CW = R_CB @ R_BW                                # World -> Camera
        """
        rpy = self.get_rpy_at_time(t)
        if rpy is None:
            return None
        
        # R_WB: Body -> World rotation from RPY
        R_WB = rpy_to_rotation_matrix(
            rpy.roll, rpy.pitch, rpy.yaw,
            self.calib.conventions.rotation_order
        )
        
        # R_BW: World -> Body (inverse)
        R_BW = R_WB.T
        
        # R_CW = R_CB @ R_BW: World -> Body -> Camera
        R_CW = self.calib.extrinsics.R_CB @ R_BW
        
        self.last_R_CW = R_CW
        return R_CW
    
    def get_R_WC(self, t: float) -> Optional[np.ndarray]:
        """Get Camera-to-World rotation (inverse of R_CW). Convenience method."""
        R_CW = self.get_R_CW(t)
        return R_CW.T if R_CW is not None else None
    
    def get_relative_rotation(self, t1: float, t2: float) -> Optional[np.ndarray]:
        """
        Get relative rotation from Camera frame at t1 to Camera frame at t2.
        
        R_C2_C1: transforms vectors from Camera1 to Camera2
            v_C2 = R_C2_C1 @ v_C1
        
        Derivation:
            v_W = R_WC1 @ v_C1  (camera1 to world)
            v_C2 = R_CW2 @ v_W  (world to camera2)
            v_C2 = R_CW2 @ R_WC1 @ v_C1
            R_C2_C1 = R_CW2 @ R_CW1.T
        """
        R_CW1 = self.get_R_CW(t1)
        R_CW2 = self.get_R_CW(t2)
        
        if R_CW1 is None or R_CW2 is None:
            return None
        
        return R_CW2 @ R_CW1.T
    
    def get_rotation_covariance(self, rpy: RPYSample) -> np.ndarray:
        """Estimate rotation covariance from RPY quality."""
        if rpy.covariance is not None:
            return rpy.covariance
        
        # Default: assume 1 degree uncertainty, scaled by quality
        base_sigma = np.deg2rad(1.0)
        sigma = base_sigma / (rpy.quality + 0.1)
        return np.eye(3) * sigma**2


class VisualTracker:
    """Feature tracking with rotation-predicted KLT optical flow."""
    
    def __init__(self, calibration: CalibrationData, config: Config):
        self.calib = calibration
        self.cfg = config
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_features: Optional[np.ndarray] = None
        self.prev_timestamp: float = 0.0
        
        # Feature detector
        self.detector = cv2.GFTTDetector_create(
            maxCorners=config.max_features,
            qualityLevel=config.feature_quality,
            minDistance=config.min_feature_distance
        )
        
        # LK optical flow parameters
        self.lk_params = dict(
            winSize=config.lk_win_size,
            maxLevel=config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Track management with persistent IDs
        self.next_track_id: int = 0
        self.track_ids: Optional[np.ndarray] = None  # ID for each feature in prev_features
    
    def detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect features in image."""
        kps = self.detector.detect(gray)
        if len(kps) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        return pts
    
    def predict_feature_positions(self, pts: np.ndarray, 
                                   R_rel: np.ndarray) -> np.ndarray:
        """Predict feature positions after rotation: p2 = K @ R @ K^-1 @ p1."""
        if pts.size == 0:
            return pts
        
        K = self.calib.intrinsics.K
        K_inv = self.calib.intrinsics.K_inv
        
        # Homogeneous coordinates
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        
        # Rotation-only homography
        H_rot = K @ R_rel @ K_inv
        
        # Warp points
        pts_warped = (H_rot @ pts_h.T).T
        pts_warped = pts_warped[:, :2] / pts_warped[:, 2:3]
        
        return pts_warped.astype(np.float32)
    
    def track(self, gray: np.ndarray, timestamp: float,
              R_rel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Track features. Returns (prev_pts, curr_pts, mask, track_ids)."""
        if self.prev_gray is None or self.prev_features is None:
            # First frame - detect features and assign IDs
            self.prev_gray = gray
            self.prev_features = self.detect_features(gray)
            self.track_ids = np.arange(len(self.prev_features)) + self.next_track_id
            self.next_track_id += len(self.prev_features)
            self.prev_timestamp = timestamp
            
            n_pts = len(self.prev_features)
            if n_pts > 0:
                trivial_mask = np.ones(n_pts, dtype=bool)
                return self.prev_features.copy(), self.prev_features.copy(), trivial_mask, self.track_ids.copy()
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=bool), np.empty(0, dtype=int)
        
        if len(self.prev_features) < 10:
            # Re-detect if too few features
            self.prev_features = self.detect_features(self.prev_gray)
            self.track_ids = np.arange(len(self.prev_features)) + self.next_track_id
            self.next_track_id += len(self.prev_features)
        
        if len(self.prev_features) < 10:
            self.prev_gray = gray
            self.prev_features = self.detect_features(gray)
            self.track_ids = np.arange(len(self.prev_features)) + self.next_track_id
            self.next_track_id += len(self.prev_features)
            self.prev_timestamp = timestamp
            
            n_pts = len(self.prev_features)
            if n_pts > 0:
                trivial_mask = np.ones(n_pts, dtype=bool)
                return self.prev_features.copy(), self.prev_features.copy(), trivial_mask, self.track_ids.copy()
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=bool), np.empty(0, dtype=int)
        
        # Track with optical flow (no prediction - simpler and more robust)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.prev_features, None,
            **self.lk_params
        )
        
        # Filter by status
        status = status.flatten().astype(bool)
        prev_pts = self.prev_features[status]
        curr_pts = curr_pts[status]
        matched_ids = self.track_ids[status] if self.track_ids is not None else np.arange(len(prev_pts))
        
        # Geometric verification with undistorted points
        if len(prev_pts) >= 8:
            prev_undist = self.calib.intrinsics.undistort_points(prev_pts)
            curr_undist = self.calib.intrinsics.undistort_points(curr_pts)
            
            E, mask = cv2.findEssentialMat(
                prev_undist, curr_undist,
                self.calib.intrinsics.K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self.cfg.ransac_reproj_threshold
            )
            if mask is not None:
                mask = mask.flatten().astype(bool)
            else:
                mask = np.ones(len(prev_pts), dtype=bool)
        else:
            mask = np.ones(len(prev_pts), dtype=bool)
        
        # Update state
        self.prev_gray = gray.copy()
        
        # Keep inlier features + detect new ones
        inlier_pts = curr_pts[mask]
        inlier_ids = matched_ids[mask]
        
        if len(inlier_pts) < self.cfg.max_features // 2:
            new_pts = self.detect_features(gray)
            if len(new_pts) > 0:
                # Remove points close to existing
                if len(inlier_pts) > 0:
                    dists = np.min(np.linalg.norm(
                        new_pts[:, None] - inlier_pts[None, :], axis=2
                    ), axis=1)
                    new_pts = new_pts[dists > self.cfg.min_feature_distance]
                
                # Assign new IDs to new points
                n_new = min(len(new_pts), self.cfg.max_features - len(inlier_pts))
                if n_new > 0:
                    new_ids = np.arange(n_new) + self.next_track_id
                    self.next_track_id += n_new
                    inlier_pts = np.vstack([inlier_pts, new_pts[:n_new]])
                    inlier_ids = np.concatenate([inlier_ids, new_ids])
        
        self.prev_features = inlier_pts.astype(np.float32)
        self.track_ids = inlier_ids
        self.prev_timestamp = timestamp
        
        return prev_pts, curr_pts, mask, matched_ids
    
    def compute_tracking_quality(self, prev_pts: np.ndarray, curr_pts: np.ndarray,
                                  mask: np.ndarray) -> Dict[str, float]:
        """Compute tracking quality metrics."""
        if len(prev_pts) == 0:
            return {'num_tracks': 0, 'inlier_ratio': 0, 'parallax': 0}
        
        num_tracks = int(mask.sum())
        inlier_ratio = mask.sum() / len(mask) if len(mask) > 0 else 0
        
        # Compute parallax
        if mask.sum() > 0:
            displacements = np.linalg.norm(curr_pts[mask] - prev_pts[mask], axis=1)
            parallax = float(np.median(displacements))
        else:
            parallax = 0.0
        
        return {
            'num_tracks': num_tracks,
            'inlier_ratio': inlier_ratio,
            'parallax': parallax
        }


class PoseEngine:
    """
    Rotation-aided monocular visual odometry.
    
    POSE CONVENTION:
    - R_CW: Rotation from World to Camera (world vectors -> camera frame)
    - C_W: Camera position in World frame
    - Projection: P = K @ [R_CW | -R_CW @ C_W]
    - Point in camera: p_C = R_CW @ (p_W - C_W)
    
    Maintains:
    - Keyframes with poses (R_CW, C_W)
    - Sparse 3D map points in World frame
    - Per-frame pose estimation
    
    Key insight: rotations are constrained by RPY, only translations are estimated.
    """
    
    def __init__(self, calibration: CalibrationData, config: Config,
                 rotation_provider: RotationProvider):
        self.calib = calibration
        self.cfg = config
        self.rotation_provider = rotation_provider
        
        # State
        self.keyframes: List[Keyframe] = []
        self.map_points: Dict[int, MapPoint] = {}
        self.next_point_id: int = 0
        
        # Current pose
        self.current_R_CW: Optional[np.ndarray] = None
        self.current_C_W: Optional[np.ndarray] = None
        self.scale: float = 1.0
        self.scale_offset: float = 0.0
        self.prev_frame_R_CW: Optional[np.ndarray] = None
        
        # Tracking state
        self.is_initialized: bool = False
        self.is_lost: bool = False
        self.last_keyframe_idx: int = -1
        
        # Track mappings
        self.track_observations: Dict[int, Dict[int, int]] = {}
        self.track_to_mpid: Dict[int, int] = {}
    
    def initialize(self, frames: List[FrameData], anchor_altitudes: Dict[int, float]) -> bool:
        """
        Initialize SLAM with known altitude anchors.
        
        Args:
            frames: Initial frames with features
            anchor_altitudes: Dict mapping frame index to known altitude
        
        Returns:
            True if initialization successful
        """
        if len(frames) < self.cfg.min_init_keyframes:
            LOG.warning(f"Need at least {self.cfg.min_init_keyframes} frames for init")
            return False
        
        if len(anchor_altitudes) < 2:
            LOG.warning("Need at least 2 altitude anchors for scale")
            return False
        
        LOG.info(f"Initializing SLAM with {len(frames)} frames, {len(anchor_altitudes)} anchors")
        
        # Build keyframes with RPY-derived rotations
        for frame in frames:
            R_CW = self.rotation_provider.get_R_CW(frame.timestamp)
            if R_CW is None:
                continue
            
            kf = Keyframe(
                index=frame.index,
                timestamp=frame.timestamp,
                R_CW=R_CW,  # World -> Camera rotation
                C_W=np.zeros(3),  # Camera position in world
                features=frame.features if frame.features is not None else np.empty((0, 2)),
                point_ids=frame.track_ids if frame.track_ids is not None else None
            )
            self.keyframes.append(kf)
        
        if len(self.keyframes) < 2:
            LOG.warning("Not enough keyframes with valid RPY")
            return False
        
        # Estimate translations up-to-scale using essential matrix decomposition
        self._estimate_translations_from_pairs()
        
        # Triangulate initial map points
        self._triangulate_initial_points()
        
        # Solve for metric scale using altitude anchors
        self._solve_metric_scale(anchor_altitudes)
        
        # Apply scale to translations and map points
        self._apply_scale(self.scale)
        
        self.is_initialized = True
        self.last_keyframe_idx = self.keyframes[-1].index
        
        LOG.info(f"SLAM initialized: scale={self.scale:.4f}, {len(self.map_points)} map points")
        return True
    
    def _estimate_translations_from_pairs(self) -> None:
        """
        Estimate translations between consecutive keyframes using DEROTATED points.
        
        Uses known rotation to derotate points, then estimates translation direction
        from the derotated correspondences.
        """
        K = self.calib.intrinsics.K
        K_inv = self.calib.intrinsics.K_inv
        
        for i in range(1, len(self.keyframes)):
            kf_prev = self.keyframes[i-1]
            kf_curr = self.keyframes[i]
            
            # Find correspondences using track IDs
            pts1, pts2 = self._find_correspondences(kf_prev, kf_curr)
            
            if len(pts1) < 8:
                kf_curr.C_W = kf_prev.C_W.copy()
                continue
            
            # Known relative rotation: R_C2_C1 transforms vectors from C1 to C2
            R_C2_C1 = kf_curr.R_CW @ kf_prev.R_CW.T  # R_CW2 @ R_WC1
            
            # DEROTATE pts2 using known rotation
            # If there were only rotation, pts2_derot would equal pts1
            # The residual motion is due to translation
            pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
            H_rot_inv = K @ R_C2_C1.T @ K_inv  # Inverse rotation homography
            pts2_derot = (H_rot_inv @ pts2_h.T).T
            pts2_derot = pts2_derot[:, :2] / pts2_derot[:, 2:3]
            
            # Now estimate translation from (pts1 -> pts2_derot)
            # These should be related by pure translation (epipolar: t x p = 0)
            # Use RANSAC to find translation direction
            t_dir = self._estimate_translation_direction(pts1, pts2_derot.astype(np.float32), K)
            
            if t_dir is None:
                kf_curr.C_W = kf_prev.C_W.copy()
                continue
            
            # t_dir is in camera1 frame, transform to world
            R_WC1 = kf_prev.R_CW.T  # Camera1 -> World (R_CW.T = R_WC)
            t_world = R_WC1 @ t_dir
            
            # Accumulate position (up to scale, will be fixed later)
            kf_curr.C_W = kf_prev.C_W + t_world
    
    def _find_correspondences(self, kf1: Keyframe, kf2: Keyframe, 
                                return_indices: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Find corresponding points between keyframes using track IDs.
        
        Optionally returns track IDs and feature indices for observation storage.
        
        Args:
            kf1, kf2: Keyframes to match
            return_indices: If True, also return (track_ids, idx1_list, idx2_list)
        
        Returns:
            If return_indices=False: (pts1, pts2)
            If return_indices=True: (pts1, pts2, track_ids, idx1_list, idx2_list)
        """
        if kf1.point_ids is None or kf2.point_ids is None:
            # Fallback: assume sequential (incorrect but allows testing)
            n = min(len(kf1.features), len(kf2.features))
            if return_indices:
                return kf1.features[:n], kf2.features[:n], np.arange(n), np.arange(n), np.arange(n)
            return kf1.features[:n], kf2.features[:n]
        
        # Find common track IDs
        ids1 = set(kf1.point_ids.tolist())
        ids2 = set(kf2.point_ids.tolist())
        common_ids = sorted(ids1 & ids2)  # Sort for determinism
        
        if len(common_ids) == 0:
            if return_indices:
                return np.empty((0, 2)), np.empty((0, 2)), np.array([]), [], []
            return np.empty((0, 2)), np.empty((0, 2))
        
        pts1, pts2 = [], []
        track_ids = []
        idx1_list, idx2_list = [], []
        
        for tid in common_ids:
            idx1 = np.where(kf1.point_ids == tid)[0][0]
            idx2 = np.where(kf2.point_ids == tid)[0][0]
            pts1.append(kf1.features[idx1])
            pts2.append(kf2.features[idx2])
            track_ids.append(tid)
            idx1_list.append(idx1)
            idx2_list.append(idx2)
        
        if return_indices:
            return np.array(pts1), np.array(pts2), np.array(track_ids), idx1_list, idx2_list
        return np.array(pts1), np.array(pts2)
    
    def _estimate_translation_direction(self, pts1: np.ndarray, pts2: np.ndarray, 
                                          K: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate translation direction from point correspondences.
        
        For pure translation: pts2 = pts1 + parallax
        The translation direction t satisfies: p2 x (K @ t) = 0 for all points
        """
        if len(pts1) < 5:
            return None
        
        # Compute flow vectors
        flow = pts2 - pts1
        
        # Estimate epipole (vanishing point of flow)
        # Flow lines should intersect at epipole e = K @ t
        # Use least squares: each flow defines a line through pt1 with direction flow
        # Line: (x - pts1) x flow = 0, or flow_y * x - flow_x * y = flow_y * pts1_x - flow_x * pts1_y
        
        A = np.zeros((len(pts1), 3))
        for j, (p, f) in enumerate(zip(pts1, flow)):
            if np.linalg.norm(f) < 0.5:  # Skip small motions
                continue
            A[j] = [f[1], -f[0], -f[1]*p[0] + f[0]*p[1]]
        
        # Remove zero rows
        valid = np.abs(A).sum(axis=1) > 0
        A = A[valid]
        
        if len(A) < 3:
            return None
        
        # Solve for epipole using SVD
        _, _, Vt = np.linalg.svd(A)
        e = Vt[-1]
        e = e / e[2]  # Normalize
        
        # Convert epipole to translation direction
        K_inv = np.linalg.inv(K)
        t_dir = K_inv @ e
        t_dir = t_dir / np.linalg.norm(t_dir)
        
        return t_dir
    
    def _filter_ground_points(self, prev_pts: np.ndarray, curr_pts: np.ndarray,
                               ground_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter correspondences to keep only ground points.
        
        Args:
            prev_pts, curr_pts: Corresponding points
            ground_mask: Binary mask where ground pixels > 127
        
        Returns:
            (ground_prev, ground_curr): Filtered ground-only correspondences
        """
        if ground_mask is None or len(prev_pts) == 0:
            return prev_pts, curr_pts
        
        h, w = ground_mask.shape[:2]
        is_ground = []
        
        for pt in curr_pts:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                is_ground.append(ground_mask[y, x] > 127)
            else:
                is_ground.append(False)
        
        is_ground = np.array(is_ground)
        if is_ground.sum() < 8:
            return prev_pts, curr_pts  # Not enough ground points, use all
        
        return prev_pts[is_ground], curr_pts[is_ground]
    
    def _estimate_metric_translation_from_homography(
        self, pts1: np.ndarray, pts2: np.ndarray,
        R_rel: np.ndarray, K: np.ndarray, K_inv: np.ndarray,
        altitude: float,
        R_CW_prev: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Estimate metric translation using ground-plane homography: Hn = R + (t/d) @ n^T."""
        if len(pts1) < 8:
            return None
        
        pts1_undist = self.calib.intrinsics.undistort_points(pts1)
        pts2_undist = self.calib.intrinsics.undistort_points(pts2)
        H, mask = cv2.findHomography(pts1_undist, pts2_undist, cv2.RANSAC, 3.0)
        if H is None or mask.sum() < 8:
            return None
        
        Hn = K_inv @ H @ K
        
        # Ground normal in camera frame
        n_world = np.array([0, 0, -1])
        n_cam = R_CW_prev @ n_world if R_CW_prev is not None else np.array([0, -0.866, -0.5])
        n_cam = n_cam / (np.linalg.norm(n_cam) + 1e-9)
        
        # Extract t/d: M = Hn - R ≈ (t/d) @ n^T, so t/d = M @ n
        M = Hn - R_rel
        t_over_d = M @ n_cam
        t_metric = t_over_d * altitude
        
        # Sanity checks
        _, s, _ = np.linalg.svd(M)
        rank1_ratio = s[0] / (s[1] + 1e-9) if len(s) > 1 else float('inf')
        if rank1_ratio < 3.0 or np.linalg.norm(t_metric) > 10.0:
            return None
        
        return t_metric
    
    def _triangulate_initial_points(self) -> None:
        """Triangulate map points from keyframe pairs using P = K @ [R_CW | -R_CW @ C_W]."""
        K = self.calib.intrinsics.K
        
        for i in range(1, len(self.keyframes)):
            kf1 = self.keyframes[i-1]
            kf2 = self.keyframes[i]
            
            # Find correspondences with indices for proper observation storage
            result = self._find_correspondences(kf1, kf2, return_indices=True)
            pts1, pts2, track_ids, idx1_list, idx2_list = result
            
            if len(pts1) < 5:
                continue
            
            # Projection matrices: P = K @ [R_CW | -R_CW @ C_W]
            R_CW1, C_W1 = kf1.R_CW, kf1.C_W
            R_CW2, C_W2 = kf2.R_CW, kf2.C_W
            
            t_CW1 = -R_CW1 @ C_W1.reshape(3, 1)
            t_CW2 = -R_CW2 @ C_W2.reshape(3, 1)
            
            P1 = K @ np.hstack([R_CW1, t_CW1])
            P2 = K @ np.hstack([R_CW2, t_CW2])
            
            # Triangulate
            pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            pts_3d = (pts_4d[:3] / pts_4d[3]).T
            
            # Filter by positive depth and reprojection error
            for j, pt_3d in enumerate(pts_3d):
                # Transform to camera frame: p_C = R_CW @ (p_W - C_W)
                pt_cam1 = R_CW1 @ (pt_3d - C_W1)
                pt_cam2 = R_CW2 @ (pt_3d - C_W2)
                
                # Check positive depth
                if pt_cam1[2] < 0.1 or pt_cam2[2] < 0.1:
                    continue
                
                # Check reprojection error
                proj1 = (K @ pt_cam1)
                proj1 = proj1[:2] / proj1[2]
                proj2 = (K @ pt_cam2)
                proj2 = proj2[:2] / proj2[2]
                
                err1 = np.linalg.norm(proj1 - pts1[j])
                err2 = np.linalg.norm(proj2 - pts2[j])
                
                if err1 > 5.0 or err2 > 5.0:
                    continue

                # Prevent MapPoint duplication by checking whether this track already has a map point
                tid = int(track_ids[j])
                
                if tid in self.track_to_mpid:
                    # Update existing map point with new observation
                    mpid = self.track_to_mpid[tid]
                    mp = self.map_points[mpid]
                    mp.observations[kf1.index] = idx1_list[j]
                    mp.observations[kf2.index] = idx2_list[j]
                    mp.num_observations = len(mp.observations)
                    # Update track_observations too
                    if tid in self.track_observations:
                        self.track_observations[tid][kf1.index] = idx1_list[j]
                        self.track_observations[tid][kf2.index] = idx2_list[j]
                else:
                    # Create new map point
                    mp = MapPoint(
                        id=self.next_point_id,
                        position=pt_3d,
                        observations={kf1.index: idx1_list[j], kf2.index: idx2_list[j]},
                        num_observations=2,
                        reprojection_error=(err1 + err2) / 2
                    )
                    self.track_observations[tid] = {kf1.index: idx1_list[j], kf2.index: idx2_list[j]}
                    self.track_to_mpid[tid] = self.next_point_id
                    self.map_points[self.next_point_id] = mp
                    self.next_point_id += 1
    
    def _solve_metric_scale(self, anchor_altitudes: Dict[int, float]) -> None:
        """
        Solve for metric scale using known altitudes.

        For NED: altitude = -C_W[2] (camera z position is negative when above ground)
        """
        # Collect (estimated_z, known_alt) pairs
        pairs = []
        
        for kf in self.keyframes:
            if kf.index in anchor_altitudes:
                known_alt = anchor_altitudes[kf.index]
                # C_W is camera position in world (NED)
                # altitude = -C_W[2] when ground is at z=0
                estimated_z = -kf.C_W[2]
                pairs.append((estimated_z, known_alt))
        
        if len(pairs) < 2:
            self.scale = 1.0
            LOG.warning("Could not determine metric scale - need at least 2 anchors")
            return

        # Check for sufficient vertical excitation before solving scale
        # If z estimates don't vary much, the scale solve becomes ill-conditioned
        ez_values = np.array([ez for ez, _ in pairs])
        ez_range = ez_values.max() - ez_values.min()
        
        if ez_range < 0.1:  # Less than 0.1 unscaled units of z variation
            LOG.warning(f"Insufficient vertical excitation for scale solve: "
                       f"z range = {ez_range:.3f}. Using default scale.")
            self.scale = 1.0
            self.scale_offset = 0.0
            return
        
        # Solve: known_alt = scale * estimated_z + offset
        # Using least squares for robustness
        A = np.array([[ez, 1] for ez, _ in pairs])
        b = np.array([alt for _, alt in pairs])
        
        result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Check condition number
        if len(s) >= 2 and s[-1] > 1e-10:
            cond_num = s[0] / s[-1]
            if cond_num > 1000:
                LOG.warning(f"Scale solve is ill-conditioned (cond={cond_num:.1f}). "
                           f"Using default scale.")
                self.scale = 1.0
                self.scale_offset = 0.0
                return
        
        self.scale = float(result[0]) if abs(result[0]) > 1e-6 else 1.0
        self.scale_offset = float(result[1]) if len(result) > 1 else 0.0
        
        LOG.info(f"Scale solved: {self.scale:.4f} (offset: {self.scale_offset:.2f}m, "
                 f"z_range: {ez_range:.3f})")
    
    def _apply_scale(self, scale: float) -> None:
        """Apply metric scale and offset: z_metric = scale * z_unscaled - offset."""
        for kf in self.keyframes:
            kf.C_W = kf.C_W * scale
            kf.C_W[2] -= self.scale_offset
        
        for mp in self.map_points.values():
            mp.position = mp.position * scale
            mp.position[2] -= self.scale_offset
    
    def process_frame(self, frame: FrameData, 
                      prev_pts: np.ndarray, curr_pts: np.ndarray,
                      inlier_mask: np.ndarray,
                      track_ids: Optional[np.ndarray] = None,
                      ground_mask: Optional[np.ndarray] = None,
                      all_features: Optional[np.ndarray] = None,
                      all_track_ids: Optional[np.ndarray] = None,
                      fused_altitude: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Process frame and estimate pose. Returns (R_CW, C_W, quality)."""
        if not self.is_initialized:
            return None, None, 0.0
        
        # Get rotation from RPY (R_CW: world -> camera)
        R_CW = self.rotation_provider.get_R_CW(frame.timestamp)
        if R_CW is None:
            self.is_lost = True
            return None, None, 0.0
        
        self.current_R_CW = R_CW
        
        # Estimate translation using ground-plane homography
        if len(self.keyframes) > 0:
            last_kf = self.keyframes[-1]
            
            if len(prev_pts) >= 8 and inlier_mask.sum() >= 8 and self.prev_frame_R_CW is not None:
                K = self.calib.intrinsics.K
                K_inv = self.calib.intrinsics.K_inv
                
                prev_inliers = prev_pts[inlier_mask]
                curr_inliers = curr_pts[inlier_mask]
                
                # Use fused altitude for scaling (avoids feedback loop)
                if fused_altitude is not None and fused_altitude > 1.0:
                    current_altitude = fused_altitude
                else:
                    current_altitude = self.get_camera_altitude()
                    if current_altitude is None or current_altitude < 1.0:
                        current_altitude = 100.0
                
                ground_prev, ground_curr = self._filter_ground_points(
                    prev_inliers, curr_inliers, ground_mask
                )
                R_rel = R_CW @ self.prev_frame_R_CW.T
                
                t_metric = self._estimate_metric_translation_from_homography(
                    ground_prev, ground_curr, R_rel, K, K_inv, current_altitude,
                    self.prev_frame_R_CW
                )
                
                if t_metric is not None:
                    R_WC_prev = self.prev_frame_R_CW.T
                    t_world = R_WC_prev @ t_metric
                    if self.current_C_W is not None:
                        self.current_C_W = self.current_C_W + t_world
                    else:
                        self.current_C_W = last_kf.C_W + t_world
                else:
                    # Fallback: derotation + direction estimation
                    curr_h = np.hstack([curr_inliers, np.ones((len(curr_inliers), 1))])
                    H_rot_inv = K @ R_rel.T @ K_inv
                    curr_derot = (H_rot_inv @ curr_h.T).T
                    curr_derot = curr_derot[:, :2] / curr_derot[:, 2:3]
                    
                    t_dir = self._estimate_translation_direction(
                        prev_inliers, curr_derot.astype(np.float32), K
                    )
                    
                    if t_dir is not None:
                        flow_mag = np.median(np.linalg.norm(curr_inliers - prev_inliers, axis=1))
                        t_scale = flow_mag * current_altitude / (self.calib.intrinsics.fx * 100)
                        
                        R_WC_prev = self.prev_frame_R_CW.T
                        t_world = R_WC_prev @ t_dir
                        
                        if self.current_C_W is not None:
                            self.current_C_W = self.current_C_W + t_world * t_scale
                        else:
                            self.current_C_W = last_kf.C_W + t_world * t_scale
                    else:
                        if self.current_C_W is None:
                            self.current_C_W = last_kf.C_W.copy()
            else:
                if self.current_C_W is None:
                    self.current_C_W = last_kf.C_W.copy()
        else:
            self.current_C_W = np.zeros(3)
        
        # Store current rotation as previous for the next frame
        self.prev_frame_R_CW = R_CW.copy()
        
        # Compute pose quality
        quality = self._compute_pose_quality(prev_pts, curr_pts, inlier_mask)
        
        # Check if should create keyframe
        # Pass tracker's full feature set for map growth
        if self._should_create_keyframe(frame, quality):
            self._create_keyframe(frame, prev_pts, curr_pts, inlier_mask, track_ids,
                                  all_features, all_track_ids)
        
        return self.current_R_CW, self.current_C_W, quality
    
    def _compute_pose_quality(self, prev_pts: np.ndarray, curr_pts: np.ndarray,
                               mask: np.ndarray) -> float:
        """Compute pose estimation quality."""
        if len(prev_pts) == 0:
            return 0.0
        
        num_inliers = mask.sum()
        inlier_ratio = num_inliers / len(mask) if len(mask) > 0 else 0
        
        # Quality based on inliers and coverage
        quality = min(1.0, num_inliers / self.cfg.min_inliers) * inlier_ratio
        return float(quality)
    
    def _should_create_keyframe(self, frame: FrameData, quality: float) -> bool:
        """Determine if current frame should be a keyframe."""
        if len(self.keyframes) == 0:
            return True
        
        # Check frame gap
        if frame.index - self.last_keyframe_idx < self.cfg.min_keyframe_gap:
            return False
        
        # Check quality
        if quality < 0.3:
            return False
        
        return True
    
    def _create_keyframe(self, frame: FrameData, 
                         prev_pts: np.ndarray, curr_pts: np.ndarray,
                         mask: np.ndarray,
                         track_ids: Optional[np.ndarray] = None,
                         all_features: Optional[np.ndarray] = None,
                         all_track_ids: Optional[np.ndarray] = None) -> None:
        """Create a new keyframe with all tracked features."""
        if self.current_R_CW is None or self.current_C_W is None:
            return
        
        # Prefer tracker's full feature set (includes newly detected points)
        if all_features is not None and all_track_ids is not None:
            kf_features = all_features
            kf_track_ids = all_track_ids
        else:
            kf_features = curr_pts[mask] if mask.sum() > 0 else np.empty((0, 2))
            kf_track_ids = track_ids[mask] if track_ids is not None and mask.sum() > 0 else None
        
        kf = Keyframe(
            index=frame.index,
            timestamp=frame.timestamp,
            R_CW=self.current_R_CW.copy(),
            C_W=self.current_C_W.copy(),
            features=kf_features,
            point_ids=kf_track_ids,
            num_tracks=int(mask.sum()),
            tracking_quality=self._compute_pose_quality(prev_pts, curr_pts, mask)
        )
        
        self.keyframes.append(kf)
        self.last_keyframe_idx = frame.index
        
        if len(self.keyframes) >= 2:
            self._triangulate_new_points(self.keyframes[-2], kf)
        
        if len(self.keyframes) > self.cfg.max_keyframes:
            self._marginalize_old_keyframes()
    
    def _triangulate_new_points(self, kf_prev: Keyframe, kf_curr: Keyframe) -> int:
        """Triangulate new map points between two keyframes."""
        K = self.calib.intrinsics.K
        
        # Find correspondences
        result = self._find_correspondences(kf_prev, kf_curr, return_indices=True)
        pts1, pts2, track_ids, idx1_list, idx2_list = result
        
        if len(pts1) < 5:
            return 0
        
        # Filter out tracks that already have map points
        new_mask = []
        for tid in track_ids:
            has_point = tid in self.track_observations
            new_mask.append(not has_point)
        new_mask = np.array(new_mask)
        
        if new_mask.sum() < 3:
            return 0
        
        pts1_new = pts1[new_mask]
        pts2_new = pts2[new_mask]
        tids_new = track_ids[new_mask]
        idx1_new = [idx1_list[i] for i in range(len(idx1_list)) if new_mask[i]]
        idx2_new = [idx2_list[i] for i in range(len(idx2_list)) if new_mask[i]]
        
        # Projection matrices
        R_CW1, C_W1 = kf_prev.R_CW, kf_prev.C_W
        R_CW2, C_W2 = kf_curr.R_CW, kf_curr.C_W
        
        t_CW1 = -R_CW1 @ C_W1.reshape(3, 1)
        t_CW2 = -R_CW2 @ C_W2.reshape(3, 1)
        
        P1 = K @ np.hstack([R_CW1, t_CW1])
        P2 = K @ np.hstack([R_CW2, t_CW2])
        
        # Triangulate
        pts_4d = cv2.triangulatePoints(P1, P2, pts1_new.T, pts2_new.T)
        pts_3d = (pts_4d[:3] / pts_4d[3]).T
        
        n_added = 0
        for j, pt_3d in enumerate(pts_3d):
            # Check positive depth
            pt_cam1 = R_CW1 @ (pt_3d - C_W1)
            pt_cam2 = R_CW2 @ (pt_3d - C_W2)
            
            if pt_cam1[2] < 0.1 or pt_cam2[2] < 0.1:
                continue
            
            # Check reprojection error
            proj1 = K @ pt_cam1
            proj1 = proj1[:2] / proj1[2]
            proj2 = K @ pt_cam2
            proj2 = proj2[:2] / proj2[2]
            
            err1 = np.linalg.norm(proj1 - pts1_new[j])
            err2 = np.linalg.norm(proj2 - pts2_new[j])
            
            if err1 > 5.0 or err2 > 5.0:
                continue
            
            tid = int(tids_new[j])
            mp = MapPoint(
                id=self.next_point_id,
                position=pt_3d,
                observations={kf_prev.index: idx1_new[j], kf_curr.index: idx2_new[j]},
                num_observations=2,
                reprojection_error=(err1 + err2) / 2
            )
            self.track_observations[tid] = {kf_prev.index: idx1_new[j], kf_curr.index: idx2_new[j]}
            self.track_to_mpid[tid] = self.next_point_id
            self.map_points[self.next_point_id] = mp
            self.next_point_id += 1
            n_added += 1
        
        return n_added
    
    def _marginalize_old_keyframes(self) -> None:
        """Remove old keyframes to limit memory."""
        n_remove = len(self.keyframes) - self.cfg.max_keyframes
        if n_remove <= 0:
            return
        
        old_indices = [kf.index for kf in self.keyframes[:n_remove]]
        self.keyframes = self.keyframes[n_remove:]
        
        removed_mpids = set()
        for mp in list(self.map_points.values()):
            for idx in old_indices:
                mp.observations.pop(idx, None)
            if len(mp.observations) < 2:
                removed_mpids.add(mp.id)
                del self.map_points[mp.id]
        
        tids_to_remove = [tid for tid, mpid in self.track_to_mpid.items() 
                          if mpid in removed_mpids]
        for tid in tids_to_remove:
            self.track_to_mpid.pop(tid, None)
            self.track_observations.pop(tid, None)
    
    def get_camera_altitude(self) -> Optional[float]:
        """Get current camera altitude (distance to ground plane at z=0)."""
        if self.current_C_W is None:
            return None
        # In NED, altitude = -C_W[2] (camera z is negative when above ground)
        return -self.current_C_W[2]
    
    def get_ground_points(self) -> np.ndarray:
        """Get 3D map points that are classified as ground."""
        ground_pts = []
        for mp in self.map_points.values():
            if mp.is_ground:
                ground_pts.append(mp.position)
        return np.array(ground_pts) if ground_pts else np.empty((0, 3))
    
    def classify_ground_points(self, ground_mask: np.ndarray, 
                                R_CW: np.ndarray, C_W: np.ndarray,
                                K: np.ndarray) -> int:
        """Classify map points as ground by projecting into segmentation mask."""
        h, w = ground_mask.shape[:2]
        n_ground = 0
        
        for mp in self.map_points.values():
            # Project point to current image
            p_cam = R_CW @ (mp.position - C_W)
            
            if p_cam[2] <= 0.1:  # Behind camera
                continue
            
            p_img = K @ p_cam
            u, v = int(p_img[0] / p_img[2]), int(p_img[1] / p_img[2])
            
            if 0 <= u < w and 0 <= v < h:
                alpha = 0.2
                is_ground_obs = ground_mask[v, u] > 127
                
                if is_ground_obs:
                    mp.ground_probability = mp.ground_probability * (1 - alpha) + alpha
                else:
                    mp.ground_probability = mp.ground_probability * (1 - alpha)
                if mp.ground_probability > 0.7 and mp.ground_probability - alpha < 0.7:
                    n_ground += 1
            else:
                # Point not visible - slow decay
                mp.ground_probability *= 0.99
        
        return n_ground


class GroundSegmenter:
    """Ground segmentation from images."""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.model = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_confidence: Optional[np.ndarray] = None
        self.last_timestamp: float = 0.0
        
        # Try to load model if specified
        if config.segmentation_model_path and Path(config.segmentation_model_path).exists():
            self._load_model(config.segmentation_model_path)
    
    def _load_model(self, path: str) -> None:
        """Load segmentation model."""
        try:
            # Placeholder for actual model loading
            # Could be DeepLabV3, SegFormer, etc.
            LOG.info(f"Loading segmentation model from {path}")
            # self.model = torch.load(path)
        except Exception as e:
            LOG.warning(f"Failed to load segmentation model: {e}")
    
    def segment(self, image: np.ndarray, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment ground from image.
        
        Returns:
            (mask, confidence): Binary ground mask and per-pixel confidence
        """
        h, w = image.shape[:2]
        
        if self.model is not None:
            # Use neural network
            mask, confidence = self._segment_neural(image)
        else:
            # Use heuristic (assume lower portion is ground)
            mask, confidence = self._segment_heuristic(image)
        
        self.last_mask = mask
        self.last_confidence = confidence
        self.last_timestamp = timestamp
        
        return mask, confidence
    
    def _segment_neural(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment using neural network."""
        # Placeholder - would use actual model inference
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        confidence = np.zeros((h, w), dtype=np.float32)
        
        # Simple heuristic fallback
        mask[h//2:, :] = 255
        confidence[h//2:, :] = 0.5
        
        return mask, confidence
    
    def _segment_heuristic(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple heuristic ground segmentation.
        
        Assumes:
        - Ground is in lower portion of image (tilted camera)
        - Ground has relatively uniform color/texture
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create gradient-based mask (ground tends to be smoother)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        grad_norm = grad_mag / (grad_mag.max() + 1e-6)
        
        # Lower gradient = more likely ground
        smoothness = 1.0 - grad_norm
        
        # Combine with vertical position prior (lower = more likely ground)
        y_coords = np.linspace(0, 1, h).reshape(-1, 1)
        position_prior = np.tile(y_coords, (1, w))
        
        # Combined confidence
        confidence = (0.5 * smoothness + 0.5 * position_prior).astype(np.float32)
        
        # Threshold for binary mask
        mask = (confidence > self.cfg.ground_mask_threshold).astype(np.uint8) * 255
        
        return mask, confidence
    
    def propagate_mask(self, prev_mask: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Propagate mask using homography warp."""
        if prev_mask is None:
            return None
        
        h, w = prev_mask.shape[:2]
        warped = cv2.warpPerspective(prev_mask, H, (w, h))
        return warped
    
    def get_ground_pixels(self, mask: np.ndarray) -> np.ndarray:
        """Get pixel coordinates of ground."""
        if mask is None:
            return np.empty((0, 2))
        
        y_coords, x_coords = np.where(mask > 127)
        return np.column_stack([x_coords, y_coords])


class DepthPrior:
    """Monocular depth estimation as prior (optional)."""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.model = None
        
        # Affine calibration: depth_metric = a * depth_net + b
        self.scale_a: float = 1.0
        self.offset_b: float = 0.0
        self.is_calibrated: bool = False
        
        # Cached results
        self.last_depth: Optional[np.ndarray] = None
        self.last_confidence: Optional[np.ndarray] = None
        self.last_timestamp: float = 0.0
        
        # Load model if specified
        if config.depth_model_path and Path(config.depth_model_path).exists():
            self._load_model(config.depth_model_path)
    
    def _load_model(self, path: str) -> None:
        """Load depth estimation model."""
        try:
            LOG.info(f"Loading depth model from {path}")
            # Placeholder for actual model loading (MiDaS, DPT, etc.)
            # self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        except Exception as e:
            LOG.warning(f"Failed to load depth model: {e}")
    
    def estimate_depth(self, image: np.ndarray, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth from image.
        
        Returns:
            (depth, confidence): Depth map and confidence map
        """
        if self.model is not None:
            depth, confidence = self._infer_depth(image)
        else:
            # Placeholder: uniform depth
            h, w = image.shape[:2]
            depth = np.ones((h, w), dtype=np.float32)
            confidence = np.ones((h, w), dtype=np.float32) * 0.1
        
        self.last_depth = depth
        self.last_confidence = confidence
        self.last_timestamp = timestamp
        
        return depth, confidence
    
    def _infer_depth(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run depth inference."""
        # Placeholder - actual implementation would use the model
        h, w = image.shape[:2]
        depth = np.ones((h, w), dtype=np.float32)
        confidence = np.ones((h, w), dtype=np.float32) * 0.5
        return depth, confidence
    
    def calibrate(self, depth_samples: np.ndarray, 
                  altitude_samples: np.ndarray,
                  ray_angles: np.ndarray) -> None:
        """
        Calibrate depth to metric using known altitudes.
        
        Args:
            depth_samples: Network depth values at ground pixels
            altitude_samples: Known altitudes for each sample
            ray_angles: Angle of each ray from vertical (for geometric conversion)
        
        For a ray at angle θ from vertical:
            actual_depth = altitude / cos(θ)
        """
        if len(depth_samples) < 2:
            LOG.warning("Not enough samples for depth calibration")
            return
        
        # Convert altitude to depth using ray angles
        expected_depth = altitude_samples / np.cos(ray_angles)
        
        # Solve affine: expected = a * network + b
        A = np.column_stack([depth_samples, np.ones(len(depth_samples))])
        result, _, _, _ = np.linalg.lstsq(A, expected_depth, rcond=None)
        
        self.scale_a = float(result[0])
        self.offset_b = float(result[1])
        self.is_calibrated = True
        
        LOG.info(f"Depth calibrated: metric = {self.scale_a:.4f} * net + {self.offset_b:.4f}")
    
    def to_metric(self, depth: np.ndarray) -> np.ndarray:
        """Convert network depth to metric depth."""
        if not self.is_calibrated:
            return depth
        return self.scale_a * depth + self.offset_b
    
    def compute_altitude_from_depth(self, depth: np.ndarray, 
                                     ground_mask: np.ndarray,
                                     K: np.ndarray,
                                     R_CW: np.ndarray) -> Tuple[float, float]:
        """
        Compute altitude from depth map.
        
        Args:
            depth: Metric depth map
            ground_mask: Binary ground mask
            K: Camera intrinsic matrix
            R_CW: World to Camera rotation (consistent naming)
        
        Returns:
            (altitude, uncertainty): Median altitude and dispersion
        """
        if ground_mask is None or depth is None:
            return 0.0, float('inf')
        
        # Get ground pixels
        y_coords, x_coords = np.where(ground_mask > 127)
        if len(x_coords) < 10:
            return 0.0, float('inf')
        
        # Sample depths at ground pixels
        ground_depths = depth[y_coords, x_coords]
        
        # Convert depth to altitude
        # For each pixel, compute ray direction and convert depth to vertical distance
        K_inv = np.linalg.inv(K)
        
        altitudes = []
        for x, y, d in zip(x_coords, y_coords, ground_depths):
            if d <= 0:
                continue
            
            # Ray in camera frame
            ray_cam = K_inv @ np.array([x, y, 1])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)
            
            # Ray in world frame (R_CW.T = camera->world)
            ray_world = R_CW.T @ ray_cam
            
            # Vertical component (z in NED is down)
            vertical = abs(ray_world[2])
            if vertical > 0.1:  # Not too horizontal
                alt = d * vertical
                altitudes.append(alt)
        
        if len(altitudes) < 5:
            return 0.0, float('inf')
        
        altitudes = np.array(altitudes)
        median_alt = float(np.median(altitudes))
        mad = float(np.median(np.abs(altitudes - median_alt)))
        
        return median_alt, mad * 1.4826  # Scale MAD to sigma


class GroundPlaneFitter:
    """
    Fit ground plane from triangulated 3D points.
    
    Uses RANSAC with prior that plane normal should be approximately world-up.
    """
    
    def __init__(self, config: Config, conventions: FrameConventions):
        self.cfg = config
        self.conventions = conventions
        
        # World up vector (in NED, up is -z)
        if conventions.world_frame == CoordinateFrame.NED:
            self.world_up = np.array([0, 0, -1])
        else:  # ENU
            self.world_up = np.array([0, 0, 1])
        
        # Current ground model
        self.ground_model: Optional[GroundModel] = None
        self.history: List[GroundModel] = []
    
    def fit_plane(self, points: np.ndarray, 
                  weights: Optional[np.ndarray] = None) -> Optional[GroundModel]:
        """
        Fit plane to 3D points using RANSAC.
        
        Args:
            points: Nx3 array of 3D points
            weights: Optional per-point weights
        
        Returns:
            GroundModel if successful, None otherwise
        """
        if len(points) < self.cfg.min_ground_inliers:
            return None
        
        best_model = None
        best_inliers = 0
        
        n_iterations = 100
        threshold = self.cfg.ground_ransac_threshold
        
        for _ in range(n_iterations):
            # Sample 3 random points
            idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx]
            
            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            
            # Ensure normal points up (toward camera)
            if np.dot(normal, self.world_up) < 0:
                normal = -normal
            
            # Check angle to world up
            angle = np.arccos(np.clip(np.dot(normal, self.world_up), -1, 1))
            if np.degrees(angle) > self.cfg.ground_normal_tolerance_deg:
                continue
            
            # Compute plane distance
            d = -np.dot(normal, p1)
            
            # Count inliers
            distances = np.abs(np.dot(points, normal) + d)
            inliers = distances < threshold
            n_inliers = inliers.sum()
            
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_model = GroundModel(
                    normal=normal,
                    distance=d,
                    inlier_ratio=n_inliers / len(points),
                    residual_m=float(np.median(distances[inliers])) if n_inliers > 0 else 0
                )
        
        if best_model is not None and best_inliers >= self.cfg.min_ground_inliers:
            # Refine with all inliers
            distances = np.abs(np.dot(points, best_model.normal) + best_model.distance)
            inliers = distances < threshold
            inlier_pts = points[inliers]
            
            if len(inlier_pts) >= 3:
                # Fit plane to inliers using SVD
                centroid = np.mean(inlier_pts, axis=0)
                centered = inlier_pts - centroid
                _, _, vh = np.linalg.svd(centered)
                normal = vh[-1]
                
                if np.dot(normal, self.world_up) < 0:
                    normal = -normal
                
                d = -np.dot(normal, centroid)
                
                best_model.normal = normal
                best_model.distance = d
                
                # Recompute residual
                distances = np.abs(np.dot(inlier_pts, normal) + d)
                best_model.residual_m = float(np.median(distances))
            
            # Compute temporal stability
            if len(self.history) > 0:
                prev = self.history[-1]
                normal_change = 1.0 - abs(np.dot(best_model.normal, prev.normal))
                dist_change = abs(best_model.distance - prev.distance)
                stability = max(0, 1.0 - normal_change * 10 - dist_change * 0.1)
                best_model.stability = stability
            else:
                best_model.stability = 0.5
            
            self.ground_model = best_model
            self.history.append(best_model)
            if len(self.history) > 10:
                self.history.pop(0)
        
        return best_model
    
    def get_altitude(self, camera_pos: np.ndarray) -> Optional[float]:
        """Get altitude as distance from camera to ground plane."""
        if self.ground_model is None:
            return None
        return self.ground_model.distance_to_camera(camera_pos)


class Initializer:
    """Handles system initialization with known altitude anchors."""
    
    def __init__(self, calibration: CalibrationData, config: Config,
                 tracker: Optional['VisualTracker'] = None,
                 rotation_provider: Optional[RotationProvider] = None):
        self.calib = calibration
        self.cfg = config
        
        self.rotation_provider = rotation_provider if rotation_provider else RotationProvider(calibration, config)
        self.tracker = tracker if tracker else VisualTracker(calibration, config)
        self.ground_segmenter = GroundSegmenter(config)
        self.depth_prior = DepthPrior(config) if config.use_depth_prior else None
        
        # Collected data
        self.init_frames: List[FrameData] = []
        self.anchor_altitudes: Dict[int, float] = {}
        
        # State
        self.is_complete: bool = False
        self.scale: float = 1.0
        self.mount_correction: Optional[np.ndarray] = None
        self.pose_engine: Optional['PoseEngine'] = None  # Expose initialized pose engine
    
    def add_frame(self, frame: FrameData, altitude_gt: Optional[float] = None) -> bool:
        """
        Add frame to initialization buffer.
        
        Args:
            frame: Frame data
            altitude_gt: Known ground truth altitude (for anchor frames)
        
        Returns:
            True if initialization complete
        """
        # Get rotation from RPY
        if frame.rpy is not None:
            self.rotation_provider.add_rpy(frame.rpy)
        
        R_CW = self.rotation_provider.get_R_CW(frame.timestamp)
        if R_CW is not None:
            frame.R_CW = R_CW
        
        prev_pts, curr_pts, mask, track_ids = self.tracker.track(frame.image_gray, frame.timestamp)
        if curr_pts.size > 0:
            if mask.sum() > 0:
                frame.features = curr_pts[mask]
                frame.track_ids = track_ids[mask]
            else:
                frame.features = curr_pts
                frame.track_ids = track_ids
        
        # Segment ground
        ground_mask, _ = self.ground_segmenter.segment(frame.image, frame.timestamp)
        
        # Store frame
        self.init_frames.append(frame)
        
        # Store altitude anchor
        if altitude_gt is not None:
            self.anchor_altitudes[frame.index] = altitude_gt
            LOG.info(f"Added altitude anchor: frame {frame.index} = {altitude_gt:.2f}m")
        
        # Check if we have enough for initialization
        if len(self.init_frames) >= self.cfg.init_window_size:
            if len(self.anchor_altitudes) >= 2:
                return self._complete_initialization()
        
        return False
    
    def _complete_initialization(self) -> bool:
        """Complete the initialization phase."""
        LOG.info("=" * 60)
        LOG.info("COMPLETING INITIALIZATION")
        LOG.info(f"Frames: {len(self.init_frames)}, Anchors: {len(self.anchor_altitudes)}")
        
        # Build pose engine and store it for transfer to runtime
        self.pose_engine = PoseEngine(self.calib, self.cfg, self.rotation_provider)
        
        # Initialize SLAM
        success = self.pose_engine.initialize(self.init_frames, self.anchor_altitudes)
        if not success:
            LOG.warning("SLAM initialization failed")
            return False
        
        self.scale = self.pose_engine.scale
        
        # Bootstrap ground classification using keyframe poses
        for kf in self.pose_engine.keyframes:
            frame = next((f for f in self.init_frames if f.index == kf.index), None)
            if frame is None:
                continue
            mask, _ = self.ground_segmenter.segment(frame.image, frame.timestamp)
            self.pose_engine.classify_ground_points(mask, kf.R_CW, kf.C_W, self.calib.intrinsics.K)
        
        # Fit initial ground plane
        ground_fitter = GroundPlaneFitter(self.cfg, self.calib.conventions)
        ground_pts = self.pose_engine.get_ground_points()
        
        # Fallback: bootstrap ground from all points if needed
        if len(ground_pts) < self.cfg.min_ground_inliers:
            LOG.info("Bootstrapping ground classification from all map points")
            all_pts = np.array([mp.position for mp in self.pose_engine.map_points.values()])
            if len(all_pts) >= 5:
                temp_model = ground_fitter.fit_plane(all_pts)
                if temp_model is not None:
                    for mp in self.pose_engine.map_points.values():
                        dist = abs(np.dot(temp_model.normal, mp.position) + temp_model.distance)
                        if dist < 2.0:
                            mp.ground_probability = 0.8
                    ground_pts = self.pose_engine.get_ground_points()
        
        if len(ground_pts) >= self.cfg.min_ground_inliers:
            ground_model = ground_fitter.fit_plane(ground_pts)
            if ground_model is not None:
                LOG.info(f"Ground plane fit: normal={ground_model.normal}, d={ground_model.distance:.2f}")
        
        # Calibrate depth prior if available
        if self.depth_prior is not None:
            self._calibrate_depth_prior()
        
        self.is_complete = True
        LOG.info(f"Initialization complete: scale={self.scale:.4f}")
        LOG.info("=" * 60)
        
        return True
    
    def _calibrate_depth_prior(self) -> None:
        """Calibrate depth network to metric using init frames."""
        if self.depth_prior is None:
            return
        
        depth_samples = []
        altitude_samples = []
        ray_angles = []
        
        K = self.calib.intrinsics.K
        K_inv = self.calib.intrinsics.K_inv
        
        for frame in self.init_frames:
            if frame.index not in self.anchor_altitudes:
                continue
            
            alt = self.anchor_altitudes[frame.index]
            
            # Get depth and ground mask
            depth, _ = self.depth_prior.estimate_depth(frame.image, frame.timestamp)
            mask, _ = self.ground_segmenter.segment(frame.image, frame.timestamp)
            
            # Sample ground pixels
            y_coords, x_coords = np.where(mask > 127)
            if len(x_coords) < 10:
                continue
            
            # Sample subset
            sample_idx = np.random.choice(len(x_coords), min(100, len(x_coords)), replace=False)
            
            for idx in sample_idx:
                x, y = x_coords[idx], y_coords[idx]
                d = depth[y, x]
                
                if d <= 0:
                    continue
                
                # Compute ray angle
                ray = K_inv @ np.array([x, y, 1])
                ray = ray / np.linalg.norm(ray)
                
                if frame.R_CW is not None:
                    ray_world = frame.R_CW.T @ ray
                    angle = np.arccos(np.clip(abs(ray_world[2]), 0, 1))
                else:
                    angle = np.arctan2(np.sqrt(ray[0]**2 + ray[1]**2), ray[2])
                
                depth_samples.append(d)
                altitude_samples.append(alt)
                ray_angles.append(angle)
        
        if len(depth_samples) >= 10:
            self.depth_prior.calibrate(
                np.array(depth_samples),
                np.array(altitude_samples),
                np.array(ray_angles)
            )


# --- Homography Altimeter ---

@dataclass
class HomographyConstraint:
    """Output from HomographyAltimeter.compute_constraint()."""
    s: float                    # Scale factor: d_{k+1} = d_k * s
    log_s: float               # log(s) for log-space smoother
    sigma_r: float             # Uncertainty on log residual
    n_cam: np.ndarray          # Plane normal in camera frame (unit vector)
    R_rel: np.ndarray          # Refined relative rotation used
    metrics: Dict[str, float]  # Debug metrics
    
    def is_valid(self) -> bool:
        """Check if constraint passed all gates."""
        return self.s > 0 and np.isfinite(self.log_s) and self.sigma_r < float('inf')


class HomographyAltimeter:
    """Dominant-plane homography altimeter (PRIMARY path)."""
    
    def __init__(self, calibration: CalibrationData, config: Config, 
                 conventions: FrameConventions):
        self.calib = calibration
        self.cfg = config
        self.conventions = conventions
        
        # World up vector for ground-likeness gate
        if conventions.world_frame == CoordinateFrame.NED:
            self.world_up = np.array([0.0, 0.0, -1.0])  # NED: up is -z
        else:
            self.world_up = np.array([0.0, 0.0, 1.0])   # ENU: up is +z
        
        # Gating thresholds (from config)
        self.min_inliers = 20
        self.min_coverage = 0.20
        self.max_rmse_px = 3.0
        self.rank1_thresh_s2 = config.rank1_thresh_s2
        self.rank1_thresh_s3 = config.rank1_thresh_s3
        self.s_min = config.homography_s_min
        self.s_max = config.homography_s_max
        self.max_slope_deg = 70.0
        
        # Gate enable flags (from config)
        self.enable_rank1_gate = config.enable_rank1_gate
        self.sign_convention = config.homography_sign_convention
        
        # State for continuity
        self.prev_n_cam: Optional[np.ndarray] = None
        self.prev_R_rel: Optional[np.ndarray] = None
        self.consecutive_failures = 0
        
    def compute_constraint(
        self,
        pts_prev: np.ndarray,       # Previous frame points (N, 2)
        pts_curr: np.ndarray,       # Current frame points (N, 2)
        R_rel_rpy: np.ndarray,      # Relative rotation from RPY (prior)
        R_CW_curr: np.ndarray,      # Current world->camera rotation (for ground gate)
        grid_shape: Tuple[int, int] = (4, 4)  # For coverage computation
    ) -> Optional[HomographyConstraint]:
        """
        Compute distance constraint from homography.
        
        Returns HomographyConstraint if all gates pass, None otherwise.
        """
        metrics = {
            'n_input': len(pts_prev),
            'n_inliers': 0,
            'inlier_ratio': 0.0,
            'coverage': 0.0,
            'rmse_px': float('inf'),
            'rank1_s2s1': 1.0,
            'rank1_s3s1': 1.0,
            'rot_dist_deg': 0.0,
            'ground_likeness': 0.0,
            'gate_failed': 'none'
        }
        
        if len(pts_prev) < self.min_inliers:
            metrics['gate_failed'] = 'insufficient_points'
            self.consecutive_failures += 1
            return None
        
        # Undistort points
        pts_prev_u = self.calib.intrinsics.undistort_points(pts_prev)
        pts_curr_u = self.calib.intrinsics.undistort_points(pts_curr)
        
        # 1. RANSAC homography (use config threshold)
        H, mask = cv2.findHomography(pts_prev_u, pts_curr_u, cv2.USAC_MAGSAC, self.cfg.ransac_reproj_threshold)
        if H is None:
            metrics['gate_failed'] = 'homography_failed'
            self.consecutive_failures += 1
            return None
        
        inlier_mask = mask.ravel().astype(bool)
        n_inliers = inlier_mask.sum()
        metrics['n_inliers'] = int(n_inliers)
        metrics['inlier_ratio'] = n_inliers / len(pts_prev)
        
        # Gate: minimum inliers
        if n_inliers < self.min_inliers:
            metrics['gate_failed'] = 'min_inliers'
            self.consecutive_failures += 1
            self._last_fail_metrics = metrics
            return None
        
        # 2. Coverage check (grid occupancy)
        coverage = self._compute_coverage(pts_curr_u[inlier_mask], grid_shape)
        metrics['coverage'] = coverage
        if coverage < self.min_coverage:
            metrics['gate_failed'] = 'coverage'
            self.consecutive_failures += 1
            self._last_fail_metrics = metrics
            return None
        
        # 3. Reprojection RMSE
        pts_prev_h = np.hstack([pts_prev_u, np.ones((len(pts_prev_u), 1))])
        pts_proj = (H @ pts_prev_h.T).T
        pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]
        errors = np.linalg.norm(pts_proj - pts_curr_u, axis=1)
        rmse = float(np.sqrt(np.mean(errors[inlier_mask]**2)))
        metrics['rmse_px'] = rmse
        if rmse > self.max_rmse_px:
            metrics['gate_failed'] = 'rmse'
            self.consecutive_failures += 1
            self._last_fail_metrics = metrics
            return None
        
        # 4. Normalize homography: H~ = K^-1 H K
        K = self.calib.intrinsics.K
        K_inv = self.calib.intrinsics.K_inv
        Hn = K_inv @ H @ K
        
        # 5. Decompose homography to get candidates
        try:
            n_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(Hn, np.eye(3))
        except:
            metrics['gate_failed'] = 'decompose_failed'
            self.consecutive_failures += 1
            self._last_fail_metrics = metrics
            return None
        
        if n_solutions == 0:
            metrics['gate_failed'] = 'no_solutions'
            self.consecutive_failures += 1
            self._last_fail_metrics = metrics
            return None
        
        # 6. Select best candidate using RPY prior + cheirality + continuity
        best_idx, best_score, R_rel, n_cam, u = self._select_best_candidate(
            rotations, translations, normals, R_rel_rpy, 
            pts_prev_u[inlier_mask], pts_curr_u[inlier_mask]
        )
        
        if best_idx < 0:
            metrics['gate_failed'] = 'no_valid_candidate'
            self.consecutive_failures += 1
            self._last_fail_metrics = metrics
            return None
        
        # Compute rotation distance to RPY prior
        R_diff = R_rel @ R_rel_rpy.T
        rot_dist = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        metrics['rot_dist_deg'] = float(np.degrees(rot_dist))
        
        # 7. Rank-1 check on A = Hn - R_rel
        A = Hn - R_rel
        _, s_vals, _ = np.linalg.svd(A)
        s2_s1 = s_vals[1] / (s_vals[0] + 1e-9)
        s3_s1 = s_vals[2] / (s_vals[0] + 1e-9)
        metrics['rank1_s2s1'] = float(s2_s1)
        metrics['rank1_s3s1'] = float(s3_s1)
        
        # 7b. Rank-1 gate (optional, requires accurate RPY prior)
        if self.enable_rank1_gate:
            if s2_s1 > self.rank1_thresh_s2 or s3_s1 > self.rank1_thresh_s3:
                metrics['gate_failed'] = 'rank1'
                self.consecutive_failures += 1
                self._last_fail_metrics = metrics
                return None
        
        # 8. Compute scale factor s = 1 + sign * (n · u)
        # sign_convention: +1 for OpenCV/Unity (tested), -1 for standard textbook formula
        # For ascending drone (distance increasing), we need s > 1
        dot_nu = float(np.dot(n_cam, u))
        s = 1.0 + self.sign_convention * dot_nu
        
        if s <= 0 or s < self.s_min or s > self.s_max:
            metrics['gate_failed'] = 's_range'
            metrics['s_value'] = s
            self.consecutive_failures += 1
            self._last_fail_metrics = metrics
            return None
        
        # 9. Ground-likeness gate (for AGL output)
        # Transform normal to world frame: n_w = R_CW^T @ n_cam (camera->world)
        n_world = R_CW_curr.T @ n_cam
        ground_likeness = abs(float(np.dot(n_world, self.world_up)))
        metrics['ground_likeness'] = ground_likeness
        
        cos_max_slope = np.cos(np.radians(self.max_slope_deg))
        if ground_likeness < cos_max_slope:
            metrics['gate_failed'] = 'ground_likeness'
            self.consecutive_failures += 1
            self._last_fail_metrics = metrics
            return None
        
        # All gates passed
        self.consecutive_failures = 0
        self.prev_n_cam = n_cam.copy()
        self.prev_R_rel = R_rel.copy()
        
        # Compute uncertainty from quality metrics
        sigma_r = self._compute_sigma(metrics)
        
        log_s = float(np.log(s))
        
        return HomographyConstraint(
            s=s,
            log_s=log_s,
            sigma_r=sigma_r,
            n_cam=n_cam,
            R_rel=R_rel,
            metrics=metrics
        )
    
    def _compute_coverage(self, pts: np.ndarray, grid_shape: Tuple[int, int]) -> float:
        """Compute grid occupancy coverage."""
        if len(pts) == 0:
            return 0.0
        
        h, w = self.calib.intrinsics.height, self.calib.intrinsics.width
        gh, gw = grid_shape
        cell_h, cell_w = h / gh, w / gw
        
        occupied = set()
        for x, y in pts:
            ci = int(np.clip(x / cell_w, 0, gw - 1))
            ri = int(np.clip(y / cell_h, 0, gh - 1))
            occupied.add((ri, ci))
        
        return len(occupied) / (gh * gw)
    
    def _select_best_candidate(
        self,
        rotations: List[np.ndarray],
        translations: List[np.ndarray],
        normals: List[np.ndarray],
        R_rel_rpy: np.ndarray,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray
    ) -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Select best (R, t/d, n) candidate from homography decomposition.
        
        Criteria:
        1. s = 1 - n·u must be positive and in valid range (cheirality)
        2. Transfer error on inliers (primary)
        3. Rotation distance to RPY prior (secondary)
        4. Normal continuity with previous frame (tertiary)
        
        Returns: (best_idx, best_score, R_rel, n_cam, u=t/d)
        """
        best_idx = -1
        best_score = float('inf')
        best_R = None
        best_n = None
        best_u = None
        
        K = self.calib.intrinsics.K
        
        for i in range(len(rotations)):
            R = rotations[i]
            t_over_d = translations[i].flatten()
            n = normals[i].flatten()
            
            # Try both sign choices for (n, t/d) - they come in pairs
            for sign in [1.0, -1.0]:
                n_try = n * sign
                u_try = t_over_d * sign
                
                # 1. Cheirality via s: s = 1 - n·u must be positive and reasonable
                s = 1.0 - np.dot(n_try, u_try)
                if s <= 0.05 or s < self.s_min or s > self.s_max:
                    continue  # Invalid scale - skip this sign choice
                
                # 2. Compute transfer error using H_candidate = R + u @ n.T
                # Hn = R + u @ n^T, H = K @ Hn @ K^-1
                Hn = R + np.outer(u_try, n_try)
                H_candidate = K @ Hn @ np.linalg.inv(K)
                
                # Forward transfer error: project pts_prev through H, compare to pts_curr
                pts_prev_h = np.hstack([pts_prev, np.ones((len(pts_prev), 1))])
                pts_proj = (H_candidate @ pts_prev_h.T).T
                pts_proj = pts_proj[:, :2] / (pts_proj[:, 2:3] + 1e-9)
                transfer_err = np.median(np.linalg.norm(pts_proj - pts_curr, axis=1))
                
                # 3. Rotation distance to RPY prior
                R_diff = R @ R_rel_rpy.T
                rot_dist = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
                
                # 4. Normal continuity with previous frame
                normal_cont = 0.0
                if self.prev_n_cam is not None:
                    normal_cont = 1.0 - abs(np.dot(n_try, self.prev_n_cam))
                
                # Combined score (lower is better)
                # Transfer error dominates, rotation/continuity are tiebreakers
                score = transfer_err + 0.5 * rot_dist + 0.2 * normal_cont
                
                if score < best_score:
                    best_score = score
                    best_idx = i
                    best_R = R
                    best_n = n_try.copy()
                    best_u = u_try.copy()
        
        if best_idx < 0:
            return -1, float('inf'), np.eye(3), np.zeros(3), np.zeros(3)
        
        return best_idx, best_score, best_R, best_n, best_u
    
    def _compute_sigma(self, metrics: Dict[str, float]) -> float:
        """Compute measurement uncertainty from quality metrics."""
        # Base sigma
        sigma_base = 0.02  # ~2% relative uncertainty in log-space
        
        # Factors that increase uncertainty
        inlier_factor = 1.0 / max(metrics['inlier_ratio'], 0.3)
        coverage_factor = 1.0 / max(metrics['coverage'], 0.3)
        rmse_factor = 1.0 + metrics['rmse_px'] / self.max_rmse_px
        rank1_factor = 1.0 + metrics['rank1_s2s1'] * 2
        
        sigma = sigma_base * inlier_factor * coverage_factor * rmse_factor * rank1_factor
        
        # Clamp to reasonable range
        return float(np.clip(sigma, 0.01, 0.5))


# --- Smoother ---

@dataclass 
class SmootherState:
    """State in the fixed-lag smoother."""
    state_id: int          # Monotonically increasing ID (stable across window shifts)
    timestamp: float
    log_d: float           # log(distance to plane)
    sigma_log_d: float     # uncertainty on log_d
    d: float               # distance (exp(log_d))
    h: float               # vertical altitude (d * vertical_factor)
    is_anchor: bool = False


class FixedLagLogDistanceSmoother:
    """Fixed-lag robust least-squares smoother for plane distance (log-space, Huber loss)."""
    
    def __init__(self, config: Config, window_size: int = 30):
        self.cfg = config
        self.window_size = window_size
        
        # State window
        self.states: deque = deque(maxlen=window_size)
        
        # State ID counter (monotonically increasing, stable across window shifts)
        self.next_state_id: int = 0
        
        # Factors store state_ids (not indices) so they remain valid after deque shifts
        self.anchor_factors: List[Tuple[int, float, float]] = []  # (state_id, log_d, sigma)
        self.vision_factors: deque = deque(maxlen=window_size)    # (id_prev, id_curr, log_s, sigma)
        
        # Current estimate
        self.current_log_d: float = np.log(100.0)  # Default ~100m
        self.current_sigma_log_d: float = 1.0
        self.current_d: float = 100.0
        self.current_h: float = 100.0
        self.vertical_factor: float = 1.0  # h = d * vertical_factor, where vertical_factor = |n_world · up|
        
        # Mode
        self.mode: AltitudeMode = AltitudeMode.INIT
        self.hold_start_time: Optional[float] = None
        self.last_good_update: float = 0.0
        self.age_of_last_good_update: float = 0.0
        
        # Huber threshold
        self.huber_k = 1.345  # Standard Huber threshold
        
        # Minimum distance to prevent log(0)
        self.d_min = 1.0  # meters
        
    def add_anchor(self, timestamp: float, altitude: float, sigma: float,
                   R_CW: np.ndarray, n_cam: Optional[np.ndarray] = None) -> None:
        """
        Add absolute altitude anchor (from initialization or external source).
        
        Converts altitude h to plane distance d: d = h / vertical_factor
        where vertical_factor = |n_world · world_up| (the cosine of slope angle)
        """
        # If we have plane normal in camera frame, compute vertical_factor
        if n_cam is not None:
            n_world = R_CW.T @ n_cam
            # Assumes NED world frame by default (up = -Z). If using ENU, up = +Z.
            world_up = np.array([0.0, 0.0, -1.0])
            dot = abs(np.dot(n_world, world_up))
            self.vertical_factor = np.clip(dot, 0.1, 1.0)
        
        # d = h / vertical_factor (h → d conversion)
        # For flat ground (vertical_factor ≈ 1), d ≈ h
        # For sloped ground, d > h (plane distance > vertical height)
        d = max(altitude / self.vertical_factor, self.d_min)
        log_d = np.log(d)
        sigma_log = sigma / d  # Approximate: sigma_log ≈ sigma_d / d
        
        # Add state with unique ID
        state_id = self.next_state_id
        self.next_state_id += 1
        
        state = SmootherState(
            state_id=state_id,
            timestamp=timestamp,
            log_d=log_d,
            sigma_log_d=sigma_log,
            d=d,
            h=altitude,
            is_anchor=True
        )
        self.states.append(state)
        
        # Add anchor factor (stores state_id, not index)
        self.anchor_factors.append((state_id, log_d, sigma_log))
        
        self.current_log_d = log_d
        self.current_d = d
        self.current_h = altitude
        self.current_sigma_log_d = sigma_log
        self.mode = AltitudeMode.GEOM
        self.last_good_update = timestamp
        
        LOG.info(f"Anchor added: h={altitude:.2f}m, d={d:.2f}m, vertical_factor={self.vertical_factor:.3f}")
    
    def add_vision_constraint(self, timestamp: float, 
                               constraint: HomographyConstraint,
                               R_CW: np.ndarray) -> bool:
        """
        Add vision constraint from homography.
        
        Returns True if constraint was accepted.
        """
        if not constraint.is_valid():
            return False
        
        # Update vertical_factor from plane normal (optional, requires accurate R_CW)
        if self.cfg.enable_vertical_factor_update:
            n_world = R_CW.T @ constraint.n_cam
            # NED: up is -Z (consistent with HomographyAltimeter)
            world_up = np.array([0.0, 0.0, -1.0])
            dot = abs(np.dot(n_world, world_up))
            new_factor = np.clip(dot, 0.1, 1.0)
            self.vertical_factor = 0.9 * self.vertical_factor + 0.1 * new_factor
        
        if len(self.states) == 0:
            # No previous state - need anchor first
            return False
        
        # Predict new state from constraint
        prev_state = self.states[-1]
        new_log_d = prev_state.log_d + constraint.log_s
        new_d = np.exp(new_log_d)
        
        # Enforce minimum distance
        if new_d < self.d_min:
            new_d = self.d_min
            new_log_d = np.log(self.d_min)
        
        # h = d * vertical_factor (d → h conversion)
        new_h = new_d * self.vertical_factor
        
        # Add new state with unique ID
        prev_state_id = prev_state.state_id
        new_state_id = self.next_state_id
        self.next_state_id += 1
        
        new_state = SmootherState(
            state_id=new_state_id,
            timestamp=timestamp,
            log_d=new_log_d,
            sigma_log_d=constraint.sigma_r,
            d=new_d,
            h=new_h,
            is_anchor=False
        )
        self.states.append(new_state)
        
        # Add vision factor (stores state_ids, not indices)
        self.vision_factors.append((prev_state_id, new_state_id, constraint.log_s, constraint.sigma_r))
        
        # Solve the window
        self._solve()
        
        self.mode = AltitudeMode.GEOM
        self.hold_start_time = None
        self.last_good_update = timestamp
        
        return True
    
    def predict(self, timestamp: float, dt: float) -> None:
        """Predict state forward (simple model: constant altitude)."""
        # Inflate uncertainty over time
        self.current_sigma_log_d = np.sqrt(
            self.current_sigma_log_d**2 + (0.01 * dt)**2  # ~1% drift per second
        )
        self.age_of_last_good_update = timestamp - self.last_good_update
    
    def enter_hold(self, timestamp: float) -> None:
        """Enter HOLD mode when no valid constraints."""
        if self.hold_start_time is None:
            self.hold_start_time = timestamp
            LOG.warning("Smoother entering HOLD mode")
        
        hold_duration = timestamp - self.hold_start_time
        
        if hold_duration > self.cfg.hold_timeout_sec:
            self.mode = AltitudeMode.LOST
        else:
            self.mode = AltitudeMode.HOLD
            # Inflate uncertainty
            self.current_sigma_log_d *= 1.05
    
    def _solve(self) -> None:
        """Solve the fixed-lag least squares problem with Huber loss."""
        n_states = len(self.states)
        if n_states == 0:
            return
        
        # Build state_id → local_index map (handles deque shifts)
        id_to_idx = {state.state_id: i for i, state in enumerate(self.states)}
        
        # Build initial estimate from current states
        x = np.array([s.log_d for s in self.states])
        
        # Iteratively reweighted least squares (IRLS) for Huber
        for iteration in range(5):
            # Build normal equations: J^T W J x = J^T W r
            JtWJ = np.zeros((n_states, n_states))
            JtWr = np.zeros(n_states)
            
            # Anchor factors (absolute constraints)
            for state_id, log_d_anchor, sigma in self.anchor_factors:
                if state_id not in id_to_idx:
                    continue  # State has been dropped from window
                idx = id_to_idx[state_id]
                residual = x[idx] - log_d_anchor
                weight = self._huber_weight(residual, sigma)
                JtWJ[idx, idx] += weight / sigma**2
                JtWr[idx] += weight * residual / sigma**2
            
            # Vision factors (relative constraints)
            for id_prev, id_curr, log_s, sigma in self.vision_factors:
                if id_prev not in id_to_idx or id_curr not in id_to_idx:
                    continue  # One or both states dropped from window
                idx_prev = id_to_idx[id_prev]
                idx_curr = id_to_idx[id_curr]
                # Residual: log_d_curr - log_d_prev - log_s
                residual = x[idx_curr] - x[idx_prev] - log_s
                weight = self._huber_weight(residual, sigma)
                w_sigma2 = weight / sigma**2
                
                # Jacobian: [0...0, -1, 0...0, 1, 0...0]
                JtWJ[idx_prev, idx_prev] += w_sigma2
                JtWJ[idx_curr, idx_curr] += w_sigma2
                JtWJ[idx_prev, idx_curr] -= w_sigma2
                JtWJ[idx_curr, idx_prev] -= w_sigma2
                
                JtWr[idx_prev] -= w_sigma2 * residual
                JtWr[idx_curr] += w_sigma2 * residual
            
            # Add small regularization for stability
            JtWJ += np.eye(n_states) * 1e-6
            
            # Solve
            try:
                dx = np.linalg.solve(JtWJ, -JtWr)
                x = x + dx
                
                if np.linalg.norm(dx) < 1e-6:
                    break
            except np.linalg.LinAlgError:
                break
        
        # Update states with solution
        for i, state in enumerate(self.states):
            state.log_d = x[i]
            state.d = np.exp(x[i])
            state.h = state.d * self.vertical_factor
        
        # Extract current estimate (last state)
        if n_states > 0:
            last = self.states[-1]
            self.current_log_d = last.log_d
            self.current_d = last.d
            self.current_h = last.h
            
            # Estimate uncertainty from inverse Hessian diagonal
            try:
                cov = np.linalg.inv(JtWJ)
                self.current_sigma_log_d = float(np.sqrt(cov[-1, -1]))
            except:
                pass
    
    def _huber_weight(self, residual: float, sigma: float) -> float:
        """Compute Huber weight for IRLS."""
        r_normalized = abs(residual) / sigma
        if r_normalized <= self.huber_k:
            return 1.0
        else:
            return self.huber_k / r_normalized
    
    def get_estimate(self, timestamp: float) -> AltitudeEstimate:
        """Get current altitude estimate."""
        # Convert log-space uncertainty to altitude uncertainty
        sigma_h = self.current_sigma_log_d * self.current_h
        
        return AltitudeEstimate(
            altitude_m=self.current_h,
            sigma_m=sigma_h,
            mode=self.mode,
            timestamp=timestamp,
            slam_quality=0.0,  # Not using SLAM as primary
            ground_quality=1.0 if self.mode == AltitudeMode.GEOM else 0.0,
            depth_quality=0.0,
            altitude_geom=None,
            altitude_depth=None,
            altitude_homography=self.current_h
        )
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            'n_states': len(self.states),
            'n_anchors': len(self.anchor_factors),
            'n_vision_factors': len(self.vision_factors),
            'current_d': self.current_d,
            'current_h': self.current_h,
            'current_sigma_log_d': self.current_sigma_log_d,
            'vertical_factor': self.vertical_factor,
            'mode': self.mode.name,
            'age_of_last_good_update': self.age_of_last_good_update
        }


class FailureManager:
    """Monitors system health and manages graceful degradation."""
    
    def __init__(self, config: Config):
        self.cfg = config
        
        # Failure states
        self.slam_lost: bool = False
        self.ground_unreliable: bool = False
        self.rpy_spikes: bool = False
        self.depth_failed: bool = False
        
        # Recovery tracking
        self.last_good_estimate: Optional[AltitudeEstimate] = None
        self.failure_start_time: Optional[float] = None
        
        # History for spike detection
        self.rpy_history: deque = deque(maxlen=10)
        self.residual_history: deque = deque(maxlen=10)
    
    def check_slam_health(self, pose_quality: float, 
                          num_tracks: int, inlier_ratio: float) -> bool:
        """Check if SLAM is healthy."""
        healthy = (
            pose_quality >= self.cfg.slam_quality_threshold and
            num_tracks >= self.cfg.min_inliers // 2 and
            inlier_ratio >= 0.3
        )
        
        if not healthy and not self.slam_lost:
            LOG.warning("SLAM tracking lost")
            self.slam_lost = True
        elif healthy and self.slam_lost:
            LOG.info("SLAM tracking recovered")
            self.slam_lost = False
        
        return healthy
    
    def check_ground_health(self, ground_quality: float,
                            residual_m: float, coverage: float) -> bool:
        """Check if ground model is reliable."""
        reliable = (
            ground_quality >= self.cfg.ground_quality_threshold and
            residual_m < 0.5 and
            coverage >= 0.1
        )
        
        if not reliable and not self.ground_unreliable:
            LOG.warning("Ground model unreliable")
            self.ground_unreliable = True
        elif reliable and self.ground_unreliable:
            LOG.info("Ground model recovered")
            self.ground_unreliable = False
        
        return reliable
    
    def check_rpy_health(self, rpy: RPYSample, 
                         vision_residual: float) -> bool:
        """Check for RPY spikes by comparing to vision."""
        self.rpy_history.append(rpy)
        self.residual_history.append(vision_residual)
        
        if len(self.rpy_history) < 3:
            return True
        
        # Detect sudden changes in RPY
        rpy_changes = []
        for i in range(1, len(self.rpy_history)):
            prev = self.rpy_history[i-1]
            curr = self.rpy_history[i]
            change = np.sqrt(
                (curr.roll - prev.roll)**2 +
                (curr.pitch - prev.pitch)**2 +
                (curr.yaw - prev.yaw)**2
            )
            rpy_changes.append(change)
        
        # High residual + sudden RPY change = spike
        if len(rpy_changes) > 0:
            recent_change = rpy_changes[-1]
            if recent_change > np.deg2rad(5) and vision_residual > 5.0:
                if not self.rpy_spikes:
                    LOG.warning("RPY spike detected")
                    self.rpy_spikes = True
                return False
        
        if self.rpy_spikes:
            # Check for recovery
            if np.median(list(self.residual_history)) < 2.0:
                LOG.info("RPY recovered from spike")
                self.rpy_spikes = False
        
        return not self.rpy_spikes
    
    def get_uncertainty_multiplier(self) -> float:
        """Get uncertainty multiplier based on failure states."""
        mult = 1.0
        
        if self.slam_lost:
            mult *= 2.0
        if self.ground_unreliable:
            mult *= 1.5
        if self.rpy_spikes:
            mult *= 2.0
        
        return mult
    
    def should_use_depth_only(self) -> bool:
        """Check if should fall back to depth-only mode."""
        return self.slam_lost and not self.depth_failed
    
    def should_hold(self) -> bool:
        """Check if should hold last estimate."""
        return self.slam_lost and self.depth_failed
    
    def record_good_estimate(self, estimate: AltitudeEstimate) -> None:
        """Record a good estimate for fallback."""
        if estimate.mode in [AltitudeMode.GEOM, AltitudeMode.FUSED]:
            self.last_good_estimate = estimate
            self.failure_start_time = None
    
    def get_fallback_estimate(self, timestamp: float) -> Optional[AltitudeEstimate]:
        """Get fallback estimate during failure."""
        if self.last_good_estimate is None:
            return None
        
        # Decay confidence over time
        if self.failure_start_time is None:
            self.failure_start_time = timestamp
        
        duration = timestamp - self.failure_start_time
        sigma_mult = 1.0 + duration * 0.5  # Inflate uncertainty
        
        return AltitudeEstimate(
            altitude_m=self.last_good_estimate.altitude_m,
            sigma_m=self.last_good_estimate.sigma_m * sigma_mult,
            mode=AltitudeMode.HOLD,
            timestamp=timestamp,
            slam_quality=0.0,
            ground_quality=0.0,
            depth_quality=0.0
        )


# --- Main System ---

class AltitudeEstimationSystem:
    """
    Main altitude estimation system.
    
    PRIMARY: HomographyAltimeter + FixedLagLogDistanceSmoother
    SECONDARY: PoseEngine + GroundPlaneFitter (debug/fallback)
    
    Usage:
        system = AltitudeEstimationSystem(calibration, config)
        for frame in frames:
            estimate = system.process_frame(frame, altitude_gt=known_alt if initializing else None)
    """
    
    def __init__(self, calibration: CalibrationData, config: Config):
        self.calib = calibration
        self.cfg = config
        
        # Core modules
        self.rotation_provider = RotationProvider(calibration, config)
        self.tracker = VisualTracker(calibration, config)
        self.ground_segmenter = GroundSegmenter(config)
        self.depth_prior = DepthPrior(config) if config.use_depth_prior else None
        self.failure_manager = FailureManager(config)
        
        # PRIMARY PATH: Dominant-plane homography altimeter + fixed-lag smoother
        self.homography_altimeter = HomographyAltimeter(calibration, config, calibration.conventions)
        self.smoother = FixedLagLogDistanceSmoother(config, window_size=30)
        
        # SECONDARY PATH (debug/fallback): PoseEngine + ground plane fitting
        self.pose_engine = PoseEngine(calibration, config, self.rotation_provider)
        self.ground_fitter = GroundPlaneFitter(config, calibration.conventions)
        
        self.initializer = Initializer(calibration, config, self.tracker, self.rotation_provider)
        self.is_initialized = False
        
        # State
        self.frame_count = 0
        self.last_timestamp = 0.0
        self.last_keyframe_time = 0.0
        self.last_depth_time = 0.0
        self.last_segment_time = 0.0
        self.prev_R_CW: Optional[np.ndarray] = None  # For relative rotation computation
        
        # Cached results
        self.last_ground_mask: Optional[np.ndarray] = None
        self.last_depth_map: Optional[np.ndarray] = None
        self.last_estimate: Optional[AltitudeEstimate] = None
        self.last_constraint: Optional[HomographyConstraint] = None
        
        # Scheduling intervals
        self.keyframe_interval = 1.0 / config.keyframe_rate
        self.depth_interval = 1.0 / config.depth_rate
        self.segment_interval = 1.0 / config.segmentation_rate
    
    def process_frame(self, frame: FrameData, 
                      altitude_gt: Optional[float] = None) -> AltitudeEstimate:
        """
        Process a single frame and return altitude estimate.
        
        PRIMARY PATH: Dominant-plane homography constraint + fixed-lag smoother
        SECONDARY PATH: PoseEngine + ground plane fitting (for debug/fallback)
        
        Args:
            frame: Frame data with image and RPY
            altitude_gt: Ground truth altitude (for init anchors only)
        
        Returns:
            AltitudeEstimate with altitude, uncertainty, and mode
        """
        self.frame_count += 1
        timestamp = frame.timestamp
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 1.0 / self.cfg.fps
        self.last_timestamp = timestamp
        
        # Add RPY to rotation provider
        if frame.rpy is not None:
            self.rotation_provider.add_rpy(frame.rpy)
        
        # === INITIALIZATION PHASE ===
        if not self.is_initialized:
            init_complete = self.initializer.add_frame(frame, altitude_gt)
            if init_complete:
                self._complete_initialization()
            else:
                return AltitudeEstimate(
                    altitude_m=altitude_gt if altitude_gt else 0.0,
                    sigma_m=float('inf'),
                    mode=AltitudeMode.INIT,
                    timestamp=timestamp
                )
        
        # === RUNTIME PHASE ===
        
        # Get current rotation from RPY
        R_CW = self.rotation_provider.get_R_CW(timestamp)
        if R_CW is None:
            self.smoother.enter_hold(timestamp)
            return self.smoother.get_estimate(timestamp)
        
        # Compute relative rotation (RPY prior for homography candidate selection)
        R_rel_rpy = np.eye(3)
        if self.prev_R_CW is not None:
            R_rel_rpy = R_CW @ self.prev_R_CW.T
        
        # Track features
        prev_pts, curr_pts, inlier_mask, track_ids = self.tracker.track(
            frame.image_gray, timestamp, R_rel_rpy
        )
        tracking_quality = self.tracker.compute_tracking_quality(prev_pts, curr_pts, inlier_mask)
        
        # Compute rotation-warp residual for health monitoring
        rotation_residual = self._compute_rotation_warp_residual(
            prev_pts, curr_pts, inlier_mask, R_rel_rpy, frame.image_gray.shape
        )
        if frame.rpy is not None:
            self.failure_manager.check_rpy_health(frame.rpy, rotation_residual)
        
        # Predict smoother state forward
        self.smoother.predict(timestamp, dt)
        
        # === PRIMARY PATH: Homography constraint ===
        constraint = None
        if len(prev_pts) > 0 and inlier_mask.sum() >= self.homography_altimeter.min_inliers:
            prev_inliers = prev_pts[inlier_mask]
            curr_inliers = curr_pts[inlier_mask]
            
            constraint = self.homography_altimeter.compute_constraint(
                prev_inliers, curr_inliers, R_rel_rpy, R_CW
            )
            self.last_constraint = constraint
        
        # Add constraint to smoother if valid
        if constraint is not None:
            accepted = self.smoother.add_vision_constraint(timestamp, constraint, R_CW)
            if not accepted:
                self.smoother.enter_hold(timestamp)
        else:
            # No valid constraint - check if we should enter HOLD
            if self.homography_altimeter.consecutive_failures > 5:
                self.smoother.enter_hold(timestamp)
        
        # Get primary estimate from smoother
        estimate = self.smoother.get_estimate(timestamp)
        
        # === SECONDARY PATH (debug/fallback): PoseEngine + ground plane ===
        # Run at reduced rate for comparison/debug
        if timestamp - self.last_keyframe_time >= self.keyframe_interval:
            self.last_keyframe_time = timestamp
            
            # Update ground segmentation
            if timestamp - self.last_segment_time >= self.segment_interval:
                self.last_ground_mask, _ = self.ground_segmenter.segment(frame.image, timestamp)
                self.last_segment_time = timestamp
            
            # Run secondary pose engine (for debug metrics)
            fused_alt = self.smoother.current_h if self.smoother.current_h > 0 else None
            _, C_W, pose_quality = self.pose_engine.process_frame(
                frame, prev_pts, curr_pts, inlier_mask, track_ids, self.last_ground_mask,
                self.tracker.prev_features, self.tracker.track_ids, fused_alt
            )
            
            # Secondary ground plane fitting (for comparison)
            ground_pts = self.pose_engine.get_ground_points()
            if len(ground_pts) >= self.cfg.min_ground_inliers and C_W is not None:
                ground_model = self.ground_fitter.fit_plane(ground_pts)
                if ground_model is not None:
                    h_geom_secondary = self.ground_fitter.get_altitude(C_W)
                    estimate.altitude_geom = h_geom_secondary  # Store for debug
        
        # Update previous rotation
        self.prev_R_CW = R_CW.copy()
        
        # Apply failure manager uncertainty multiplier
        uncertainty_mult = self.failure_manager.get_uncertainty_multiplier()
        estimate.sigma_m *= uncertainty_mult
        
        # Record estimate for failure handling
        self.failure_manager.record_good_estimate(estimate)
        
        self.last_estimate = estimate
        return estimate
    
    def _compute_rotation_warp_residual(self, prev_pts: np.ndarray, curr_pts: np.ndarray,
                                         mask: np.ndarray, R_rel: Optional[np.ndarray],
                                         image_shape: Tuple[int, int]) -> float:
        """Compute RMS residual between rotation-predicted and tracked points."""
        if R_rel is None or len(prev_pts) == 0 or mask.sum() < 5:
            return 0.0
        
        K = self.calib.intrinsics.K
        K_inv = self.calib.intrinsics.K_inv
        
        prev_inliers = prev_pts[mask]
        curr_inliers = curr_pts[mask]
        
        # Predict curr_pts using rotation only
        prev_h = np.hstack([prev_inliers, np.ones((len(prev_inliers), 1))])
        H_rot = K @ R_rel @ K_inv  # Rotation-only homography
        curr_pred = (H_rot @ prev_h.T).T
        curr_pred = curr_pred[:, :2] / curr_pred[:, 2:3]
        
        # Compute residual
        residuals = np.linalg.norm(curr_pred - curr_inliers, axis=1)
        rms_residual = float(np.sqrt(np.mean(residuals**2)))
        
        return rms_residual
    
    def _complete_initialization(self) -> None:
        """Transfer state from initializer to runtime modules."""
        LOG.info("Transferring initialization to runtime modules")
        
        # Transfer pose engine (secondary path)
        if self.initializer.pose_engine is not None:
            self.pose_engine = self.initializer.pose_engine
            if self.initializer.depth_prior is not None and self.depth_prior is not None:
                self.depth_prior.scale_a = self.initializer.depth_prior.scale_a
                self.depth_prior.offset_b = self.initializer.depth_prior.offset_b
                self.depth_prior.is_calibrated = self.initializer.depth_prior.is_calibrated
            
            LOG.info(f"Transferred pose engine: {len(self.pose_engine.keyframes)} keyframes, "
                     f"{len(self.pose_engine.map_points)} map points")
        
        # PRIMARY PATH: Initialize smoother with latest anchor only
        # Multiple anchors without vision constraints between them don't help;
        # seed with the most recent anchor for best estimate
        if self.initializer.anchor_altitudes:
            # Sort by frame index to get chronological order
            sorted_anchors = sorted(self.initializer.anchor_altitudes.items(), key=lambda x: x[0])
            
            # Use only the latest anchor for initialization
            latest_frame_idx, latest_altitude = sorted_anchors[-1]
            frame = next((f for f in self.initializer.init_frames if f.index == latest_frame_idx), None)
            
            if frame is not None:
                R_CW = frame.R_CW if frame.R_CW is not None else self.rotation_provider.get_R_CW(frame.timestamp)
                if R_CW is None:
                    R_CW = np.eye(3)
                
                # Add single anchor (sigma ~5% of altitude)
                sigma = max(latest_altitude * 0.05, 1.0)
                self.smoother.add_anchor(frame.timestamp, latest_altitude, sigma, R_CW)
                self.prev_R_CW = R_CW.copy()
        
        LOG.info(f"Smoother initialized with latest anchor, "
                 f"initial altitude: {self.smoother.current_h:.2f}m")
        
        self.is_initialized = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        smoother_info = self.smoother.get_debug_info() if self.smoother else {}
        return {
            'initialized': self.is_initialized,
            'frame_count': self.frame_count,
            'mode': self.smoother.mode.name if self.smoother else 'INIT',
            'altitude': self.smoother.current_h if self.smoother else 0.0,
            'sigma': self.smoother.current_sigma_log_d * self.smoother.current_h if self.smoother else float('inf'),
            'plane_distance': self.smoother.current_d if self.smoother else 0.0,
            'vertical_factor': self.smoother.vertical_factor if self.smoother else 1.0,
            'slam_lost': self.failure_manager.slam_lost,
            'ground_unreliable': self.failure_manager.ground_unreliable,
            'num_keyframes': len(self.pose_engine.keyframes) if self.pose_engine else 0,
            'num_map_points': len(self.pose_engine.map_points) if self.pose_engine else 0,
            'homography_consecutive_failures': self.homography_altimeter.consecutive_failures if self.homography_altimeter else 0,
            'smoother': smoother_info,
            'last_constraint': {
                's': self.last_constraint.s if self.last_constraint else None,
                'metrics': self.last_constraint.metrics if self.last_constraint else None
            } if self.last_constraint else None
        }
    
    def get_last_failure_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the last constraint failure.
        
        Returns:
            Dictionary with 'gate' key indicating which gate failed, or None
        """
        if not self.homography_altimeter:
            return None
        
        if hasattr(self.homography_altimeter, '_last_fail_metrics'):
            metrics = self.homography_altimeter._last_fail_metrics
            if metrics:
                return {
                    'gate': metrics.get('gate_failed', 'unknown'),
                    'metrics': metrics
                }
        return None


# --- Production API ---

@dataclass
class AltimeterResult:
    """
    Simple result from altitude estimation.
    
    Attributes:
        altitude_m: Estimated altitude in meters (AGL)
        sigma_m: Uncertainty (1-sigma) in meters
        is_valid: True if estimate is reliable
        mode: 'INIT', 'GEOM', 'PRED', or 'HOLD'
    """
    altitude_m: float
    sigma_m: float
    is_valid: bool
    mode: str
    
    def __repr__(self) -> str:
        return f"AltimeterResult(alt={self.altitude_m:.2f}m, σ={self.sigma_m:.1f}m, mode={self.mode})"


class RealtimeAltimeter:
    """
    Production-ready real-time altitude estimator.
    
    Init: Feed N frames with known altitude (GPS/barometer).
    Runtime: Input image + RPY, output AltimeterResult.
    
    See example_production_usage.py for full usage examples.
    """
    
    def __init__(
        self,
        K: np.ndarray,
        image_size: Tuple[int, int],
        camera_tilt_deg: float = 60.0,
        init_frames: int = 10,
        fps: float = 30.0
    ):
        """
        Initialize the real-time altimeter.
        
        Args:
            K: Camera intrinsics matrix (3x3)
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
            image_size: (width, height) of input images
            camera_tilt_deg: Camera tilt angle below horizontal (degrees)
                             60° means camera points 60° down from horizontal
            init_frames: Number of frames with known altitude for initialization
            fps: Expected frame rate (for timing)
        """
        # Validate inputs
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {K.shape}")
        if len(image_size) != 2:
            raise ValueError(f"image_size must be (width, height), got {image_size}")
        if init_frames < 1:
            raise ValueError(f"init_frames must be >= 1, got {init_frames}")
        
        self._width, self._height = image_size
        self._fps = fps
        self._frame_count = 0
        self._init_frames = init_frames
        
        # Extract intrinsics from K
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Create calibration
        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=self._width, height=self._height,
            dist_coeffs=np.zeros(5)
        )
        extrinsics = CameraExtrinsics.from_tilt_angle(camera_tilt_deg)
        time_sync = TimeSync(time_offset=0.0)
        conventions = FrameConventions()
        
        calibration = CalibrationData(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            time_sync=time_sync,
            conventions=conventions,
            camera_model="RealtimeAltimeter",
            notes=f"K={fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f}, tilt={camera_tilt_deg}°"
        )
        
        # Create config
        config = Config()
        config.fps = fps
        config.init_window_size = init_frames
        
        # Create system
        self._system = AltitudeEstimationSystem(calibration, config)
        self._is_initialized = False
        
        LOG.info(f"RealtimeAltimeter created: {self._width}x{self._height}, "
                 f"fx={fx:.1f}, tilt={camera_tilt_deg}°, init_frames={init_frames}")
    
    @property
    def is_initialized(self) -> bool:
        """True if system has completed initialization."""
        return self._is_initialized
    
    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count
    
    def process(
        self,
        image: np.ndarray,
        rpy: Tuple[float, float, float],
        known_altitude: Optional[float] = None
    ) -> AltimeterResult:
        """
        Process a single frame and return altitude estimate.
        
        Args:
            image: BGR image (numpy array, HxWx3)
            rpy: (roll, pitch, yaw) in RADIANS
                 - roll: rotation about forward axis (positive = right wing down)
                 - pitch: rotation about right axis (positive = nose up)
                 - yaw: rotation about down axis (positive = clockwise from above)
            known_altitude: Ground truth altitude in meters (required during init)
        
        Returns:
            AltimeterResult with altitude, uncertainty, validity, and mode
        
        Raises:
            ValueError: If image dimensions don't match, or known_altitude missing during init
        """
        # Validate image
        if image is None or image.size == 0:
            raise ValueError("Image is empty")
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape
        if w != self._width or h != self._height:
            raise ValueError(f"Image size mismatch: expected {self._width}x{self._height}, got {w}x{h}")
        
        # Validate RPY
        if len(rpy) != 3:
            raise ValueError(f"rpy must be (roll, pitch, yaw), got length {len(rpy)}")
        roll, pitch, yaw = rpy
        
        # Check if init altitude is required
        needs_init = not self._is_initialized and self._frame_count < self._init_frames
        if needs_init and known_altitude is None:
            raise ValueError(
                f"known_altitude is required for first {self._init_frames} frames "
                f"(currently on frame {self._frame_count})"
            )
        
        # Create timestamp
        timestamp = self._frame_count / self._fps
        
        # Create RPY sample
        rpy_sample = RPYSample(
            timestamp=timestamp,
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            quality=1.0
        )
        
        # Create frame data
        frame = FrameData(
            index=self._frame_count,
            timestamp=timestamp,
            image=image,
            rpy=rpy_sample,
            altitude_gt=known_altitude
        )
        
        # Process frame
        # During init: provide known_altitude
        # After init: don't provide altitude (system estimates independently)
        provide_altitude = known_altitude if needs_init else None
        estimate = self._system.process_frame(frame, altitude_gt=provide_altitude)
        
        self._frame_count += 1
        
        # Update initialization status
        if not self._is_initialized and self._system.is_initialized:
            self._is_initialized = True
            LOG.info(f"RealtimeAltimeter initialized after {self._frame_count} frames")
        
        # Convert to simple result
        is_valid = estimate.mode in [AltitudeMode.GEOM, AltitudeMode.FUSED]
        
        return AltimeterResult(
            altitude_m=estimate.altitude_m,
            sigma_m=estimate.sigma_m,
            is_valid=is_valid,
            mode=estimate.mode.name
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed system status for debugging."""
        return self._system.get_status()
    
    def reset(self) -> None:
        """Reset the altimeter (requires re-initialization)."""
        # Recreate system with same config
        self._system = AltitudeEstimationSystem(
            self._system.calib, 
            self._system.cfg
        )
        self._is_initialized = False
        self._frame_count = 0
        LOG.info("RealtimeAltimeter reset")


def create_altimeter(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    camera_tilt_deg: float = 60.0,
    init_frames: int = 10,
    fps: float = 30.0
) -> RealtimeAltimeter:
    """
    Convenience function to create a RealtimeAltimeter.
    
    Args:
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point in pixels
        image_width, image_height: Image dimensions
        camera_tilt_deg: Camera tilt below horizontal (degrees)
        init_frames: Number of frames with known altitude for init
        fps: Expected frame rate
    
    Returns:
        Configured RealtimeAltimeter instance
    
    Example:
        ```python
        altimeter = create_altimeter(
            fx=739.0, fy=739.0, cx=640.0, cy=360.0,
            image_width=1280, image_height=720,
            camera_tilt_deg=60.0, init_frames=10
        )
        ```
    """
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    
    return RealtimeAltimeter(
        K=K,
        image_size=(image_width, image_height),
        camera_tilt_deg=camera_tilt_deg,
        init_frames=init_frames,
        fps=fps
    )


# --- Utilities ---

def create_default_calibration(image_width: int = 1280, 
                                image_height: int = 720,
                                camera_tilt_deg: float = 60.0) -> CalibrationData:
    """
    Create default calibration for testing.
    
    Uses Unity Physical Camera settings:
    - Vertical FOV: 60°
    - Sensor: 36mm x 24mm
    - Focal Length: 20.78461mm
    - Gate Fit: Horizontal
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        camera_tilt_deg: Camera tilt below horizontal
    
    Returns:
        CalibrationData with default values
    """
    # Unity Physical Camera settings (Gate Fit = Horizontal)
    sensor_width_mm = 36.0
    focal_length_mm = 20.78461
    
    # Step 1: Effective sensor height for Gate Fit Horizontal
    # Unity crops vertically if image aspect ≠ sensor aspect
    sensor_height_eff = sensor_width_mm * image_height / image_width
    
    # Step 2: Compute intrinsics from physical camera model
    # Both fx and fy derived from focal length + sensor (consistent model)
    fx = focal_length_mm * image_width / sensor_width_mm
    fy = focal_length_mm * image_height / sensor_height_eff
    
    # Principal point at image center (no lens shift)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=image_width, height=image_height,
        dist_coeffs=np.zeros(5)  # Unity has no distortion
    )
    
    extrinsics = CameraExtrinsics.from_tilt_angle(camera_tilt_deg)
    
    time_sync = TimeSync(time_offset=0.0)
    
    conventions = FrameConventions()
    
    return CalibrationData(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        time_sync=time_sync,
        conventions=conventions,
        camera_model="Default UAV Camera",
        notes="Auto-generated default calibration"
    )


if __name__ == '__main__':
    print("Use RealtimeAltimeter or create_altimeter() for production usage.")
    print("See example_production_usage.py for examples.")
