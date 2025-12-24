/**
 * @file calibration.cpp
 * @brief Implementation of calibration data structures
 */

#include "altitude_estimator/calibration.hpp"
#include <opencv2/calib3d.hpp>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace altitude_estimator {

// ============================================================================
// CameraIntrinsics
// ============================================================================

std::vector<cv::Point2f> CameraIntrinsics::undistortPoints(
    const std::vector<cv::Point2f>& pts
) const {
    if (pts.empty()) {
        return pts;
    }
    
    std::vector<cv::Point2f> undistorted;
    cv::undistortPoints(pts, undistorted, K_cv(), distCoeffs_cv(), cv::noArray(), K_cv());
    return undistorted;
}

// ============================================================================
// CameraExtrinsics
// ============================================================================

CameraExtrinsics CameraExtrinsics::fromTiltAngle(
    double tilt_deg,
    double roll_offset_deg,
    double yaw_offset_deg,
    const Eigen::Vector3d& translation
) {
    CameraExtrinsics ext;
    
    // Base rotation to align camera to body (no tilt):
    // Camera z (forward) -> Body x (forward)
    // Camera x (right) -> Body y (right)
    // Camera y (down) -> Body z (down)
    Eigen::Matrix3d R_base;
    R_base << 0, 0, 1,
              1, 0, 0,
              0, 1, 0;
    
    // Apply tilt: rotate about Body y-axis (camera pitches down)
    double tilt_rad = tilt_deg * M_PI / 180.0;
    Eigen::Matrix3d R_tilt = Ry(-tilt_rad);
    
    // Apply roll and yaw offsets
    double roll_rad = roll_offset_deg * M_PI / 180.0;
    double yaw_rad = yaw_offset_deg * M_PI / 180.0;
    Eigen::Matrix3d R_offsets = Rz(yaw_rad) * Rx(roll_rad);
    
    ext.R_BC = R_offsets * R_tilt * R_base;
    ext.t_BC = translation;
    
    return ext;
}

std::vector<Eigen::Vector3d> CameraExtrinsics::transformToBody(
    const std::vector<Eigen::Vector3d>& pts_camera
) const {
    std::vector<Eigen::Vector3d> pts_body;
    pts_body.reserve(pts_camera.size());
    
    for (const auto& pt : pts_camera) {
        pts_body.push_back(R_BC * pt + t_BC);
    }
    
    return pts_body;
}

std::vector<Eigen::Vector3d> CameraExtrinsics::transformToCamera(
    const std::vector<Eigen::Vector3d>& pts_body
) const {
    std::vector<Eigen::Vector3d> pts_camera;
    pts_camera.reserve(pts_body.size());
    
    Eigen::Matrix3d R_CB_mat = R_CB();
    for (const auto& pt : pts_body) {
        pts_camera.push_back(R_CB_mat * (pt - t_BC));
    }
    
    return pts_camera;
}

// ============================================================================
// CalibrationData
// ============================================================================

CalibrationData CalibrationData::createDefault(
    int image_width,
    int image_height,
    double camera_tilt_deg
) {
    // Unity Physical Camera settings (Gate Fit = Horizontal)
    const double sensor_width_mm = 36.0;
    const double focal_length_mm = 20.78461;
    
    // Effective sensor height for Gate Fit Horizontal
    double sensor_height_eff = sensor_width_mm * image_height / image_width;
    
    // Compute intrinsics from physical camera model
    double fx = focal_length_mm * image_width / sensor_width_mm;
    double fy = focal_length_mm * image_height / sensor_height_eff;
    double cx = image_width / 2.0;
    double cy = image_height / 2.0;
    
    CameraIntrinsics intrinsics;
    intrinsics.fx = fx;
    intrinsics.fy = fy;
    intrinsics.cx = cx;
    intrinsics.cy = cy;
    intrinsics.dist_coeffs = std::vector<double>(5, 0.0);
    intrinsics.width = image_width;
    intrinsics.height = image_height;
    
    CameraExtrinsics extrinsics = CameraExtrinsics::fromTiltAngle(camera_tilt_deg);
    
    TimeSync time_sync;
    time_sync.time_offset = 0.0;
    
    CalibrationData calib;
    calib.intrinsics = intrinsics;
    calib.extrinsics = extrinsics;
    calib.time_sync = time_sync;
    calib.conventions = FrameConventions();
    calib.camera_model = "Default UAV Camera";
    calib.notes = "Auto-generated default calibration";
    
    return calib;
}

} // namespace altitude_estimator

