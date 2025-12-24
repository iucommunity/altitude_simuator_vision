/**
 * @file pose_engine.cpp
 * @brief Stub implementation for PoseEngine (SECONDARY path)
 */

#include "altitude_estimator/pose_engine.hpp"

namespace altitude_estimator {

PoseEngine::PoseEngine(
    const CalibrationData& calibration,
    const Config& config,
    RotationProvider* rotation_provider
) : calib_(calibration), cfg_(config), rotation_provider_(rotation_provider) {
}

bool PoseEngine::initialize(
    const std::vector<FrameData>& frames,
    const std::map<FrameIndex, double>& anchor_altitudes
) {
    // Simplified initialization - just mark as initialized
    is_initialized_ = !frames.empty() && !anchor_altitudes.empty();
    return is_initialized_;
}

std::tuple<std::optional<Eigen::Matrix3d>, std::optional<Eigen::Vector3d>, double>
PoseEngine::processFrame(
    const FrameData& frame,
    const std::vector<cv::Point2f>& prev_pts,
    const std::vector<cv::Point2f>& curr_pts,
    const std::vector<bool>& inlier_mask,
    const std::vector<TrackId>& track_ids,
    const cv::Mat* ground_mask,
    const std::vector<cv::Point2f>& all_features,
    const std::vector<TrackId>& all_track_ids,
    const std::optional<double>& fused_altitude
) {
    // Stub - return nullopt (not used in primary path)
    return {std::nullopt, std::nullopt, 0.0};
}

std::optional<double> PoseEngine::getCameraAltitude() const {
    if (!current_C_W_) {
        return std::nullopt;
    }
    return -(*current_C_W_)(2);  // NED: altitude = -z
}

std::vector<Eigen::Vector3d> PoseEngine::getGroundPoints() const {
    std::vector<Eigen::Vector3d> pts;
    for (const auto& [id, mp] : map_points_) {
        if (mp.isGround()) {
            pts.push_back(mp.position);
        }
    }
    return pts;
}

int PoseEngine::classifyGroundPoints(
    const cv::Mat& ground_mask,
    const Eigen::Matrix3d& R_CW,
    const Eigen::Vector3d& C_W,
    const Eigen::Matrix3d& K
) {
    // Stub - not used in primary path
    return 0;
}

} // namespace altitude_estimator

