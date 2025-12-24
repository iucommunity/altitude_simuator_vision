/**
 * @file pose_engine.hpp
 * @brief Rotation-aided monocular visual odometry (SECONDARY path)
 */

#pragma once

#include "common.hpp"
#include "calibration.hpp"
#include "config.hpp"
#include "data_types.hpp"
#include "rotation_provider.hpp"
#include <map>
#include <vector>
#include <optional>

namespace altitude_estimator {

/**
 * @brief Rotation-aided monocular visual odometry (SECONDARY path for debug/fallback)
 */
class PoseEngine {
public:
    PoseEngine(const CalibrationData& calibration,
               const Config& config,
               RotationProvider* rotation_provider);
    
    bool initialize(const std::vector<FrameData>& frames,
                   const std::map<FrameIndex, double>& anchor_altitudes);
    
    std::tuple<std::optional<Eigen::Matrix3d>, std::optional<Eigen::Vector3d>, double>
    processFrame(const FrameData& frame,
                const std::vector<cv::Point2f>& prev_pts,
                const std::vector<cv::Point2f>& curr_pts,
                const std::vector<bool>& inlier_mask,
                const std::vector<TrackId>& track_ids,
                const cv::Mat* ground_mask,
                const std::vector<cv::Point2f>& all_features,
                const std::vector<TrackId>& all_track_ids,
                const std::optional<double>& fused_altitude);
    
    std::optional<double> getCameraAltitude() const;
    std::vector<Eigen::Vector3d> getGroundPoints() const;
    int classifyGroundPoints(const cv::Mat& ground_mask,
                            const Eigen::Matrix3d& R_CW,
                            const Eigen::Vector3d& C_W,
                            const Eigen::Matrix3d& K);
    
    const std::vector<Keyframe>& keyframes() const { return keyframes_; }
    const std::map<MapPointId, MapPoint>& mapPoints() const { return map_points_; }
    
private:
    const CalibrationData& calib_;
    Config cfg_;  // Store by value
    RotationProvider* rotation_provider_;
    
    std::vector<Keyframe> keyframes_;
    std::map<MapPointId, MapPoint> map_points_;
    MapPointId next_point_id_ = 0;
    
    std::optional<Eigen::Matrix3d> current_R_CW_;
    std::optional<Eigen::Vector3d> current_C_W_;
    double scale_ = 1.0;
    bool is_initialized_ = false;
};

} // namespace altitude_estimator

