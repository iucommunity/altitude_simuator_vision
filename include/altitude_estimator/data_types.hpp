/**
 * @file data_types.hpp
 * @brief Core data structures for frame data, estimates, etc.
 */

#pragma once

#include "common.hpp"
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>
#include <map>
#include <optional>

namespace altitude_estimator {

/**
 * @brief Single RPY measurement from autopilot
 */
struct RPYSample {
    double timestamp;  ///< seconds
    double roll;       ///< radians
    double pitch;      ///< radians
    double yaw;        ///< radians
    
    std::optional<Eigen::Matrix3d> covariance;  ///< 3x3 covariance if available
    double quality = 1.0;  ///< 0-1 quality indicator
};

/**
 * @brief Data associated with a single frame
 */
struct FrameData {
    FrameIndex index;
    double timestamp;
    
    cv::Mat image;       ///< BGR or grayscale
    cv::Mat image_gray;  ///< Grayscale version
    
    std::optional<RPYSample> rpy;
    std::optional<double> altitude_gt;  ///< Ground truth if available
    
    // Computed data
    std::optional<Eigen::Matrix3d> R_CW;  ///< World-to-Camera rotation
    std::optional<Eigen::Vector3d> C_W;   ///< Camera position in World frame
    
    std::vector<cv::Point2f> features;
    std::vector<TrackId> track_ids;
    cv::Mat descriptors;
    
    FrameData() = default;
    
    FrameData(FrameIndex idx, double ts, const cv::Mat& img)
        : index(idx), timestamp(ts), image(img) {
        if (img.channels() == 3) {
            cv::cvtColor(img, image_gray, cv::COLOR_BGR2GRAY);
        } else {
            image_gray = img.clone();
        }
    }
};

/**
 * @brief Keyframe for SLAM/VO
 */
struct Keyframe {
    FrameIndex index;
    double timestamp;
    
    Eigen::Matrix3d R_CW;  ///< World-to-Camera rotation
    Eigen::Vector3d C_W;   ///< Camera position in World
    
    std::vector<cv::Point2f> features;
    cv::Mat descriptors;
    std::vector<TrackId> point_ids;  ///< Persistent track IDs
    
    // Quality metrics
    int num_tracks = 0;
    double avg_parallax = 0.0;
    double tracking_quality = 1.0;
};

/**
 * @brief 3D map point in world coordinates
 */
struct MapPoint {
    MapPointId id;
    Eigen::Vector3d position;  ///< [x, y, z] in World frame
    
    std::map<FrameIndex, int> observations;  ///< keyframe_idx -> feature_idx
    
    // Quality
    double reprojection_error = 0.0;
    int num_observations = 0;
    double ground_probability = 0.0;
    
    bool isGround() const {
        return ground_probability > 0.7;
    }
};

/**
 * @brief Ground surface model
 */
struct GroundModel {
    Eigen::Vector3d normal = Eigen::Vector3d(0, 0, -1);  ///< Up vector in NED
    double distance = 0.0;  ///< Distance from origin
    
    // Quality metrics
    double inlier_ratio = 0.0;
    double residual_m = 0.0;
    double coverage = 0.0;
    double stability = 0.0;
    
    double quality() const {
        double effective_coverage = (coverage > 0) ? coverage : 1.0;
        return inlier_ratio * effective_coverage * std::max(stability, 0.5);
    }
    
    double distanceToPoint(const Eigen::Vector3d& point) const {
        return normal.dot(point) + distance;
    }
    
    double distanceToCamera(const Eigen::Vector3d& camera_pos) const {
        return std::abs(distanceToPoint(camera_pos));
    }
};

/**
 * @brief Single altitude estimate with metadata
 */
struct AltitudeEstimate {
    double altitude_m;
    double sigma_m;
    AltitudeMode mode;
    double timestamp;
    
    // Quality breakdown
    double slam_quality = 0.0;
    double ground_quality = 0.0;
    double depth_quality = 0.0;
    
    // Component estimates
    std::optional<double> altitude_geom;
    std::optional<double> altitude_depth;
    std::optional<double> altitude_homography;
    
    std::string toString() const {
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "Alt=%.2fm ± %.2fm [%s]",
                 altitude_m, sigma_m, altitude_estimator::toString(mode).c_str());
        return std::string(buffer);
    }
};

/**
 * @brief Simple result from altitude estimation (Production API)
 */
struct AltimeterResult {
    double altitude_m;
    double sigma_m;
    bool is_valid;
    std::string mode;
    
    std::string toString() const {
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "AltimeterResult(alt=%.2fm, σ=%.1fm, mode=%s)",
                 altitude_m, sigma_m, mode.c_str());
        return std::string(buffer);
    }
};

} // namespace altitude_estimator

