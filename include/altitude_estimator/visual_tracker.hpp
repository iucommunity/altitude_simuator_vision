/**
 * @file visual_tracker.hpp
 * @brief Feature tracking with KLT optical flow
 */

#pragma once

#include "common.hpp"
#include "calibration.hpp"
#include "config.hpp"
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>  // For GFTTDetector
#include <vector>
#include <map>

namespace altitude_estimator {

/**
 * @brief Feature tracking with rotation-predicted KLT optical flow
 */
class VisualTracker {
public:
    VisualTracker(const CalibrationData& calibration, const Config& config);
    
    /**
     * @brief Track features between frames
     * 
     * @param gray Current grayscale image
     * @param timestamp Current timestamp
     * @param R_rel Optional relative rotation for prediction
     * @return Tuple of (prev_pts, curr_pts, inlier_mask, track_ids)
     */
    struct TrackResult {
        std::vector<cv::Point2f> prev_pts;
        std::vector<cv::Point2f> curr_pts;
        std::vector<bool> mask;
        std::vector<TrackId> track_ids;
    };
    
    TrackResult track(const cv::Mat& gray, double timestamp, 
                     const Eigen::Matrix3d* R_rel = nullptr);
    
    /**
     * @brief Compute tracking quality metrics
     */
    struct TrackQuality {
        int num_tracks;
        double inlier_ratio;
        double parallax;
    };
    
    TrackQuality computeTrackingQuality(
        const std::vector<cv::Point2f>& prev_pts,
        const std::vector<cv::Point2f>& curr_pts,
        const std::vector<bool>& mask
    ) const;
    
    // Access to current state
    const std::vector<cv::Point2f>& prevFeatures() const { return prev_features_; }
    const std::vector<TrackId>& trackIds() const { return track_ids_; }
    
private:
    std::vector<cv::Point2f> detectFeatures(const cv::Mat& gray);
    
    std::vector<cv::Point2f> predictFeaturePositions(
        const std::vector<cv::Point2f>& pts,
        const Eigen::Matrix3d& R_rel
    );
    
    const CalibrationData& calib_;
    Config config_;  // Store by value, not reference!
    
    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_features_;
    std::vector<TrackId> track_ids_;
    double prev_timestamp_ = 0.0;
    
    cv::Ptr<cv::GFTTDetector> detector_;
    TrackId next_track_id_ = 0;
};

} // namespace altitude_estimator

