/**
 * @file visual_tracker.cpp
 * @brief Implementation of VisualTracker
 */

#include "altitude_estimator/visual_tracker.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <iostream>

namespace altitude_estimator {

VisualTracker::VisualTracker(const CalibrationData& calibration, const Config& config)
    : calib_(calibration), config_(config) {
    
    detector_ = cv::GFTTDetector::create(
        config.max_features,
        config.feature_quality,
        config.min_feature_distance
    );
}

std::vector<cv::Point2f> VisualTracker::detectFeatures(const cv::Mat& gray) {
    std::vector<cv::KeyPoint> kps;
    detector_->detect(gray, kps);
    
    std::vector<cv::Point2f> pts;
    pts.reserve(kps.size());
    for (const auto& kp : kps) {
        pts.push_back(kp.pt);
    }
    return pts;
}

VisualTracker::TrackResult VisualTracker::track(
    const cv::Mat& gray,
    double timestamp,
    const Eigen::Matrix3d* R_rel
) {
    TrackResult result;
    
    if (prev_gray_.empty() || prev_features_.empty()) {
        // First frame - detect features
        prev_gray_ = gray.clone();
        prev_features_ = detectFeatures(gray);
        track_ids_.resize(prev_features_.size());
        for (size_t i = 0; i < prev_features_.size(); ++i) {
            track_ids_[i] = next_track_id_++;
        }
        prev_timestamp_ = timestamp;
        
        // Return trivial result
        result.prev_pts = prev_features_;
        result.curr_pts = prev_features_;
        result.mask.assign(prev_features_.size(), true);
        result.track_ids = track_ids_;
        return result;
    }
    
    // Re-detect if too few features
    if (prev_features_.size() < 10) {
        prev_features_ = detectFeatures(prev_gray_);
        track_ids_.resize(prev_features_.size());
        for (size_t i = 0; i < prev_features_.size(); ++i) {
            track_ids_[i] = next_track_id_++;
        }
    }
    
    // Safety check: need features to track
    if (prev_features_.empty()) {
        result.prev_pts.clear();
        result.curr_pts.clear();
        result.mask.clear();
        result.track_ids.clear();
        return result;
    }
    
    // KLT optical flow
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    
    cv::Size win_size(config_.lk_win_size_width, config_.lk_win_size_height);
    
    try {
        cv::calcOpticalFlowPyrLK(
            prev_gray_, gray,
            prev_features_, curr_pts,
            status, err,
            win_size,
            config_.lk_max_level
        );
    } catch (const cv::Exception& e) {
        // OpenCV error - return empty result
        result.prev_pts.clear();
        result.curr_pts.clear();
        result.mask.clear();
        result.track_ids.clear();
        return result;
    }
    
    // Safety check: ensure all vectors are same size
    size_t n = status.size();
    if (n != prev_features_.size() || n != curr_pts.size() || n != track_ids_.size()) {
        // Return empty result
        result.prev_pts.clear();
        result.curr_pts.clear();
        result.mask.clear();
        result.track_ids.clear();
        return result;
    }
    
    // Filter by status
    std::vector<cv::Point2f> prev_good, curr_good;
    std::vector<TrackId> ids_good;
    for (size_t i = 0; i < n; ++i) {
        if (status[i]) {
            prev_good.push_back(prev_features_[i]);
            curr_good.push_back(curr_pts[i]);
            ids_good.push_back(track_ids_[i]);
        }
    }
    
    // All tracked points are inliers (no geometric verification for simplicity)
    std::vector<bool> inlier_mask(prev_good.size(), true);
    
    // Update state - keep inliers + detect new
    std::vector<cv::Point2f> inlier_pts;
    std::vector<TrackId> inlier_ids;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_pts.push_back(curr_good[i]);
            inlier_ids.push_back(ids_good[i]);
        }
    }
    
    // Detect new features if needed
    if (inlier_pts.size() < config_.max_features / 2) {
        auto new_pts = detectFeatures(gray);
        
        // IMPORTANT: If we have no surviving tracks, we must bootstrap from scratch.
        // Otherwise the tracker can get stuck with 0 features forever.
        if (inlier_pts.empty()) {
            size_t n_add = std::min(new_pts.size(), size_t(config_.max_features));
            inlier_pts.reserve(n_add);
            inlier_ids.reserve(n_add);
            for (size_t i = 0; i < n_add; ++i) {
                inlier_pts.push_back(new_pts[i]);
                inlier_ids.push_back(next_track_id_++);
            }
        } else {
            // Remove points close to existing
            std::vector<cv::Point2f> far_pts;
            for (const auto& np : new_pts) {
                bool too_close = false;
                for (const auto& ep : inlier_pts) {
                    double dist = cv::norm(np - ep);
                    if (dist < config_.min_feature_distance) {
                        too_close = true;
                        break;
                    }
                }
                if (!too_close) {
                    far_pts.push_back(np);
                }
            }
            
            size_t n_add = std::min(far_pts.size(),
                                   size_t(config_.max_features) - inlier_pts.size());
            for (size_t i = 0; i < n_add; ++i) {
                inlier_pts.push_back(far_pts[i]);
                inlier_ids.push_back(next_track_id_++);
            }
        }
    }
    
    prev_gray_ = gray.clone();
    prev_features_ = inlier_pts;
    track_ids_ = inlier_ids;
    prev_timestamp_ = timestamp;
    
    result.prev_pts = prev_good;
    result.curr_pts = curr_good;
    result.mask = inlier_mask;
    result.track_ids = ids_good;
    
    return result;
}

VisualTracker::TrackQuality VisualTracker::computeTrackingQuality(
    const std::vector<cv::Point2f>& prev_pts,
    const std::vector<cv::Point2f>& curr_pts,
    const std::vector<bool>& mask
) const {
    TrackQuality quality;
    quality.num_tracks = 0;
    quality.inlier_ratio = 0.0;
    quality.parallax = 0.0;
    
    if (prev_pts.empty()) {
        return quality;
    }
    
    for (bool m : mask) {
        if (m) quality.num_tracks++;
    }
    
    quality.inlier_ratio = double(quality.num_tracks) / mask.size();
    
    // Compute median parallax
    if (quality.num_tracks > 0) {
        std::vector<double> displacements;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i]) {
                double dist = cv::norm(curr_pts[i] - prev_pts[i]);
                displacements.push_back(dist);
            }
        }
        std::sort(displacements.begin(), displacements.end());
        quality.parallax = displacements[displacements.size() / 2];
    }
    
    return quality;
}

} // namespace altitude_estimator
