/**
 * @file initializer.cpp
 * @brief Stub implementation for Initializer
 */

#include "altitude_estimator/initializer.hpp"
#include "altitude_estimator/pose_engine.hpp"
#include "altitude_estimator/visual_tracker.hpp"
#include "altitude_estimator/rotation_provider.hpp"
#include "altitude_estimator/ground_segmenter.hpp"
#include <iostream>

namespace altitude_estimator {

Initializer::Initializer(
    const CalibrationData& calibration,
    const Config& config,
    VisualTracker* tracker,
    RotationProvider* rotation_provider
) : calib_(calibration), cfg_(config),
    tracker_(tracker), rotation_provider_(rotation_provider) {
}

bool Initializer::addFrame(const FrameData& frame, const std::optional<double>& altitude_gt) {
    // Add RPY if available
    if (frame.rpy) {
        rotation_provider_->addRPY(*frame.rpy);
    }
    
    // Track features - IMPORTANT: call tracker every frame to maintain state
    if (tracker_) {
        auto R_CW = rotation_provider_->getR_CW(frame.timestamp);
        Eigen::Matrix3d* R_rel_ptr = nullptr;
        tracker_->track(frame.image_gray, frame.timestamp, R_rel_ptr);
    }
    
    // Get rotation for frame
    auto R_CW = rotation_provider_->getR_CW(frame.timestamp);
    
    FrameData frame_copy = frame;
    if (R_CW) {
        frame_copy.R_CW = *R_CW;
    }
    
    // Store frame
    init_frames_.push_back(frame_copy);
    
    // Store altitude anchor if provided
    if (altitude_gt) {
        anchor_altitudes_[frame.index] = *altitude_gt;
    }
    
    // Check if ready to initialize
    if (init_frames_.size() >= size_t(cfg_.init_window_size) &&
        anchor_altitudes_.size() >= 2) {
        return completeInitialization();
    }
    
    return false;
}

bool Initializer::completeInitialization() {
    // Create pose engine
    pose_engine_ = std::make_unique<PoseEngine>(calib_, cfg_, rotation_provider_);
    
    // Initialize SLAM
    bool success = pose_engine_->initialize(init_frames_, anchor_altitudes_);
    
    if (success) {
        scale_ = 1.0;  // Simplified
        is_complete_ = true;
    }
    
    return success;
}

} // namespace altitude_estimator

