/**
 * @file altitude_estimation_system.cpp
 * @brief Implementation of main altitude estimation system
 */

#include "altitude_estimator/altitude_estimation_system.hpp"
#include "altitude_estimator/pose_engine.hpp"
#include "altitude_estimator/ground_plane_fitter.hpp"
#include "altitude_estimator/ground_segmenter.hpp"
#include "altitude_estimator/initializer.hpp"
#include <iostream>

namespace altitude_estimator {

AltitudeEstimationSystem::AltitudeEstimationSystem(
    const CalibrationData& calibration,
    const Config& config
) : calib_(calibration), cfg_(config) {
    
    // Create core modules - use member variables calib_ and cfg_, not parameters
    rotation_provider_ = std::make_unique<RotationProvider>(calib_, cfg_);
    tracker_ = std::make_unique<VisualTracker>(calib_, cfg_);
    ground_segmenter_ = std::make_unique<GroundSegmenter>(cfg_);
    
    // PRIMARY PATH
    homography_altimeter_ = std::make_unique<HomographyAltimeter>(
        calib_, cfg_, calib_.conventions
    );
    smoother_ = std::make_unique<FixedLagLogDistanceSmoother>(cfg_, 30);
    
    // SECONDARY PATH
    pose_engine_ = std::make_unique<PoseEngine>(calib_, cfg_, rotation_provider_.get());
    ground_fitter_ = std::make_unique<GroundPlaneFitter>(cfg_, calib_.conventions);
    
    // Initializer
    initializer_ = std::make_unique<Initializer>(
        calib_, cfg_, tracker_.get(), rotation_provider_.get()
    );
}

AltitudeEstimationSystem::~AltitudeEstimationSystem() = default;

AltitudeEstimate AltitudeEstimationSystem::processFrame(
    const FrameData& frame,
    const std::optional<double>& altitude_gt
) {
    frame_count_++;
    double timestamp = frame.timestamp;
    double dt = (last_timestamp_ > 0) ? (timestamp - last_timestamp_) : (1.0 / cfg_.fps);
    last_timestamp_ = timestamp;
    
    // Add RPY
    if (frame.rpy) {
        rotation_provider_->addRPY(*frame.rpy);
    }
    
    // === INITIALIZATION PHASE ===
    if (!is_initialized_) {
        bool init_complete = initializer_->addFrame(frame, altitude_gt);
        if (init_complete) {
            completeInitialization();
        } else {
            AltitudeEstimate est;
            est.altitude_m = altitude_gt ? *altitude_gt : 0.0;
            est.sigma_m = std::numeric_limits<double>::infinity();
            est.mode = AltitudeMode::INIT;
            est.timestamp = timestamp;
            return est;
        }
    }
    
    // === RUNTIME PHASE ===
    
    // Get current rotation
    auto R_CW = rotation_provider_->getR_CW(timestamp);
    if (!R_CW) {
        smoother_->enterHold(timestamp);
        return smoother_->getEstimate(timestamp);
    }
    
    // Compute relative rotation
    Eigen::Matrix3d R_rel_rpy = Eigen::Matrix3d::Identity();
    if (prev_R_CW_) {
        R_rel_rpy = *R_CW * prev_R_CW_->transpose();
    }
    
    // Track features
    auto track_result = tracker_->track(frame.image_gray, timestamp, &R_rel_rpy);
    
    // Predict smoother
    smoother_->predict(timestamp, dt);
    
    // === PRIMARY PATH: Homography constraint ===
    std::optional<HomographyConstraint> constraint;
    if (track_result.mask.size() >= size_t(homography_altimeter_->consecutiveFailures())) {
        std::vector<cv::Point2f> prev_inliers, curr_inliers;
        for (size_t i = 0; i < track_result.mask.size(); ++i) {
            if (track_result.mask[i]) {
                prev_inliers.push_back(track_result.prev_pts[i]);
                curr_inliers.push_back(track_result.curr_pts[i]);
            }
        }
        
        if (prev_inliers.size() >= 20) {
            constraint = homography_altimeter_->computeConstraint(
                prev_inliers, curr_inliers, R_rel_rpy, *R_CW
            );
            last_constraint_ = constraint;
        }
    }
    
    // Add constraint to smoother
    if (constraint) {
        bool accepted = smoother_->addVisionConstraint(timestamp, *constraint, *R_CW);
        if (!accepted) {
            smoother_->enterHold(timestamp);
        }
    } else {
        if (homography_altimeter_->consecutiveFailures() > 5) {
            smoother_->enterHold(timestamp);
        }
    }
    
    // Get estimate
    auto estimate = smoother_->getEstimate(timestamp);
    
    // Update previous rotation
    prev_R_CW_ = *R_CW;
    
    return estimate;
}

void AltitudeEstimationSystem::completeInitialization() {
    // Transfer pose engine from initializer
    if (initializer_->poseEngine()) {
        // pose_engine_ already created, just mark initialized
    }
    
    // Initialize smoother with latest anchor
    const auto& anchors = initializer_->anchorAltitudes();
    if (!anchors.empty()) {
        auto latest = anchors.rbegin();
        auto frame_it = std::find_if(
            initializer_->initFrames().begin(),
            initializer_->initFrames().end(),
            [&](const FrameData& f) { return f.index == latest->first; }
        );
        
        if (frame_it != initializer_->initFrames().end()) {
            Eigen::Matrix3d R_CW = frame_it->R_CW ? *frame_it->R_CW : Eigen::Matrix3d::Identity();
            double sigma = std::max(latest->second * 0.05, 1.0);
            smoother_->addAnchor(frame_it->timestamp, latest->second, sigma, R_CW);
            prev_R_CW_ = R_CW;
        }
    }
    
    is_initialized_ = true;
}

double AltitudeEstimationSystem::computeRotationWarpResidual(
    const std::vector<cv::Point2f>& prev_pts,
    const std::vector<cv::Point2f>& curr_pts,
    const std::vector<bool>& mask,
    const Eigen::Matrix3d* R_rel,
    const cv::Size& image_shape
) {
    // Simplified - return 0
    return 0.0;
}

std::map<std::string, double> AltitudeEstimationSystem::getStatus() const {
    std::map<std::string, double> status;
    status["initialized"] = is_initialized_ ? 1.0 : 0.0;
    status["frame_count"] = frame_count_;
    status["altitude"] = smoother_->currentH();
    status["mode"] = static_cast<double>(smoother_->mode());
    
    auto smoother_info = smoother_->getDebugInfo();
    for (const auto& [k, v] : smoother_info) {
        status["smoother_" + k] = v;
    }
    
    return status;
}

} // namespace altitude_estimator

