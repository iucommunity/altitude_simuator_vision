/**
 * @file altitude_estimation_system.hpp
 * @brief Main altitude estimation system (integrates all components)
 */

#pragma once

#include "common.hpp"
#include "calibration.hpp"
#include "config.hpp"
#include "data_types.hpp"
#include "rotation_provider.hpp"
#include "visual_tracker.hpp"
#include "homography_altimeter.hpp"
#include "smoother.hpp"
#include <memory>

namespace altitude_estimator {

// Forward declarations for secondary path components
class PoseEngine;
class GroundPlaneFitter;
class GroundSegmenter;
class Initializer;

/**
 * @brief Main altitude estimation system
 * 
 * PRIMARY: HomographyAltimeter + FixedLagLogDistanceSmoother
 * SECONDARY: PoseEngine + GroundPlaneFitter (debug/fallback)
 */
class AltitudeEstimationSystem {
public:
    AltitudeEstimationSystem(const CalibrationData& calibration, const Config& config);
    ~AltitudeEstimationSystem();
    
    /**
     * @brief Process a single frame and return altitude estimate
     * 
     * PRIMARY PATH: Dominant-plane homography constraint + fixed-lag smoother
     * SECONDARY PATH: PoseEngine + ground plane fitting (for debug/fallback)
     * 
     * @param frame Frame data with image and RPY
     * @param altitude_gt Ground truth altitude (for init anchors only)
     * @return AltitudeEstimate with altitude, uncertainty, and mode
     */
    AltitudeEstimate processFrame(
        const FrameData& frame,
        const std::optional<double>& altitude_gt = std::nullopt
    );
    
    /**
     * @brief True if initialization is complete
     */
    bool isInitialized() const { return is_initialized_; }
    
    /**
     * @brief Get current system status
     */
    std::map<std::string, double> getStatus() const;
    
    /**
     * @brief Get calibration data
     */
    const CalibrationData& calibration() const { return calib_; }
    
    /**
     * @brief Get configuration
     */
    const Config& config() const { return cfg_; }
    
private:
    void completeInitialization();
    
    double computeRotationWarpResidual(
        const std::vector<cv::Point2f>& prev_pts,
        const std::vector<cv::Point2f>& curr_pts,
        const std::vector<bool>& mask,
        const Eigen::Matrix3d* R_rel,
        const cv::Size& image_shape
    );
    
    // IMPORTANT: store by value.
    // RealtimeAltimeter constructs CalibrationData/Config as locals and passes them in;
    // storing by reference here would dangle and cause undefined behavior (e.g. stuck altitude).
    CalibrationData calib_;
    Config cfg_;
    
    // Core modules
    std::unique_ptr<RotationProvider> rotation_provider_;
    std::unique_ptr<VisualTracker> tracker_;
    std::unique_ptr<GroundSegmenter> ground_segmenter_;
    
    // PRIMARY PATH
    std::unique_ptr<HomographyAltimeter> homography_altimeter_;
    std::unique_ptr<FixedLagLogDistanceSmoother> smoother_;
    
    // SECONDARY PATH (debug/fallback)
    std::unique_ptr<PoseEngine> pose_engine_;
    std::unique_ptr<GroundPlaneFitter> ground_fitter_;
    
    std::unique_ptr<Initializer> initializer_;
    bool is_initialized_ = false;
    
    // State
    int frame_count_ = 0;
    double last_timestamp_ = 0.0;
    std::optional<Eigen::Matrix3d> prev_R_CW_;
    
    std::optional<HomographyConstraint> last_constraint_;

    // Debug/telemetry (helps diagnose "stuck at anchor")
    size_t last_track_total_ = 0;
    size_t last_track_inliers_ = 0;
    bool last_homography_attempted_ = false;
    bool last_homography_succeeded_ = false;
};

} // namespace altitude_estimator

