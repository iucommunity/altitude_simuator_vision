/**
 * @file initializer.hpp
 * @brief Handles system initialization with known altitude anchors
 */

#pragma once

#include "common.hpp"
#include "calibration.hpp"
#include "config.hpp"
#include "data_types.hpp"
#include <map>
#include <vector>

namespace altitude_estimator {

// Forward declarations
class RotationProvider;
class VisualTracker;
class PoseEngine;

class Initializer {
public:
    Initializer(const CalibrationData& calibration,
               const Config& config,
               VisualTracker* tracker,
               RotationProvider* rotation_provider);
    
    bool addFrame(const FrameData& frame, const std::optional<double>& altitude_gt);
    
    bool isComplete() const { return is_complete_; }
    double scale() const { return scale_; }
    PoseEngine* poseEngine() { return pose_engine_.get(); }
    
    const std::vector<FrameData>& initFrames() const { return init_frames_; }
    const std::map<FrameIndex, double>& anchorAltitudes() const { return anchor_altitudes_; }
    
private:
    bool completeInitialization();
    
    const CalibrationData& calib_;
    const Config& cfg_;
    VisualTracker* tracker_;
    RotationProvider* rotation_provider_;
    
    std::vector<FrameData> init_frames_;
    std::map<FrameIndex, double> anchor_altitudes_;
    
    bool is_complete_ = false;
    double scale_ = 1.0;
    std::unique_ptr<PoseEngine> pose_engine_;
};

} // namespace altitude_estimator

