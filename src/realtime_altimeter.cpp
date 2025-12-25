/**
 * @file realtime_altimeter.cpp
 * @brief Implementation of production RealtimeAltimeter API
 */

#include "altitude_estimator/realtime_altimeter.hpp"
#include "altitude_estimator/altitude_estimation_system.hpp"
#include <stdexcept>
#include <iostream>

namespace altitude_estimator {

RealtimeAltimeter::RealtimeAltimeter(
    const Eigen::Matrix3d& K,
    const cv::Size& image_size,
    double camera_tilt_deg,
    int init_frames,
    double fps
) : width_(image_size.width),
    height_(image_size.height),
    fps_(fps),
    init_frames_(init_frames) {
    
    // Validate inputs
    if (width_ <= 0 || height_ <= 0) {
        throw std::invalid_argument("Image size must be positive");
    }
    if (init_frames < 1) {
        throw std::invalid_argument("init_frames must be >= 1");
    }
    
    // Extract intrinsics from K
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);
    
    // Create calibration
    CameraIntrinsics intrinsics;
    intrinsics.fx = fx;
    intrinsics.fy = fy;
    intrinsics.cx = cx;
    intrinsics.cy = cy;
    intrinsics.width = width_;
    intrinsics.height = height_;
    intrinsics.dist_coeffs = std::vector<double>(5, 0.0);
    
    CameraExtrinsics extrinsics = CameraExtrinsics::fromTiltAngle(camera_tilt_deg);
    
    TimeSync time_sync;
    FrameConventions conventions;
    
    CalibrationData calibration;
    calibration.intrinsics = intrinsics;
    calibration.extrinsics = extrinsics;
    calibration.time_sync = time_sync;
    calibration.conventions = conventions;
    calibration.camera_model = "RealtimeAltimeter";
    
    // Create config
    Config config;
    config.fps = fps;
    config.init_window_size = init_frames;
    
    // Create system
    system_ = std::make_unique<AltitudeEstimationSystem>(calibration, config);
}

RealtimeAltimeter::~RealtimeAltimeter() = default;

// =============================================================================
// INITIALIZATION PHASE
// =============================================================================

bool RealtimeAltimeter::addInitFrame(
    const cv::Mat& image,
    const std::tuple<double, double, double>& rpy,
    double known_altitude
) {
    if (is_initialized_) {
        throw std::runtime_error(
            "addInitFrame() called after initialization is complete. "
            "Use process() for runtime frames."
        );
    }
    
    // Process with known altitude
    processInternal(image, rpy, known_altitude);
    init_count_++;
    
    // Check if initialization is now complete
    if (system_->isInitialized()) {
        is_initialized_ = true;
    }
    
    return is_initialized_;
}

int RealtimeAltimeter::initFramesRemaining() const {
    if (is_initialized_) {
        return 0;
    }
    return std::max(0, init_frames_ - init_count_);
}

// =============================================================================
// RUNTIME PHASE
// =============================================================================

AltimeterResult RealtimeAltimeter::process(
    const cv::Mat& image,
    const std::tuple<double, double, double>& rpy
) {
    if (!is_initialized_) {
        throw std::runtime_error(
            "process() called before initialization is complete. "
            "Call addInitFrame() " + std::to_string(initFramesRemaining()) + 
            " more time(s) with known altitude."
        );
    }
    
    return processInternal(image, rpy, std::nullopt);
}

// =============================================================================
// INTERNAL
// =============================================================================

AltimeterResult RealtimeAltimeter::processInternal(
    const cv::Mat& image,
    const std::tuple<double, double, double>& rpy,
    const std::optional<double>& known_altitude
) {
    // Validate image
    if (image.empty()) {
        throw std::invalid_argument("Image is empty");
    }
    if (image.cols != width_ || image.rows != height_) {
        throw std::invalid_argument("Image size mismatch");
    }
    
    // Create timestamp
    double timestamp = double(frame_count_) / fps_;
    
    // Create RPY sample
    RPYSample rpy_sample;
    rpy_sample.timestamp = timestamp;
    rpy_sample.roll = std::get<0>(rpy);
    rpy_sample.pitch = std::get<1>(rpy);
    rpy_sample.yaw = std::get<2>(rpy);
    rpy_sample.quality = 1.0;
    
    // Create frame data
    FrameData frame;
    frame.index = frame_count_;
    frame.timestamp = timestamp;
    frame.image = image;
    if (image.channels() == 3) {
        cv::cvtColor(image, frame.image_gray, cv::COLOR_BGR2GRAY);
    } else {
        frame.image_gray = image.clone();
    }
    frame.rpy = rpy_sample;
    frame.altitude_gt = known_altitude;
    
    // Process frame
    auto estimate = system_->processFrame(frame, known_altitude);
    
    frame_count_++;
    
    // Convert to simple result
    // All modes are valid except LOST
    bool is_valid = (estimate.mode != AltitudeMode::LOST);
    
    AltimeterResult result;
    result.altitude_m = estimate.altitude_m;
    result.sigma_m = estimate.sigma_m;
    result.is_valid = is_valid;
    result.mode = toString(estimate.mode);
    
    return result;
}

// =============================================================================
// STATUS & CONTROL
// =============================================================================

std::map<std::string, double> RealtimeAltimeter::getStatus() const {
    return system_->getStatus();
}

void RealtimeAltimeter::reset() {
    system_ = std::make_unique<AltitudeEstimationSystem>(
        system_->calibration(),
        system_->config()
    );
    is_initialized_ = false;
    frame_count_ = 0;
    init_count_ = 0;
}

// =============================================================================
// FACTORY
// =============================================================================

std::unique_ptr<RealtimeAltimeter> createAltimeter(
    double fx,
    double fy,
    double cx,
    double cy,
    int image_width,
    int image_height,
    double camera_tilt_deg,
    int init_frames,
    double fps
) {
    Eigen::Matrix3d K;
    K << fx,  0, cx,
          0, fy, cy,
          0,  0,  1;
    
    return std::make_unique<RealtimeAltimeter>(
        K,
        cv::Size(image_width, image_height),
        camera_tilt_deg,
        init_frames,
        fps
    );
}

} // namespace altitude_estimator
