/**
 * @file realtime_altimeter.hpp
 * @brief Production-ready real-time altitude estimator API
 */

#pragma once

#include "common.hpp"
#include "calibration.hpp"
#include "config.hpp"
#include "data_types.hpp"
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <memory>
#include <tuple>

namespace altitude_estimator {

// Forward declarations
class AltitudeEstimationSystem;

/**
 * @brief Production-ready real-time altitude estimator
 * 
 * Init: Feed N frames with known altitude (GPS/barometer).
 * Runtime: Input image + RPY, output AltimeterResult.
 * 
 * Example usage:
 * ```cpp
 * // Create altimeter
 * auto altimeter = createAltimeter(
 *     fx, fy, cx, cy, width, height,
 *     60.0,  // camera_tilt_deg
 *     10     // init_frames
 * );
 * 
 * // Initialization phase (first 10 frames with known altitude)
 * for (int i = 0; i < 10; i++) {
 *     auto result = altimeter->process(image, rpy, known_altitude);
 * }
 * 
 * // Runtime phase (no known altitude needed)
 * while (true) {
 *     auto result = altimeter->process(image, rpy);
 *     if (result.is_valid) {
 *         printf("Altitude: %.2f ± %.2f m\n", result.altitude_m, result.sigma_m);
 *     }
 * }
 * ```
 */
class RealtimeAltimeter {
public:
    /**
     * @brief Initialize the real-time altimeter
     * 
     * @param K Camera intrinsics matrix (3x3)
     *          [[fx,  0, cx],
     *           [ 0, fy, cy],
     *           [ 0,  0,  1]]
     * @param image_size (width, height) of input images
     * @param camera_tilt_deg Camera tilt angle below horizontal (degrees)
     *                        60° means camera points 60° down from horizontal
     * @param init_frames Number of frames with known altitude for initialization
     * @param fps Expected frame rate (for timing)
     */
    RealtimeAltimeter(
        const Eigen::Matrix3d& K,
        const cv::Size& image_size,
        double camera_tilt_deg = 60.0,
        int init_frames = 10,
        double fps = 30.0
    );
    
    ~RealtimeAltimeter();
    
    /**
     * @brief Process a single frame and return altitude estimate
     * 
     * @param image BGR image (HxWx3 or HxW grayscale)
     * @param rpy (roll, pitch, yaw) in RADIANS
     *            - roll: rotation about forward axis (positive = right wing down)
     *            - pitch: rotation about right axis (positive = nose up)
     *            - yaw: rotation about down axis (positive = clockwise from above)
     * @param known_altitude Ground truth altitude in meters (required during init)
     * @return AltimeterResult with altitude, uncertainty, validity, and mode
     * 
     * @throws std::invalid_argument if image dimensions don't match, 
     *         or known_altitude missing during init
     */
    AltimeterResult process(
        const cv::Mat& image,
        const std::tuple<double, double, double>& rpy,
        const std::optional<double>& known_altitude = std::nullopt
    );
    
    /**
     * @brief True if system has completed initialization
     */
    bool isInitialized() const { return is_initialized_; }
    
    /**
     * @brief Number of frames processed
     */
    int frameCount() const { return frame_count_; }
    
    /**
     * @brief Get detailed system status for debugging
     */
    std::map<std::string, double> getStatus() const;
    
    /**
     * @brief Reset the altimeter (requires re-initialization)
     */
    void reset();
    
private:
    int width_;
    int height_;
    double fps_;
    int frame_count_ = 0;
    int init_frames_;
    bool is_initialized_ = false;
    
    std::unique_ptr<AltitudeEstimationSystem> system_;
};

/**
 * @brief Convenience function to create a RealtimeAltimeter
 * 
 * @param fx, fy Focal lengths in pixels
 * @param cx, cy Principal point in pixels
 * @param image_width, image_height Image dimensions
 * @param camera_tilt_deg Camera tilt below horizontal (degrees)
 * @param init_frames Number of frames with known altitude for init
 * @param fps Expected frame rate
 * @return Configured RealtimeAltimeter instance
 */
std::unique_ptr<RealtimeAltimeter> createAltimeter(
    double fx,
    double fy,
    double cx,
    double cy,
    int image_width,
    int image_height,
    double camera_tilt_deg = 60.0,
    int init_frames = 10,
    double fps = 30.0
);

} // namespace altitude_estimator

