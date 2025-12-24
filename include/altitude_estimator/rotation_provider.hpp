/**
 * @file rotation_provider.hpp
 * @brief Provides rotation from RPY measurements
 */

#pragma once

#include "common.hpp"
#include "calibration.hpp"
#include "data_types.hpp"
#include "config.hpp"  // Include full Config, not forward declare
#include <deque>

namespace altitude_estimator {

/**
 * @brief Provides rotation from RPY with time synchronization
 * 
 * Convention: R_CW (World→Camera), C_W (camera pos in World)
 */
class RotationProvider {
public:
    RotationProvider(const CalibrationData& calibration, const Config& config);
    
    /**
     * @brief Add RPY sample to buffer with convention conversion
     */
    void addRPY(const RPYSample& rpy);
    
    /**
     * @brief Get RPY at given timestamp (interpolated)
     */
    std::optional<RPYSample> getRPYAtTime(double t) const;
    
    /**
     * @brief Get World-to-Camera rotation at time t
     * 
     * R_CW transforms vectors from World frame to Camera frame:
     *     v_C = R_CW @ v_W
     */
    std::optional<Eigen::Matrix3d> getR_CW(double t);
    
    /**
     * @brief Get Camera-to-World rotation (inverse of R_CW)
     */
    std::optional<Eigen::Matrix3d> getR_WC(double t);
    
    /**
     * @brief Get relative rotation from Camera at t1 to Camera at t2
     * 
     * R_C2_C1: transforms vectors from Camera1 to Camera2
     */
    std::optional<Eigen::Matrix3d> getRelativeRotation(double t1, double t2);
    
private:
    const CalibrationData& calib_;
    Config config_;  // Store by value
    
    std::deque<RPYSample> rpy_buffer_;
    static constexpr size_t MAX_BUFFER_SIZE = 100;
    
    std::optional<Eigen::Matrix3d> last_R_CW_;
};

} // namespace altitude_estimator

