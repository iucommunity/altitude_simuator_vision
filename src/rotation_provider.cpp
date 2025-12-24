/**
 * @file rotation_provider.cpp
 * @brief Implementation of RotationProvider
 */

#include "altitude_estimator/rotation_provider.hpp"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace altitude_estimator {

RotationProvider::RotationProvider(const CalibrationData& calibration, const Config& config)
    : calib_(calibration), config_(config) {
}

void RotationProvider::addRPY(const RPYSample& rpy) {
    // Apply time synchronization
    double aligned_time = calib_.time_sync.alignTimestamp(rpy.timestamp);
    
    // Apply sign conventions from calibration
    const auto& conv = calib_.conventions;
    
    double roll = rpy.roll;
    double pitch = rpy.pitch;
    double yaw = rpy.yaw;
    
    // If yaw_positive_clockwise=True (common autopilot), negate for math convention
    if (conv.yaw_positive_clockwise) {
        yaw = -yaw;
    }
    
    RPYSample rpy_aligned;
    rpy_aligned.timestamp = aligned_time;
    rpy_aligned.roll = roll;
    rpy_aligned.pitch = pitch;
    rpy_aligned.yaw = yaw;
    rpy_aligned.covariance = rpy.covariance;
    rpy_aligned.quality = rpy.quality;
    
    rpy_buffer_.push_back(rpy_aligned);
    
    // Limit buffer size
    while (rpy_buffer_.size() > MAX_BUFFER_SIZE) {
        rpy_buffer_.pop_front();
    }
}

std::optional<RPYSample> RotationProvider::getRPYAtTime(double t) const {
    if (rpy_buffer_.empty()) {
        return std::nullopt;
    }
    
    if (rpy_buffer_.size() == 1) {
        return rpy_buffer_.back();
    }
    
    // Find bracketing samples
    const RPYSample* before = nullptr;
    const RPYSample* after = nullptr;
    
    for (const auto& rpy : rpy_buffer_) {
        if (rpy.timestamp <= t) {
            before = &rpy;
        } else if (after == nullptr) {
            after = &rpy;
            break;
        }
    }
    
    if (before == nullptr) {
        return rpy_buffer_.front();
    }
    if (after == nullptr) {
        return *before;
    }
    
    // Linear interpolation
    double alpha = (t - before->timestamp) / (after->timestamp - before->timestamp + 1e-9);
    alpha = std::clamp(alpha, 0.0, 1.0);
    
    RPYSample interpolated;
    interpolated.timestamp = t;
    interpolated.roll = before->roll + alpha * (after->roll - before->roll);
    interpolated.pitch = before->pitch + alpha * (after->pitch - before->pitch);
    
    // Yaw wrap-around
    double yaw_diff = after->yaw - before->yaw;
    if (yaw_diff > M_PI) {
        yaw_diff -= 2 * M_PI;
    } else if (yaw_diff < -M_PI) {
        yaw_diff += 2 * M_PI;
    }
    interpolated.yaw = before->yaw + alpha * yaw_diff;
    
    interpolated.quality = std::min(before->quality, after->quality);
    
    return interpolated;
}

std::optional<Eigen::Matrix3d> RotationProvider::getR_CW(double t) {
    auto rpy_opt = getRPYAtTime(t);
    if (!rpy_opt) {
        return std::nullopt;
    }
    
    const auto& rpy = *rpy_opt;
    
    // R_WB: Body -> World rotation from RPY
    Eigen::Matrix3d R_WB = rpyToRotationMatrix(
        rpy.roll, rpy.pitch, rpy.yaw,
        calib_.conventions.rotation_order
    );
    
    // R_BW: World -> Body (inverse)
    Eigen::Matrix3d R_BW = R_WB.transpose();
    
    // R_CW = R_CB @ R_BW: World -> Body -> Camera
    Eigen::Matrix3d R_CW = calib_.extrinsics.R_CB() * R_BW;
    
    last_R_CW_ = R_CW;
    return R_CW;
}

std::optional<Eigen::Matrix3d> RotationProvider::getR_WC(double t) {
    auto R_CW = getR_CW(t);
    if (!R_CW) {
        return std::nullopt;
    }
    return R_CW->transpose();
}

std::optional<Eigen::Matrix3d> RotationProvider::getRelativeRotation(double t1, double t2) {
    auto R_CW1 = getR_CW(t1);
    auto R_CW2 = getR_CW(t2);
    
    if (!R_CW1 || !R_CW2) {
        return std::nullopt;
    }
    
    // R_C2_C1 = R_CW2 @ R_CW1.T
    return *R_CW2 * R_CW1->transpose();
}

} // namespace altitude_estimator

