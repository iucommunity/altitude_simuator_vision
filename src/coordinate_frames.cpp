/**
 * @file coordinate_frames.cpp
 * @brief Implementation of coordinate frame transformations
 */

#include "altitude_estimator/coordinate_frames.hpp"
#include <cmath>

namespace altitude_estimator {

Eigen::Matrix3d Rx(double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    Eigen::Matrix3d R;
    R << 1, 0, 0,
         0, c, -s,
         0, s, c;
    return R;
}

Eigen::Matrix3d Ry(double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    Eigen::Matrix3d R;
    R << c, 0, s,
         0, 1, 0,
         -s, 0, c;
    return R;
}

Eigen::Matrix3d Rz(double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    Eigen::Matrix3d R;
    R << c, -s, 0,
         s, c, 0,
         0, 0, 1;
    return R;
}

Eigen::Matrix3d rpyToRotationMatrix(double roll, double pitch, double yaw, RotationOrder order) {
    switch (order) {
        case RotationOrder::ZYX:
            // R_WB = Rz(yaw) @ Ry(pitch) @ Rx(roll)
            return Rz(yaw) * Ry(pitch) * Rx(roll);
            
        case RotationOrder::XYZ:
            return Rx(roll) * Ry(pitch) * Rz(yaw);
            
        case RotationOrder::ZXY:
            return Rz(yaw) * Rx(roll) * Ry(pitch);
            
        default:
            return Eigen::Matrix3d::Identity();
    }
}

Eigen::Vector3d rotationMatrixToRPY(const Eigen::Matrix3d& R, RotationOrder order) {
    double roll, pitch, yaw;
    
    if (order == RotationOrder::ZYX) {
        // For R = Rz @ Ry @ Rx
        pitch = -std::asin(std::clamp(R(2, 0), -1.0, 1.0));
        
        if (std::abs(std::cos(pitch)) > 1e-6) {
            roll = std::atan2(R(2, 1), R(2, 2));
            yaw = std::atan2(R(1, 0), R(0, 0));
        } else {
            // Gimbal lock
            roll = 0.0;
            yaw = std::atan2(-R(0, 1), R(1, 1));
        }
    } else {
        // Fallback using Eigen
        Eigen::Vector3d euler;
        if (order == RotationOrder::XYZ) {
            euler = R.eulerAngles(0, 1, 2);  // X-Y-Z order
        } else {  // ZXY
            euler = R.eulerAngles(2, 0, 1);  // Z-X-Y order
            return Eigen::Vector3d(euler(1), euler(2), euler(0));  // reorder to RPY
        }
        return euler;
    }
    
    return Eigen::Vector3d(roll, pitch, yaw);
}

} // namespace altitude_estimator

