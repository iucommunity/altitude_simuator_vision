/**
 * @file coordinate_frames.hpp
 * @brief Coordinate frame transformations and rotation utilities
 */

#pragma once

#include "common.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace altitude_estimator {

/**
 * @brief Frame conventions for coordinate systems
 */
struct FrameConventions {
    CoordinateFrame world_frame = CoordinateFrame::NED;
    CoordinateFrame body_frame = CoordinateFrame::FRD;
    CoordinateFrame camera_frame = CoordinateFrame::OPENCV;
    RotationOrder rotation_order = RotationOrder::ZYX;
    
    // Sign conventions
    bool yaw_positive_clockwise = true;   ///< When viewed from above
    bool pitch_positive_nose_up = true;
    bool roll_positive_right_down = true;
    
    std::string yaw_reference = "true_north";  ///< or "magnetic", "arbitrary"
    
    bool validate() const { return true; }
};

/**
 * @brief Convert RPY angles to rotation matrix
 * 
 * @param roll Roll angle (radians)
 * @param pitch Pitch angle (radians)
 * @param yaw Yaw angle (radians)
 * @param order Rotation order (default: ZYX)
 * @return R_WB Body-to-World rotation matrix
 */
Eigen::Matrix3d rpyToRotationMatrix(
    double roll, 
    double pitch, 
    double yaw,
    RotationOrder order = RotationOrder::ZYX
);

/**
 * @brief Extract RPY angles from rotation matrix
 * 
 * @param R Rotation matrix R_WB
 * @param order Rotation order (default: ZYX)
 * @return (roll, pitch, yaw) in radians
 */
Eigen::Vector3d rotationMatrixToRPY(
    const Eigen::Matrix3d& R,
    RotationOrder order = RotationOrder::ZYX
);

/**
 * @brief Rotation matrix about X-axis (roll)
 */
Eigen::Matrix3d Rx(double angle);

/**
 * @brief Rotation matrix about Y-axis (pitch)
 */
Eigen::Matrix3d Ry(double angle);

/**
 * @brief Rotation matrix about Z-axis (yaw)
 */
Eigen::Matrix3d Rz(double angle);

} // namespace altitude_estimator

