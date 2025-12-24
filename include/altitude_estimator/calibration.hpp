/**
 * @file calibration.hpp
 * @brief Camera calibration data structures
 */

#pragma once

#include "common.hpp"
#include "coordinate_frames.hpp"
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

namespace altitude_estimator {

/**
 * @brief Camera intrinsic parameters
 */
struct CameraIntrinsics {
    double fx;  ///< Focal length x (pixels)
    double fy;  ///< Focal length y (pixels)
    double cx;  ///< Principal point x (pixels)
    double cy;  ///< Principal point y (pixels)
    
    std::vector<double> dist_coeffs;  ///< Distortion coefficients (k1, k2, p1, p2, k3)
    
    int width = 1280;   ///< Image width
    int height = 720;   ///< Image height
    
    double reprojection_rms = 0.0;  ///< RMS reprojection error from calibration
    
    /**
     * @brief Get camera matrix K
     */
    Eigen::Matrix3d K() const {
        Eigen::Matrix3d mat;
        mat << fx, 0, cx,
               0, fy, cy,
               0, 0, 1;
        return mat;
    }
    
    /**
     * @brief Get OpenCV camera matrix
     */
    cv::Mat K_cv() const {
        cv::Mat mat = cv::Mat::eye(3, 3, CV_64F);
        mat.at<double>(0, 0) = fx;
        mat.at<double>(1, 1) = fy;
        mat.at<double>(0, 2) = cx;
        mat.at<double>(1, 2) = cy;
        return mat;
    }
    
    /**
     * @brief Get OpenCV distortion coefficients
     */
    cv::Mat distCoeffs_cv() const {
        if (dist_coeffs.empty()) {
            return cv::Mat::zeros(5, 1, CV_64F);
        }
        cv::Mat mat(dist_coeffs.size(), 1, CV_64F);
        for (size_t i = 0; i < dist_coeffs.size(); ++i) {
            mat.at<double>(i) = dist_coeffs[i];
        }
        return mat;
    }
    
    /**
     * @brief Undistort 2D points
     */
    std::vector<cv::Point2f> undistortPoints(const std::vector<cv::Point2f>& pts) const;
};

/**
 * @brief Camera to Body frame extrinsics
 */
struct CameraExtrinsics {
    Eigen::Matrix3d R_BC;  ///< Camera-to-Body rotation
    Eigen::Vector3d t_BC;  ///< Camera-to-Body translation
    
    double rotation_residual_deg = 0.0;  ///< Calibration quality
    
    /**
     * @brief Default constructor (identity)
     */
    CameraExtrinsics() 
        : R_BC(Eigen::Matrix3d::Identity()), 
          t_BC(Eigen::Vector3d::Zero()) {}
    
    /**
     * @brief Get Body-to-Camera rotation (inverse)
     */
    Eigen::Matrix3d R_CB() const {
        return R_BC.transpose();
    }
    
    /**
     * @brief Create extrinsics from camera tilt angle
     * 
     * @param tilt_deg Camera tilt below horizontal (degrees)
     * @param roll_offset_deg Roll offset (degrees)
     * @param yaw_offset_deg Yaw offset (degrees)
     * @param translation Translation vector
     */
    static CameraExtrinsics fromTiltAngle(
        double tilt_deg,
        double roll_offset_deg = 0.0,
        double yaw_offset_deg = 0.0,
        const Eigen::Vector3d& translation = Eigen::Vector3d::Zero()
    );
    
    /**
     * @brief Transform points from Camera frame to Body frame
     */
    std::vector<Eigen::Vector3d> transformToBody(
        const std::vector<Eigen::Vector3d>& pts_camera
    ) const;
    
    /**
     * @brief Transform points from Body frame to Camera frame
     */
    std::vector<Eigen::Vector3d> transformToCamera(
        const std::vector<Eigen::Vector3d>& pts_body
    ) const;
};

/**
 * @brief Time synchronization parameters
 */
struct TimeSync {
    double time_offset = 0.0;  ///< t_aligned = t_rpy + time_offset
    double sync_residual_ms = 0.0;  ///< Calibration quality
    
    double alignTimestamp(double t_rpy) const {
        return t_rpy + time_offset;
    }
};

/**
 * @brief Complete calibration bundle
 */
struct CalibrationData {
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
    TimeSync time_sync;
    FrameConventions conventions;
    
    // Metadata
    std::string calibration_date;
    std::string camera_model;
    std::string notes;
    
    /**
     * @brief Create default calibration for testing
     */
    static CalibrationData createDefault(
        int image_width = 1280,
        int image_height = 720,
        double camera_tilt_deg = 60.0
    );
};

} // namespace altitude_estimator

