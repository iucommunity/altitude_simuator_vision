/**
 * @file image_undistorter.hpp
 * @brief Image undistortion with optimal camera matrix computation
 */

#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>

namespace altitude_estimator {

/**
 * @brief Handles image undistortion and computes new camera matrix
 * 
 * When undistorting an image, the original camera matrix K becomes invalid.
 * This class computes the optimal new camera matrix K' for undistorted images
 * and provides efficient image undistortion using precomputed maps.
 */
class ImageUndistorter {
public:
    /**
     * @brief Construct undistorter from camera parameters
     * 
     * @param K_original Original camera matrix (3x3)
     * @param dist_coeffs Distortion coefficients (k1, k2, p1, p2, k3)
     * @param image_size Image dimensions (width, height)
     * @param alpha Free scaling parameter (0=crop, 1=keep all pixels)
     */
    ImageUndistorter(
        const cv::Mat& K_original,
        const cv::Mat& dist_coeffs,
        const cv::Size& image_size,
        double alpha = 0.0
    );
    
    /**
     * @brief Construct undistorter from intrinsic values
     * 
     * @param fx, fy Focal lengths in pixels
     * @param cx, cy Principal point in pixels
     * @param dist_coeffs Distortion coefficients (k1, k2, p1, p2, k3)
     * @param image_width, image_height Image dimensions
     * @param alpha Free scaling parameter (0=crop, 1=keep all pixels)
     */
    ImageUndistorter(
        double fx, double fy, double cx, double cy,
        const std::vector<double>& dist_coeffs,
        int image_width, int image_height,
        double alpha = 0.0
    );
    
    /**
     * @brief Check if undistortion is needed
     * @return true if distortion coefficients are non-zero
     */
    bool needsUndistortion() const { return needs_undistortion_; }
    
    /**
     * @brief Undistort an image
     * @param distorted Input distorted image
     * @return Undistorted image
     */
    cv::Mat undistort(const cv::Mat& distorted) const;
    
    /**
     * @brief Undistort image in-place
     * @param image Image to undistort (modified in place)
     */
    void undistortInPlace(cv::Mat& image) const;
    
    /**
     * @brief Get the NEW camera matrix for undistorted images
     * @return 3x3 camera matrix as cv::Mat (CV_64F)
     */
    cv::Mat getNewK_cv() const { return K_new_.clone(); }
    
    /**
     * @brief Get the NEW camera matrix for undistorted images (Eigen)
     * @return 3x3 camera matrix as Eigen::Matrix3d
     */
    Eigen::Matrix3d getNewK() const;
    
    /**
     * @brief Get the ORIGINAL camera matrix
     * @return 3x3 camera matrix as cv::Mat (CV_64F)
     */
    cv::Mat getOriginalK_cv() const { return K_original_.clone(); }
    
    /**
     * @brief Get the ORIGINAL camera matrix (Eigen)
     * @return 3x3 camera matrix as Eigen::Matrix3d
     */
    Eigen::Matrix3d getOriginalK() const;
    
    /**
     * @brief Get new focal lengths after undistortion
     */
    double getNewFx() const { return K_new_.at<double>(0, 0); }
    double getNewFy() const { return K_new_.at<double>(1, 1); }
    double getNewCx() const { return K_new_.at<double>(0, 2); }
    double getNewCy() const { return K_new_.at<double>(1, 2); }
    
    /**
     * @brief Get undistorted image size (may differ from original if alpha != 0)
     */
    cv::Size getNewImageSize() const { return new_image_size_; }
    
    /**
     * @brief Get the valid ROI in undistorted image (area with valid pixels)
     */
    cv::Rect getValidROI() const { return valid_roi_; }
    
private:
    void initialize(
        const cv::Mat& K_original,
        const cv::Mat& dist_coeffs,
        const cv::Size& image_size,
        double alpha
    );
    
    bool needs_undistortion_;
    cv::Mat K_original_;
    cv::Mat K_new_;
    cv::Mat dist_coeffs_;
    cv::Size original_size_;
    cv::Size new_image_size_;
    cv::Rect valid_roi_;
    
    // Precomputed undistortion maps for efficiency
    cv::Mat map1_, map2_;
};

} // namespace altitude_estimator

