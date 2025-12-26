/**
 * @file image_undistorter.cpp
 * @brief Implementation of ImageUndistorter
 */

#include "altitude_estimator/image_undistorter.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace altitude_estimator {

ImageUndistorter::ImageUndistorter(
    const cv::Mat& K_original,
    const cv::Mat& dist_coeffs,
    const cv::Size& image_size,
    double alpha
) {
    initialize(K_original, dist_coeffs, image_size, alpha);
}

ImageUndistorter::ImageUndistorter(
    double fx, double fy, double cx, double cy,
    const std::vector<double>& dist_coeffs,
    int image_width, int image_height,
    double alpha
) {
    // Build K matrix
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx;
    K.at<double>(1, 1) = fy;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;
    
    // Build distortion coefficients
    cv::Mat dist;
    if (dist_coeffs.empty()) {
        dist = cv::Mat::zeros(5, 1, CV_64F);
    } else {
        dist = cv::Mat(static_cast<int>(dist_coeffs.size()), 1, CV_64F);
        for (size_t i = 0; i < dist_coeffs.size(); ++i) {
            dist.at<double>(static_cast<int>(i)) = dist_coeffs[i];
        }
    }
    
    initialize(K, dist, cv::Size(image_width, image_height), alpha);
}

void ImageUndistorter::initialize(
    const cv::Mat& K_original,
    const cv::Mat& dist_coeffs,
    const cv::Size& image_size,
    double alpha
) {
    K_original_ = K_original.clone();
    dist_coeffs_ = dist_coeffs.clone();
    original_size_ = image_size;
    
    // Check if undistortion is actually needed
    // (if all distortion coefficients are zero or very small, skip undistortion)
    needs_undistortion_ = false;
    if (!dist_coeffs_.empty()) {
        for (int i = 0; i < dist_coeffs_.rows; ++i) {
            if (std::abs(dist_coeffs_.at<double>(i)) > 1e-10) {
                needs_undistortion_ = true;
                break;
            }
        }
    }
    
    if (!needs_undistortion_) {
        // No undistortion needed - new K equals original K
        K_new_ = K_original_.clone();
        new_image_size_ = original_size_;
        valid_roi_ = cv::Rect(0, 0, image_size.width, image_size.height);
        return;
    }
    
    // Compute optimal new camera matrix
    // alpha = 0: All pixels in undistorted image are valid (crops black borders)
    // alpha = 1: All source pixels are retained (may have black borders)
    K_new_ = cv::getOptimalNewCameraMatrix(
        K_original_,
        dist_coeffs_,
        image_size,
        alpha,
        image_size,  // new image size (same as original)
        &valid_roi_
    );
    new_image_size_ = image_size;
    
    // Precompute undistortion maps for efficient remap
    cv::initUndistortRectifyMap(
        K_original_,
        dist_coeffs_,
        cv::Mat(),      // R (no rectification rotation)
        K_new_,         // New camera matrix
        new_image_size_,
        CV_32FC1,       // Map type (float, single channel)
        map1_,
        map2_
    );
}

cv::Mat ImageUndistorter::undistort(const cv::Mat& distorted) const {
    if (!needs_undistortion_) {
        return distorted.clone();
    }
    
    cv::Mat undistorted;
    cv::remap(distorted, undistorted, map1_, map2_, cv::INTER_LINEAR);
    return undistorted;
}

void ImageUndistorter::undistortInPlace(cv::Mat& image) const {
    if (!needs_undistortion_) {
        return;
    }
    
    cv::Mat undistorted;
    cv::remap(image, undistorted, map1_, map2_, cv::INTER_LINEAR);
    image = undistorted;
}

Eigen::Matrix3d ImageUndistorter::getNewK() const {
    Eigen::Matrix3d K;
    K << K_new_.at<double>(0, 0), K_new_.at<double>(0, 1), K_new_.at<double>(0, 2),
         K_new_.at<double>(1, 0), K_new_.at<double>(1, 1), K_new_.at<double>(1, 2),
         K_new_.at<double>(2, 0), K_new_.at<double>(2, 1), K_new_.at<double>(2, 2);
    return K;
}

Eigen::Matrix3d ImageUndistorter::getOriginalK() const {
    Eigen::Matrix3d K;
    K << K_original_.at<double>(0, 0), K_original_.at<double>(0, 1), K_original_.at<double>(0, 2),
         K_original_.at<double>(1, 0), K_original_.at<double>(1, 1), K_original_.at<double>(1, 2),
         K_original_.at<double>(2, 0), K_original_.at<double>(2, 1), K_original_.at<double>(2, 2);
    return K;
}

} // namespace altitude_estimator

