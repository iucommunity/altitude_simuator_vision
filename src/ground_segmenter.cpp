/**
 * @file ground_segmenter.cpp
 * @brief Stub implementation for GroundSegmenter
 */

#include "altitude_estimator/ground_segmenter.hpp"
#include <opencv2/imgproc.hpp>

namespace altitude_estimator {

GroundSegmenter::GroundSegmenter(const Config& config) : cfg_(config) {
}

std::pair<cv::Mat, cv::Mat> GroundSegmenter::segment(const cv::Mat& image, double timestamp) {
    // Simple heuristic: assume lower portion of image is ground
    int h = image.rows;
    int w = image.cols;
    
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    cv::Mat confidence = cv::Mat::zeros(h, w, CV_32FC1);
    
    // Lower 70% is ground
    for (int y = int(h * 0.3); y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            mask.at<uchar>(y, x) = 255;
            confidence.at<float>(y, x) = float(y) / h;
        }
    }
    
    last_mask_ = mask;
    last_timestamp_ = timestamp;
    
    return {mask, confidence};
}

} // namespace altitude_estimator

