/**
 * @file ground_segmenter.hpp
 * @brief Ground segmentation from images
 */

#pragma once

#include "config.hpp"
#include <opencv2/core.hpp>

namespace altitude_estimator {

class GroundSegmenter {
public:
    GroundSegmenter(const Config& config);
    
    std::pair<cv::Mat, cv::Mat> segment(const cv::Mat& image, double timestamp);
    
private:
    const Config& cfg_;
    cv::Mat last_mask_;
    double last_timestamp_ = 0.0;
};

} // namespace altitude_estimator

