/**
 * @file config.hpp
 * @brief System configuration parameters
 */

#pragma once

namespace altitude_estimator {

/**
 * @brief System configuration with all tunable parameters
 */
struct Config {
    // Frame rate and timing
    double fps = 30.0;
    double keyframe_rate = 5.0;
    double depth_rate = 5.0;
    double segmentation_rate = 5.0;
    
    // Feature tracking
    int max_features = 500;
    double feature_quality = 0.01;
    int min_feature_distance = 10;
    cv::Size lk_win_size = cv::Size(21, 21);
    int lk_max_level = 3;
    
    // Geometric filtering
    double ransac_reproj_threshold = 2.0;  ///< pixels
    int min_inliers = 20;
    double min_parallax_deg = 1.0;
    double max_rotation_deg = 10.0;
    
    // Ground plane fitting
    double ground_ransac_threshold = 0.1;  ///< meters
    int min_ground_inliers = 10;
    double ground_normal_tolerance_deg = 20.0;
    
    // Depth prior
    bool use_depth_prior = false;
    std::string depth_model_path;
    double depth_confidence_threshold = 0.5;
    
    // Ground segmentation
    bool use_ground_segmentation = true;
    std::string segmentation_model_path;
    double ground_mask_threshold = 0.5;
    
    // Initialization
    int init_window_size = 10;
    int min_init_keyframes = 3;
    double init_altitude_tolerance = 0.1;
    
    // SLAM / VO
    int window_size = 12;
    int max_keyframes = 30;
    int min_keyframe_gap = 3;
    double triangulation_min_parallax_deg = 2.0;
    int ba_iterations = 10;
    
    // Fusion estimator
    double fusion_window_sec = 2.0;
    double altitude_process_sigma = 0.1;
    double altitude_velocity_sigma = 0.5;
    double max_vertical_speed_mps = 5.0;
    double huber_k = 1.345;
    
    // Quality thresholds
    double slam_quality_threshold = 0.3;
    double ground_quality_threshold = 0.3;
    double depth_quality_threshold = 0.3;
    
    // Failure handling
    double hold_timeout_sec = 2.0;
    double recovery_min_quality = 0.5;
    
    // Homography constraints
    bool enable_rank1_gate = false;
    double rank1_thresh_s2 = 0.50;
    double rank1_thresh_s3 = 0.30;
    
    double homography_s_min = 0.3;
    double homography_s_max = 3.0;
    
    bool enable_vertical_factor_update = false;
    int homography_sign_convention = 1;  ///< +1 for OpenCV/Unity
};

} // namespace altitude_estimator

