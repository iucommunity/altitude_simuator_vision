/**
 * @file config.hpp
 * @brief System configuration parameters
 */

#pragma once

#include <string>

namespace altitude_estimator {

/**
 * @brief System configuration with all tunable parameters
 * 
 * PRIMARY PATH (Homography-based): Uses feature tracking + homography constraints
 * SECONDARY PATH (SLAM/Depth): Placeholder for future expansion
 */
struct Config {
    // =========================================================================
    // CORE PARAMETERS (PRIMARY PATH)
    // =========================================================================
    
    /// Frame rate (Hz) - set this to match your camera FPS
    double fps = 30.0;
    
    // --- Feature Tracking ---
    int max_features = 500;           ///< Max features to track
    double feature_quality = 0.01;    ///< GFTT quality threshold
    int min_feature_distance = 10;    ///< Min pixel distance between features
    int lk_win_size_width = 21;       ///< LK optical flow window width
    int lk_win_size_height = 21;      ///< LK optical flow window height
    int lk_max_level = 3;             ///< LK pyramid levels
    
    // --- Geometric Filtering ---
    double ransac_reproj_threshold = 2.0;  ///< RANSAC threshold (pixels)
    int min_inliers = 20;                  ///< Min inliers for homography
    double max_rotation_deg = 10.0;        ///< Max rotation per frame (deg)
    
    // --- Initialization ---
    int init_window_size = 10;        ///< Frames needed for initialization
    
    // --- Smoother ---
    int window_size = 12;             ///< Fixed-lag smoother window size
    double hold_timeout_sec = 2.0;    ///< Timeout before LOST mode
    double huber_k = 1.345;           ///< Huber loss parameter
    
    // --- Homography Constraints ---
    bool enable_rank1_gate = false;   ///< Enable rank-1 constraint check
    double rank1_thresh_s2 = 0.50;    ///< Rank-1 s2/s1 threshold
    double rank1_thresh_s3 = 0.30;    ///< Rank-1 s3/s1 threshold
    double homography_s_min = 0.3;    ///< Min scale factor
    double homography_s_max = 3.0;    ///< Max scale factor
    bool enable_vertical_factor_update = true;  ///< Update vertical factor from normal
    int homography_sign_convention = 1;  ///< +1 for OpenCV/Unity
    
    // =========================================================================
    // SECONDARY PATH (FUTURE/DEBUG)
    // =========================================================================
    
    // --- Ground Plane Fitting (stub) ---
    double ground_ransac_threshold = 0.1;       ///< meters
    int min_ground_inliers = 10;
    double ground_normal_tolerance_deg = 20.0;
    
    // --- Depth Prior (not implemented) ---
    bool use_depth_prior = false;
    std::string depth_model_path;
    double depth_confidence_threshold = 0.5;
    
    // --- Ground Segmentation (stub) ---
    bool use_ground_segmentation = false;
    std::string segmentation_model_path;
    double ground_mask_threshold = 0.5;
    
    // --- SLAM/VO (stub) ---
    int max_keyframes = 30;
    int min_keyframe_gap = 3;
    double triangulation_min_parallax_deg = 2.0;
    int ba_iterations = 10;
    double min_parallax_deg = 1.0;
    int min_init_keyframes = 3;
    double init_altitude_tolerance = 0.1;
    
    // --- Fusion (not implemented) ---
    double fusion_window_sec = 2.0;
    double altitude_process_sigma = 0.1;
    double altitude_velocity_sigma = 0.5;
    double max_vertical_speed_mps = 5.0;
    double slam_quality_threshold = 0.3;
    double ground_quality_threshold = 0.3;
    double depth_quality_threshold = 0.3;
    double recovery_min_quality = 0.5;
    
    // Deprecated/unused
    double keyframe_rate = 5.0;
    double depth_rate = 5.0;
    double segmentation_rate = 5.0;
};

} // namespace altitude_estimator

