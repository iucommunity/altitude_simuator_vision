/**
 * @file homography_altimeter.hpp
 * @brief Dominant-plane homography altimeter (PRIMARY path)
 */

#pragma once

#include "common.hpp"
#include "calibration.hpp"
#include "config.hpp"
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <map>
#include <string>

namespace altitude_estimator {

/**
 * @brief Output from HomographyAltimeter::computeConstraint()
 */
struct HomographyConstraint {
    double s;                    ///< Scale factor: d_{k+1} = d_k * s
    double log_s;                ///< log(s) for log-space smoother
    double sigma_r;              ///< Uncertainty on log residual
    Eigen::Vector3d n_cam;       ///< Plane normal in camera frame (unit vector)
    Eigen::Matrix3d R_rel;       ///< Refined relative rotation used
    std::map<std::string, double> metrics;  ///< Debug metrics
    
    bool isValid() const {
        return s > 0 && std::isfinite(log_s) && sigma_r < std::numeric_limits<double>::infinity();
    }
};

/**
 * @brief Dominant-plane homography altimeter (PRIMARY path)
 */
class HomographyAltimeter {
public:
    HomographyAltimeter(
        const CalibrationData& calibration,
        const Config& config,
        const FrameConventions& conventions
    );
    
    /**
     * @brief Compute distance constraint from homography
     * 
     * @param pts_prev Previous frame points (N, 2)
     * @param pts_curr Current frame points (N, 2)
     * @param R_rel_rpy Relative rotation from RPY (prior)
     * @param R_CW_curr Current world->camera rotation (for ground gate)
     * @param grid_shape Grid for coverage computation
     * @return HomographyConstraint if all gates pass, nullopt otherwise
     */
    std::optional<HomographyConstraint> computeConstraint(
        const std::vector<cv::Point2f>& pts_prev,
        const std::vector<cv::Point2f>& pts_curr,
        const Eigen::Matrix3d& R_rel_rpy,
        const Eigen::Matrix3d& R_CW_curr,
        const cv::Size& grid_shape = cv::Size(4, 4)
    );
    
    int consecutiveFailures() const { return consecutive_failures_; }
    const std::map<std::string, double>& lastFailMetrics() const { return last_fail_metrics_; }
    
private:
    double computeCoverage(
        const std::vector<cv::Point2f>& pts,
        const cv::Size& grid_shape
    ) const;
    
    struct CandidateSelection {
        int best_idx;
        double best_score;
        Eigen::Matrix3d R_rel;
        Eigen::Vector3d n_cam;
        Eigen::Vector3d u;  // t/d
    };
    
    CandidateSelection selectBestCandidate(
        const std::vector<cv::Mat>& rotations,
        const std::vector<cv::Mat>& translations,
        const std::vector<cv::Mat>& normals,
        const Eigen::Matrix3d& R_rel_rpy,
        const std::vector<cv::Point2f>& pts_prev,
        const std::vector<cv::Point2f>& pts_curr
    );
    
    double computeSigma(const std::map<std::string, double>& metrics) const;
    
    const CalibrationData& calib_;
    Config config_;  // Store by value
    const FrameConventions& conventions_;
    
    Eigen::Vector3d world_up_;
    
    // Gating thresholds
    int min_inliers_ = 20;
    double min_coverage_ = 0.20;
    double max_rmse_px_ = 3.0;
    double max_slope_deg_ = 70.0;
    
    // State for continuity
    std::optional<Eigen::Vector3d> prev_n_cam_;
    std::optional<Eigen::Matrix3d> prev_R_rel_;
    int consecutive_failures_ = 0;
    std::map<std::string, double> last_fail_metrics_;
};

} // namespace altitude_estimator

