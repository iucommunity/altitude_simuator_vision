/**
 * @file homography_altimeter.cpp
 * @brief Full implementation of HomographyAltimeter matching Python version
 */

#include "altitude_estimator/homography_altimeter.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <cmath>
#include <set>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace altitude_estimator {

HomographyAltimeter::HomographyAltimeter(
    const CalibrationData& calibration,
    const Config& config,
    const FrameConventions& conventions
) : calib_(calibration), config_(config), conventions_(conventions) {
    
    // Set world up vector
    if (conventions.world_frame == CoordinateFrame::NED) {
        world_up_ = Eigen::Vector3d(0.0, 0.0, -1.0);
    } else {
        world_up_ = Eigen::Vector3d(0.0, 0.0, 1.0);
    }
    
    min_inliers_ = config.min_inliers;
    min_coverage_ = 0.20;
    max_rmse_px_ = 3.0;
    max_slope_deg_ = 70.0;
}

double HomographyAltimeter::computeCoverage(
    const std::vector<cv::Point2f>& pts,
    const cv::Size& grid_shape
) const {
    if (pts.empty()) return 0.0;
    
    int gh = grid_shape.height;
    int gw = grid_shape.width;
    double cell_h = double(calib_.intrinsics.height) / gh;
    double cell_w = double(calib_.intrinsics.width) / gw;
    
    std::set<std::pair<int, int>> occupied;
    for (const auto& pt : pts) {
        int ci = std::clamp(int(pt.x / cell_w), 0, gw - 1);
        int ri = std::clamp(int(pt.y / cell_h), 0, gh - 1);
        occupied.insert({ri, ci});
    }
    
    return double(occupied.size()) / (gh * gw);
}

std::optional<HomographyConstraint> HomographyAltimeter::computeConstraint(
    const std::vector<cv::Point2f>& pts_prev,
    const std::vector<cv::Point2f>& pts_curr,
    const Eigen::Matrix3d& R_rel_rpy,
    const Eigen::Matrix3d& R_CW_curr,
    const cv::Size& grid_shape
) {
    std::map<std::string, double> metrics;
    metrics["n_input"] = pts_prev.size();
    metrics["n_inliers"] = 0;
    metrics["inlier_ratio"] = 0.0;
    metrics["coverage"] = 0.0;
    metrics["rmse_px"] = std::numeric_limits<double>::infinity();
    metrics["rank1_s2s1"] = 1.0;
    metrics["rank1_s3s1"] = 1.0;
    metrics["rot_dist_deg"] = 0.0;
    metrics["ground_likeness"] = 0.0;
    metrics["gate_failed"] = 0.0;
    
    // Gate 1: Minimum points
    if (pts_prev.size() < size_t(min_inliers_)) {
        metrics["gate_failed"] = 1.0;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // Undistort points - CRITICAL for correct homography computation
    std::vector<cv::Point2f> pts_prev_u, pts_curr_u;
    try {
        pts_prev_u = calib_.intrinsics.undistortPoints(pts_prev);
        pts_curr_u = calib_.intrinsics.undistortPoints(pts_curr);
    } catch (const cv::Exception& e) {
        // If undistortion fails, return nullopt (can't compute correct homography)
        metrics["gate_failed"] = 1.5;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    } catch (...) {
        metrics["gate_failed"] = 1.5;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // 1. Robust homography (match Python: USAC_MAGSAC)
    cv::Mat H, mask;
    H = cv::findHomography(
        pts_prev_u, pts_curr_u,
        cv::USAC_MAGSAC,
        config_.ransac_reproj_threshold,
        mask,
        10000,   // maxIters (robust, closer to Python behavior)
        0.999    // confidence
    );
    
    if (H.empty()) {
        metrics["gate_failed"] = 2.0;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // Extract inliers
    std::vector<bool> inlier_mask(mask.rows);
    int n_inliers = 0;
    for (int i = 0; i < mask.rows; ++i) {
        inlier_mask[i] = (mask.at<uchar>(i) != 0);
        if (inlier_mask[i]) n_inliers++;
    }
    metrics["n_inliers"] = n_inliers;
    metrics["inlier_ratio"] = double(n_inliers) / pts_prev.size();
    
    // Gate 2: Minimum inliers
    if (n_inliers < min_inliers_) {
        metrics["gate_failed"] = 3.0;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // 2. Coverage check
    std::vector<cv::Point2f> inlier_pts;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_pts.push_back(pts_curr_u[i]);
        }
    }
    double coverage = computeCoverage(inlier_pts, grid_shape);
    metrics["coverage"] = coverage;
    
    if (coverage < min_coverage_) {
        metrics["gate_failed"] = 4.0;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // 3. Reprojection RMSE
    std::vector<cv::Point2f> pts_proj;
    std::vector<cv::Point3f> pts_prev_h;
    for (const auto& pt : pts_prev_u) {
        pts_prev_h.push_back(cv::Point3f(pt.x, pt.y, 1.0f));
    }
    
    cv::Mat H_float;
    H.convertTo(H_float, CV_32F);
    
    for (const auto& pt_h : pts_prev_h) {
        cv::Mat pt_mat = (cv::Mat_<float>(3, 1) << pt_h.x, pt_h.y, pt_h.z);
        cv::Mat proj_mat = H_float * pt_mat;
        float w = proj_mat.at<float>(2);
        if (std::abs(w) > 1e-6) {
            pts_proj.push_back(cv::Point2f(
                proj_mat.at<float>(0) / w,
                proj_mat.at<float>(1) / w
            ));
        } else {
            pts_proj.push_back(cv::Point2f(0, 0));
        }
    }
    
    double rmse_sum = 0.0;
    int rmse_count = 0;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            double err = cv::norm(pts_proj[i] - pts_curr_u[i]);
            rmse_sum += err * err;
            rmse_count++;
        }
    }
    double rmse = std::sqrt(rmse_sum / std::max(1, rmse_count));
    metrics["rmse_px"] = rmse;
    
    if (rmse > max_rmse_px_) {
        metrics["gate_failed"] = 5.0;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // 4. Normalize homography: Hn = K^-1 H K
    cv::Mat K_cv = calib_.intrinsics.K_cv();
    cv::Mat K_inv = K_cv.inv();
    cv::Mat Hn = K_inv * H * K_cv;
    
    // 5. Decompose homography to get candidates
    cv::Mat K_eye = cv::Mat::eye(3, 3, CV_64F);
    std::vector<cv::Mat> rotations, translations, normals;
    int n_solutions = cv::decomposeHomographyMat(Hn, K_eye, rotations, translations, normals);
    
    if (n_solutions == 0) {
        metrics["gate_failed"] = 6.0;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // 6. Select best candidate
    CandidateSelection selection = selectBestCandidate(
        rotations, translations, normals, R_rel_rpy,
        pts_prev_u, pts_curr_u, inlier_mask
    );
    
    if (selection.best_idx < 0) {
        metrics["gate_failed"] = 7.0;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // Compute rotation distance to RPY prior
    Eigen::Matrix3d R_diff = selection.R_rel * R_rel_rpy.transpose();
    double trace = R_diff.trace();
    double rot_dist = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
    metrics["rot_dist_deg"] = rot_dist * 180.0 / M_PI;
    
    // 7. Rank-1 check on A = Hn - R_rel
    Eigen::Matrix3d Hn_eigen, R_rel_eigen;
    cv::cv2eigen(Hn, Hn_eigen);
    R_rel_eigen = selection.R_rel;
    Eigen::Matrix3d A = Hn_eigen - R_rel_eigen;
    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s_vals = svd.singularValues();
    double s2_s1 = s_vals(1) / (s_vals(0) + 1e-9);
    double s3_s1 = s_vals(2) / (s_vals(0) + 1e-9);
    metrics["rank1_s2s1"] = s2_s1;
    metrics["rank1_s3s1"] = s3_s1;
    
    // Rank-1 gate (optional)
    if (config_.enable_rank1_gate) {
        if (s2_s1 > config_.rank1_thresh_s2 || s3_s1 > config_.rank1_thresh_s3) {
            metrics["gate_failed"] = 8.0;
            consecutive_failures_++;
            last_fail_metrics_ = metrics;
            return std::nullopt;
        }
    }
    
    // 8. Compute scale factor s = 1 + sign * (n·u)
    // NOTE: Python uses sign_convention = +1 for OpenCV/Unity convention
    // For candidate filtering, Python uses s = 1 - dot, but final scale uses 1 + sign * dot
    double dot_nu = selection.n_cam.dot(selection.u);
    double s = 1.0 + config_.homography_sign_convention * dot_nu;
    
    metrics["dot_nu"] = dot_nu;
    metrics["s_raw"] = s;
    
    if (s <= 0 || s < config_.homography_s_min || s > config_.homography_s_max) {
        metrics["gate_failed"] = 9.0;
        metrics["s_value"] = s;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // 9. Ground-likeness gate
    Eigen::Vector3d n_world = R_CW_curr.transpose() * selection.n_cam;
    double ground_likeness = std::abs(n_world.dot(world_up_));
    metrics["ground_likeness"] = ground_likeness;
    
    double cos_max_slope = std::cos(max_slope_deg_ * M_PI / 180.0);
    if (ground_likeness < cos_max_slope) {
        metrics["gate_failed"] = 10.0;
        consecutive_failures_++;
        last_fail_metrics_ = metrics;
        return std::nullopt;
    }
    
    // All gates passed
    consecutive_failures_ = 0;
    prev_n_cam_ = selection.n_cam;
    prev_R_rel_ = selection.R_rel;
    
    // Compute uncertainty
    double sigma_r = computeSigma(metrics);
    
    HomographyConstraint constraint;
    constraint.s = s;
    constraint.log_s = std::log(s);
    constraint.sigma_r = sigma_r;
    constraint.n_cam = selection.n_cam;
    constraint.R_rel = selection.R_rel;
    constraint.metrics = metrics;
    
    return constraint;
}

HomographyAltimeter::CandidateSelection HomographyAltimeter::selectBestCandidate(
    const std::vector<cv::Mat>& rotations,
    const std::vector<cv::Mat>& translations,
    const std::vector<cv::Mat>& normals,
    const Eigen::Matrix3d& R_rel_rpy,
    const std::vector<cv::Point2f>& pts_prev,
    const std::vector<cv::Point2f>& pts_curr,
    const std::vector<bool>& inlier_mask
) {
    CandidateSelection best;
    best.best_idx = -1;
    best.best_score = std::numeric_limits<double>::infinity();
    
    Eigen::Matrix3d K_eigen = calib_.intrinsics.K();
    Eigen::Matrix3d K_inv_eigen = calib_.intrinsics.K().inverse();
    
    for (size_t i = 0; i < rotations.size(); ++i) {
        // Convert cv::Mat to Eigen
        // OpenCV returns translations and normals as column vectors (3x1)
        Eigen::Matrix3d R;
        Eigen::Vector3d t_over_d, n;
        cv::cv2eigen(rotations[i], R);
        
        // Extract translation and normal as vectors
        // OpenCV returns them as 3x1 column vectors, cv2eigen can handle this
        cv::cv2eigen(translations[i], t_over_d);
        cv::cv2eigen(normals[i], n);
        
        // Try both sign choices for (n, t/d)
        for (double sign : {1.0, -1.0}) {
            Eigen::Vector3d n_try = n * sign;
            Eigen::Vector3d u_try = t_over_d * sign;
            
            // 1. Cheirality via s: s = 1 - n·u (Python uses this for candidate filtering)
            double s = 1.0 - n_try.dot(u_try);
            if (s <= 0.05 || s < config_.homography_s_min || s > config_.homography_s_max) {
                continue;
            }
            
            // 2. Compute transfer error using H_candidate = R + u @ n.T
            Eigen::Matrix3d Hn_candidate = R + u_try * n_try.transpose();
            Eigen::Matrix3d H_candidate = K_eigen * Hn_candidate * K_inv_eigen;
            
            // Forward transfer error
            std::vector<double> errors;
            size_t n_pts = std::min(pts_prev.size(), std::min(pts_curr.size(), inlier_mask.size()));
            for (size_t j = 0; j < n_pts; ++j) {
                if (!inlier_mask[j]) continue;
                
                Eigen::Vector3d pt_prev_h(pts_prev[j].x, pts_prev[j].y, 1.0);
                Eigen::Vector3d pt_proj_h = H_candidate * pt_prev_h;
                if (std::abs(pt_proj_h(2)) > 1e-6) {
                    Eigen::Vector2d pt_proj(pt_proj_h(0) / pt_proj_h(2), pt_proj_h(1) / pt_proj_h(2));
                    Eigen::Vector2d pt_curr(pts_curr[j].x, pts_curr[j].y);
                    double err = (pt_proj - pt_curr).norm();
                    errors.push_back(err);
                }
            }
            
            if (errors.empty()) continue;
            
            std::sort(errors.begin(), errors.end());
            double transfer_err = errors[errors.size() / 2];  // median
            
            // 3. Rotation distance to RPY prior
            Eigen::Matrix3d R_diff = R * R_rel_rpy.transpose();
            double trace = R_diff.trace();
            double rot_dist = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
            
            // 4. Normal continuity
            double normal_cont = 0.0;
            if (prev_n_cam_) {
                normal_cont = 1.0 - std::abs(n_try.dot(*prev_n_cam_));
            }
            
            // Combined score (lower is better)
            double score = transfer_err + 0.5 * rot_dist + 0.2 * normal_cont;
            
            if (score < best.best_score) {
                best.best_score = score;
                best.best_idx = int(i);
                best.R_rel = R;
                best.n_cam = n_try;
                best.u = u_try;
            }
        }
    }
    
    return best;
}

double HomographyAltimeter::computeSigma(const std::map<std::string, double>& metrics) const {
    double sigma_base = 0.02;
    
    double inlier_ratio = metrics.count("inlier_ratio") ? metrics.at("inlier_ratio") : 0.5;
    double coverage = metrics.count("coverage") ? metrics.at("coverage") : 0.5;
    double rmse_px = metrics.count("rmse_px") ? metrics.at("rmse_px") : 0.0;
    double rank1_s2s1 = metrics.count("rank1_s2s1") ? metrics.at("rank1_s2s1") : 1.0;
    
    double inlier_factor = 1.0 / std::max(inlier_ratio, 0.3);
    double coverage_factor = 1.0 / std::max(coverage, 0.3);
    double rmse_factor = 1.0 + rmse_px / max_rmse_px_;
    double rank1_factor = 1.0 + rank1_s2s1 * 2.0;
    
    double sigma = sigma_base * inlier_factor * coverage_factor * rmse_factor * rank1_factor;
    
    return std::clamp(sigma, 0.01, 0.5);
}

} // namespace altitude_estimator
