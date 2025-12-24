/**
 * @file smoother.cpp
 * @brief Implementation of FixedLagLogDistanceSmoother with IRLS solver
 */

#include "altitude_estimator/smoother.hpp"
#include <cmath>
#include <algorithm>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/QR>

namespace altitude_estimator {

FixedLagLogDistanceSmoother::FixedLagLogDistanceSmoother(const Config& config, int window_size)
    : config_(config), window_size_(window_size) {
    huber_k_ = config.huber_k;
}

void FixedLagLogDistanceSmoother::addAnchor(
    double timestamp,
    double altitude,
    double sigma,
    const Eigen::Matrix3d& R_CW,
    const Eigen::Vector3d* n_cam
) {
    // Compute vertical factor from plane normal if provided
    if (n_cam) {
        Eigen::Vector3d n_world = R_CW.transpose() * (*n_cam);
        Eigen::Vector3d world_up(0.0, 0.0, -1.0);  // NED
        double dot = std::abs(n_world.dot(world_up));
        vertical_factor_ = std::clamp(dot, 0.1, 1.0);
    }
    
    // Convert altitude to plane distance
    double d = std::max(altitude / vertical_factor_, d_min_);
    double log_d = std::log(d);
    double sigma_log = sigma / d;
    
    // Add state
    SmootherState state;
    state.state_id = next_state_id_++;
    state.timestamp = timestamp;
    state.log_d = log_d;
    state.sigma_log_d = sigma_log;
    state.d = d;
    state.h = altitude;
    state.is_anchor = true;
    
    states_.push_back(state);
    
    // Add anchor factor
    anchor_factors_.push_back({state.state_id, log_d, sigma_log});
    
    // Update current estimate
    current_log_d_ = log_d;
    current_d_ = d;
    current_h_ = altitude;
    current_sigma_log_d_ = sigma_log;
    mode_ = AltitudeMode::GEOM;
    last_good_update_ = timestamp;
}

bool FixedLagLogDistanceSmoother::addVisionConstraint(
    double timestamp,
    const HomographyConstraint& constraint,
    const Eigen::Matrix3d& R_CW
) {
    if (!constraint.isValid()) {
        return false;
    }
    
    // Update vertical factor if enabled
    if (config_.enable_vertical_factor_update) {
        Eigen::Vector3d n_world = R_CW.transpose() * constraint.n_cam;
        Eigen::Vector3d world_up(0.0, 0.0, -1.0);
        double dot = std::abs(n_world.dot(world_up));
        double new_factor = std::clamp(dot, 0.1, 1.0);
        vertical_factor_ = 0.9 * vertical_factor_ + 0.1 * new_factor;
    }
    
    if (states_.empty()) {
        return false;
    }
    
    // Predict new state from constraint
    const auto& prev_state = states_.back();
    double new_log_d = prev_state.log_d + constraint.log_s;
    double new_d = std::exp(new_log_d);
    
    if (new_d < d_min_) {
        new_d = d_min_;
        new_log_d = std::log(d_min_);
    }
    
    double new_h = new_d * vertical_factor_;
    
    // Add new state
    SmootherState new_state;
    new_state.state_id = next_state_id_++;
    new_state.timestamp = timestamp;
    new_state.log_d = new_log_d;
    new_state.sigma_log_d = constraint.sigma_r;
    new_state.d = new_d;
    new_state.h = new_h;
    new_state.is_anchor = false;
    
    states_.push_back(new_state);
    
    // Add vision factor
    vision_factors_.push_back({prev_state.state_id, new_state.state_id, 
                               constraint.log_s, constraint.sigma_r});
    
    // Maintain window size (keep at least one state for anchor)
    while (states_.size() > size_t(window_size_) && states_.size() > 1) {
        states_.pop_front();
    }
    while (vision_factors_.size() > size_t(window_size_)) {
        vision_factors_.pop_front();
    }
    
    // Solve the window (ALWAYS, like Python does)
    // solve() will extract current estimate from last state after solving
    solve();
    
    mode_ = AltitudeMode::GEOM;
    hold_start_time_ = std::nullopt;
    last_good_update_ = timestamp;
    
    return true;
}

void FixedLagLogDistanceSmoother::predict(double timestamp, double dt) {
    // Inflate uncertainty over time
    current_sigma_log_d_ = std::sqrt(
        current_sigma_log_d_ * current_sigma_log_d_ + (0.01 * dt) * (0.01 * dt)
    );
    age_of_last_good_update_ = timestamp - last_good_update_;
}

void FixedLagLogDistanceSmoother::enterHold(double timestamp) {
    if (!hold_start_time_) {
        hold_start_time_ = timestamp;
    }
    
    double hold_duration = timestamp - *hold_start_time_;
    
    if (hold_duration > config_.hold_timeout_sec) {
        mode_ = AltitudeMode::LOST;
    } else {
        mode_ = AltitudeMode::HOLD;
        current_sigma_log_d_ *= 1.05;
    }
}

void FixedLagLogDistanceSmoother::solve() {
    // IRLS solver for fixed-lag least squares with Huber loss
    size_t n_states = states_.size();
    if (n_states == 0) {
        return;
    }
    
    // If only one state, no need to solve
    if (n_states == 1) {
        const auto& last = states_.back();
        current_log_d_ = last.log_d;
        current_d_ = last.d;
        current_h_ = last.h;
        current_sigma_log_d_ = last.sigma_log_d;
        return;
    }
    
    // Build state_id -> index map
    std::map<StateId, size_t> id_to_idx;
    for (size_t i = 0; i < states_.size(); ++i) {
        id_to_idx[states_[i].state_id] = i;
    }
    
    // Initial estimate from current states
    Eigen::VectorXd x(n_states);
    for (size_t i = 0; i < n_states; ++i) {
        x(i) = states_[i].log_d;
    }
    
    // Iteratively reweighted least squares (IRLS) for Huber
    for (int iteration = 0; iteration < 5; ++iteration) {
        // Build normal equations: J^T W J x = J^T W r
        Eigen::MatrixXd JtWJ = Eigen::MatrixXd::Zero(n_states, n_states);
        Eigen::VectorXd JtWr = Eigen::VectorXd::Zero(n_states);
        
        // Anchor factors (absolute constraints)
        for (const auto& factor : anchor_factors_) {
            StateId state_id = std::get<0>(factor);
            double log_d_anchor = std::get<1>(factor);
            double sigma = std::get<2>(factor);
            auto it = id_to_idx.find(state_id);
            if (it == id_to_idx.end()) {
                continue;  // State has been dropped from window
            }
            size_t idx = it->second;
            double residual = x(idx) - log_d_anchor;
            double weight = huberWeight(residual, sigma);
            double w_sigma2 = weight / (sigma * sigma);
            JtWJ(idx, idx) += w_sigma2;
            JtWr(idx) += w_sigma2 * residual;  // Match Python: weight * residual / sigma^2
        }
        
        // Vision factors (relative constraints)
        for (const auto& factor : vision_factors_) {
            StateId id_prev = std::get<0>(factor);
            StateId id_curr = std::get<1>(factor);
            double log_s = std::get<2>(factor);
            double sigma = std::get<3>(factor);
            
            auto it_prev = id_to_idx.find(id_prev);
            auto it_curr = id_to_idx.find(id_curr);
            if (it_prev == id_to_idx.end() || it_curr == id_to_idx.end()) {
                continue;  // One or both states dropped from window
            }
            size_t idx_prev = it_prev->second;
            size_t idx_curr = it_curr->second;
            
            // Residual: log_d_curr - log_d_prev - log_s
            double residual = x(idx_curr) - x(idx_prev) - log_s;
            double weight = huberWeight(residual, sigma);
            double w_sigma2 = weight / (sigma * sigma);
            
            // Jacobian contributions
            JtWJ(idx_prev, idx_prev) += w_sigma2;
            JtWJ(idx_curr, idx_curr) += w_sigma2;
            JtWJ(idx_prev, idx_curr) -= w_sigma2;
            JtWJ(idx_curr, idx_prev) -= w_sigma2;
            
            JtWr(idx_prev) -= w_sigma2 * residual;
            JtWr(idx_curr) += w_sigma2 * residual;
        }
        
        // Add small regularization for stability
        JtWJ += Eigen::MatrixXd::Identity(n_states, n_states) * 1e-6;
        
        // Solve
        Eigen::LDLT<Eigen::MatrixXd> ldlt(JtWJ);
        if (ldlt.info() != Eigen::Success) {
            // Solve failed, break
            break;
        }
        Eigen::VectorXd dx = ldlt.solve(-JtWr);
        x = x + dx;
        
        if (dx.norm() < 1e-6) {
            break;
        }
    }
    
    // Update states with solution (states are mutable in deque)
    for (size_t i = 0; i < n_states; ++i) {
        SmootherState& state = states_[i];
        state.log_d = x(i);
        state.d = std::exp(x(i));
        state.h = state.d * vertical_factor_;
    }
    
    // Extract current estimate from last state (should be updated now)
    if (!states_.empty()) {
        const SmootherState& last = states_.back();
        current_log_d_ = last.log_d;
        current_d_ = last.d;
        current_h_ = last.h;
        current_sigma_log_d_ = last.sigma_log_d;
    }
}

double FixedLagLogDistanceSmoother::huberWeight(double residual, double sigma) const {
    double r_normalized = std::abs(residual) / sigma;
    if (r_normalized <= huber_k_) {
        return 1.0;
    } else {
        return huber_k_ / r_normalized;
    }
}

AltitudeEstimate FixedLagLogDistanceSmoother::getEstimate(double timestamp) const {
    AltitudeEstimate est;
    est.altitude_m = current_h_;
    est.sigma_m = current_sigma_log_d_ * current_h_;
    est.mode = mode_;
    est.timestamp = timestamp;
    est.altitude_homography = current_h_;
    est.ground_quality = (mode_ == AltitudeMode::GEOM) ? 1.0 : 0.0;
    
    return est;
}

std::map<std::string, double> FixedLagLogDistanceSmoother::getDebugInfo() const {
    std::map<std::string, double> info;
    info["n_states"] = states_.size();
    info["n_anchors"] = anchor_factors_.size();
    info["n_vision_factors"] = vision_factors_.size();
    info["current_d"] = current_d_;
    info["current_h"] = current_h_;
    info["current_sigma_log_d"] = current_sigma_log_d_;
    info["vertical_factor"] = vertical_factor_;
    info["mode"] = static_cast<double>(mode_);
    return info;
}

} // namespace altitude_estimator
