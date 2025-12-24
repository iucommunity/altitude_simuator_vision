/**
 * @file smoother.hpp
 * @brief Fixed-lag log-distance smoother with Huber loss
 */

#pragma once

#include "common.hpp"
#include "config.hpp"
#include "data_types.hpp"
#include "homography_altimeter.hpp"
#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <tuple>

namespace altitude_estimator {

/**
 * @brief State in the fixed-lag smoother
 */
struct SmootherState {
    StateId state_id;          ///< Monotonically increasing ID (stable across window shifts)
    double timestamp;
    double log_d;              ///< log(distance to plane)
    double sigma_log_d;        ///< Uncertainty on log_d
    double d;                  ///< Distance (exp(log_d))
    double h;                  ///< Vertical altitude (d * vertical_factor)
    bool is_anchor = false;
};

/**
 * @brief Fixed-lag robust least-squares smoother for plane distance
 * 
 * Operates in log-space with Huber loss for robustness.
 */
class FixedLagLogDistanceSmoother {
public:
    FixedLagLogDistanceSmoother(const Config& config, int window_size = 30);
    
    /**
     * @brief Add absolute altitude anchor (from initialization or external source)
     * 
     * Converts altitude h to plane distance d: d = h / vertical_factor
     */
    void addAnchor(double timestamp, double altitude, double sigma,
                   const Eigen::Matrix3d& R_CW, 
                   const Eigen::Vector3d* n_cam = nullptr);
    
    /**
     * @brief Add vision constraint from homography
     * 
     * @return True if constraint was accepted
     */
    bool addVisionConstraint(double timestamp,
                            const HomographyConstraint& constraint,
                            const Eigen::Matrix3d& R_CW);
    
    /**
     * @brief Predict state forward (simple model: constant altitude)
     */
    void predict(double timestamp, double dt);
    
    /**
     * @brief Enter HOLD mode when no valid constraints
     */
    void enterHold(double timestamp);
    
    /**
     * @brief Get current altitude estimate
     */
    AltitudeEstimate getEstimate(double timestamp) const;
    
    /**
     * @brief Get debug information
     */
    std::map<std::string, double> getDebugInfo() const;
    
    // Accessors
    double currentH() const { return current_h_; }
    double currentD() const { return current_d_; }
    double verticalFactor() const { return vertical_factor_; }
    AltitudeMode mode() const { return mode_; }
    
private:
    void solve();
    double huberWeight(double residual, double sigma) const;
    
    const Config& config_;
    int window_size_;
    
    std::deque<SmootherState> states_;
    StateId next_state_id_ = 0;
    
    // Factors store state_ids (not indices) so they remain valid after deque shifts
    std::vector<std::tuple<StateId, double, double>> anchor_factors_;  // (state_id, log_d, sigma)
    std::deque<std::tuple<StateId, StateId, double, double>> vision_factors_;  // (id_prev, id_curr, log_s, sigma)
    
    // Current estimate
    double current_log_d_ = std::log(100.0);
    double current_sigma_log_d_ = 1.0;
    double current_d_ = 100.0;
    double current_h_ = 100.0;
    double vertical_factor_ = 1.0;
    
    // Mode
    AltitudeMode mode_ = AltitudeMode::INIT;
    std::optional<double> hold_start_time_;
    double last_good_update_ = 0.0;
    double age_of_last_good_update_ = 0.0;
    
    double huber_k_ = 1.345;
    double d_min_ = 1.0;  // meters
};

} // namespace altitude_estimator

