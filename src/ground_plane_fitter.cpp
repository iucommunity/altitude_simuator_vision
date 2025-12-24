/**
 * @file ground_plane_fitter.cpp
 * @brief Stub implementation for GroundPlaneFitter
 */

#include "altitude_estimator/ground_plane_fitter.hpp"

namespace altitude_estimator {

GroundPlaneFitter::GroundPlaneFitter(const Config& config, const FrameConventions& conventions)
    : cfg_(config) {
    
    if (conventions.world_frame == CoordinateFrame::NED) {
        world_up_ = Eigen::Vector3d(0, 0, -1);
    } else {
        world_up_ = Eigen::Vector3d(0, 0, 1);
    }
}

std::optional<GroundModel> GroundPlaneFitter::fitPlane(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<double>* weights
) {
    if (points.size() < size_t(cfg_.min_ground_inliers)) {
        return std::nullopt;
    }
    
    // Simplified RANSAC - just use mean and simple normal
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& pt : points) {
        centroid += pt;
    }
    centroid /= points.size();
    
    GroundModel model;
    model.normal = world_up_;  // Assume flat ground
    model.distance = -model.normal.dot(centroid);
    model.inlier_ratio = 0.8;
    model.residual_m = 0.1;
    model.coverage = 0.5;
    model.stability = 0.8;
    
    ground_model_ = model;
    return model;
}

std::optional<double> GroundPlaneFitter::getAltitude(const Eigen::Vector3d& camera_pos) const {
    if (!ground_model_) {
        return std::nullopt;
    }
    return ground_model_->distanceToCamera(camera_pos);
}

} // namespace altitude_estimator

