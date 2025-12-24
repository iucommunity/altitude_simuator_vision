/**
 * @file ground_plane_fitter.hpp
 * @brief Fit ground plane from triangulated 3D points
 */

#pragma once

#include "common.hpp"
#include "config.hpp"
#include "data_types.hpp"
#include "coordinate_frames.hpp"
#include <optional>

namespace altitude_estimator {

class GroundPlaneFitter {
public:
    GroundPlaneFitter(const Config& config, const FrameConventions& conventions);
    
    std::optional<GroundModel> fitPlane(
        const std::vector<Eigen::Vector3d>& points,
        const std::vector<double>* weights = nullptr
    );
    
    std::optional<double> getAltitude(const Eigen::Vector3d& camera_pos) const;
    
private:
    const Config& cfg_;
    Eigen::Vector3d world_up_;
    std::optional<GroundModel> ground_model_;
};

} // namespace altitude_estimator

