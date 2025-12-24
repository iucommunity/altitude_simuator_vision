/**
 * @file common.hpp
 * @brief Common types, enums, and utilities for altitude estimation
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace altitude_estimator {

// ============================================================================
// Enumerations
// ============================================================================

/**
 * @brief Coordinate frame types
 */
enum class CoordinateFrame {
    NED,    ///< North-East-Down (world frame)
    ENU,    ///< East-North-Up (alternative world frame)
    FRD,    ///< Forward-Right-Down (body frame)
    OPENCV  ///< Right-Down-Forward (camera frame)
};

/**
 * @brief Euler angle rotation orders
 */
enum class RotationOrder {
    ZYX,  ///< Yaw-Pitch-Roll (aerospace convention)
    XYZ,  ///< Roll-Pitch-Yaw
    ZXY   ///< Alternative
};

/**
 * @brief Altitude estimation operating mode
 */
enum class AltitudeMode {
    INIT,    ///< Initialization phase
    GEOM,    ///< Geometry-based (primary)
    FUSED,   ///< Fused geometry + depth
    DEPTH,   ///< Depth-only fallback
    HOLD,    ///< Hold last value (failure)
    LOST     ///< System lost
};

// ============================================================================
// String conversions
// ============================================================================

inline std::string toString(AltitudeMode mode) {
    switch (mode) {
        case AltitudeMode::INIT: return "INIT";
        case AltitudeMode::GEOM: return "GEOM";
        case AltitudeMode::FUSED: return "FUSED";
        case AltitudeMode::DEPTH: return "DEPTH";
        case AltitudeMode::HOLD: return "HOLD";
        case AltitudeMode::LOST: return "LOST";
        default: return "UNKNOWN";
    }
}

inline std::string toString(CoordinateFrame frame) {
    switch (frame) {
        case CoordinateFrame::NED: return "NED";
        case CoordinateFrame::ENU: return "ENU";
        case CoordinateFrame::FRD: return "FRD";
        case CoordinateFrame::OPENCV: return "OPENCV";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Type aliases for clarity
// ============================================================================

using FrameIndex = int32_t;
using TrackId = int32_t;
using MapPointId = int32_t;
using StateId = int32_t;

} // namespace altitude_estimator

