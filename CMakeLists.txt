cmake_minimum_required(VERSION 3.15)
project(AltitudeEstimator VERSION 1.0.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
    add_compile_definitions(_USE_MATH_DEFINES)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc features2d video calib3d)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Optional: JSON library for metadata reading
find_package(nlohmann_json 3.2.0 QUIET)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Core library
add_library(altitude_estimator STATIC
    src/coordinate_frames.cpp
    src/calibration.cpp
    src/rotation_provider.cpp
    src/visual_tracker.cpp
    src/homography_altimeter.cpp
    src/smoother.cpp
    src/pose_engine.cpp
    src/ground_segmenter.cpp
    src/ground_plane_fitter.cpp
    src/initializer.cpp
    src/altitude_estimation_system.cpp
    src/realtime_altimeter.cpp
)

target_include_directories(altitude_estimator PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(altitude_estimator PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Test executable
if(nlohmann_json_FOUND)
    add_executable(test_drone_captures
        tests/test_drone_captures.cpp
        tests/imu_noise_simulator.cpp
    )
    
    target_link_libraries(test_drone_captures
        altitude_estimator
        nlohmann_json::nlohmann_json
    )
    
    message(STATUS "Building test_drone_captures (nlohmann_json found)")
else()
    message(WARNING "nlohmann_json not found - test_drone_captures will not be built")
    message(STATUS "Install with: sudo apt-get install nlohmann-json3-dev (Linux)")
    message(STATUS "            : vcpkg install nlohmann-json (Windows/vcpkg)")
endif()

# Example executable (minimal dependencies)
add_executable(example_simple
    examples/example_simple.cpp
)

target_link_libraries(example_simple
    altitude_estimator
)

# Installation
install(TARGETS altitude_estimator
    EXPORT AltitudeEstimatorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# Print configuration
message(STATUS "")
message(STATUS "=== Altitude Estimator Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "========================================")
message(STATUS "")

