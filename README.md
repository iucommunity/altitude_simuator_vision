# Altitude Estimator C++

**Monocular Visual Altimeter for UAVs** - C++ port of the Python implementation.

Real-time altitude estimation using a single downward-facing camera and IMU orientation data (Roll-Pitch-Yaw).

---

## Features

- ✅ **Real-time performance**: 30+ FPS on modern hardware
- ✅ **Monocular vision**: Single camera, no stereo or lidar required
- ✅ **Robust estimation**: Huber-loss smoother with quality gates
- ✅ **IMU noise tolerance**: Works with consumer-grade MEMS sensors
- ✅ **Production-ready API**: Simple `process(image, rpy) → altitude`
- ✅ **Metric scale**: Automatic initialization with known altitude anchors

---

## Algorithm Overview

### PRIMARY PATH: Homography-Based

1. **Track features** between consecutive frames (KLT optical flow)
2. **Compute homography** `H` relating the two ground-plane views
3. **Decompose** using known rotation (from IMU):
   ```
   H = K @ (R + (t/d) @ n^T) @ K^-1
   ```
   Extract scale factor `s = d_new / d_old`
4. **Feed into robust smoother** (log-space, Huber loss, fixed-lag)
5. **Output altitude**: `h = d * vertical_factor`

### SECONDARY PATH: Visual Odometry (Fallback)

- Full 3D reconstruction with triangulated map points
- Ground plane fitting from structure
- Used for validation and degraded scenarios

---

## Requirements

### Dependencies

- **C++17** compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **OpenCV 4.0+** (core, imgproc, features2d, video, calib3d)
- **Eigen3 3.3+** (linear algebra)
- **nlohmann_json 3.2+** (optional, for test application)

### Installation (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake \
    libeigen3-dev \
    libopencv-dev \
    nlohmann-json3-dev
```

### Installation (Windows with vcpkg)

```cmd
vcpkg install opencv4 eigen3 nlohmann-json
```

### Installation (macOS with Homebrew)

```bash
brew install cmake eigen opencv nlohmann-json
```

---

## Building

```bash
# Clone repository
git clone https://github.com/yourusername/altitude_estimator_cpp.git
cd altitude_estimator_cpp

# Create build directory
mkdir build && cd build

# Configure (Release mode for performance)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)

# Optionally install
sudo cmake --install .
```

### Build with vcpkg (Windows)

```cmd
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

---

## Quick Start

### Example 1: Minimal Usage

```cpp
#include <altitude_estimator/realtime_altimeter.hpp>
#include <opencv2/opencv.hpp>

using namespace altitude_estimator;

int main() {
    // Create altimeter
    auto altimeter = createAltimeter(
        739.0, 739.0,      // fx, fy (focal lengths)
        640.0, 360.0,      // cx, cy (principal point)
        1280, 720,         // image size
        60.0,              // camera tilt (60° down)
        10                 // init frames with known altitude
    );
    
    // Initialization phase (first 10 frames)
    for (int i = 0; i < 10; i++) {
        cv::Mat image = cv::imread("frame_" + std::to_string(i) + ".png");
        double roll = 0.0, pitch = 0.0, yaw = M_PI/4;  // radians
        double known_altitude = 50.0;  // meters (from GPS/barometer)
        
        auto result = altimeter->process(image, {roll, pitch, yaw}, known_altitude);
        std::cout << "Init " << i << ": " << result.toString() << std::endl;
    }
    
    // Runtime phase (no known altitude needed)
    while (true) {
        cv::Mat image = getNextImage();
        auto rpy = getIMU();  // (roll, pitch, yaw) in radians
        
        auto result = altimeter->process(image, rpy);
        
        if (result.is_valid) {
            printf("Altitude: %.2f ± %.2f m [%s]\n",
                   result.altitude_m, result.sigma_m, result.mode.c_str());
        }
    }
    
    return 0;
}
```

### Example 2: With Dataset

See `examples/example_simple.cpp` and `tests/test_drone_captures.cpp` for complete examples.

---

## API Reference

### `RealtimeAltimeter`

```cpp
class RealtimeAltimeter {
public:
    RealtimeAltimeter(
        const Eigen::Matrix3d& K,       // Camera intrinsics (3x3)
        const cv::Size& image_size,     // (width, height)
        double camera_tilt_deg = 60.0,  // Tilt below horizontal
        int init_frames = 10,           // Frames with known altitude
        double fps = 30.0               // Expected frame rate
    );
    
    // Process single frame
    AltimeterResult process(
        const cv::Mat& image,                    // BGR or grayscale
        const std::tuple<double, double, double>& rpy,  // (roll, pitch, yaw) radians
        const std::optional<double>& known_altitude = std::nullopt
    );
    
    bool isInitialized() const;
    int frameCount() const;
    std::map<std::string, double> getStatus() const;
    void reset();
};
```

### `AltimeterResult`

```cpp
struct AltimeterResult {
    double altitude_m;  // Estimated altitude (meters)
    double sigma_m;     // Uncertainty (1-sigma, meters)
    bool is_valid;      // True if estimate is reliable
    std::string mode;   // "INIT", "GEOM", "HOLD", "LOST"
};
```

---

## Configuration

### Camera Calibration

**Intrinsics**: Calibrate with checkerboard at actual resolution and focus.

```cpp
// Option 1: From calibrated parameters
Eigen::Matrix3d K;
K << fx,  0, cx,
      0, fy, cy,
      0,  0,  1;

// Option 2: Use default (Unity Physical Camera model)
auto calib = CalibrationData::createDefault(1280, 720, 60.0);
```

**Extrinsics**: Camera tilt relative to body frame.

```cpp
// Camera tilted 60° down from horizontal
auto extrinsics = CameraExtrinsics::fromTiltAngle(60.0);
```

### System Parameters

Key configuration options in `Config` struct:

```cpp
Config cfg;

// Feature tracking
cfg.max_features = 500;
cfg.min_inliers = 20;

// Homography gates
cfg.enable_rank1_gate = false;  // Requires accurate RPY calibration
cfg.homography_s_min = 0.3;     // Min scale (fast descent)
cfg.homography_s_max = 3.0;     // Max scale (fast ascent)

// Smoother
cfg.huber_k = 1.345;  // Huber threshold for outlier rejection
```

---

## Testing

### Run Test Application

```bash
# Process DroneCaptures dataset (from Python project)
./test_drone_captures --folder ../DroneCaptures \
                      --init-frames 10 \
                      --imu-noise consumer

# Options:
#   --folder PATH         Dataset folder with images/ and metadata.json
#   --init-frames N       Number of init frames (default: 10)
#   --max-frames N        Limit processing (default: all)
#   --imu-noise MODE      off|consumer|poor (default: off)
#   --imu-seed N          Random seed for reproducibility
#   --save-plot PATH      Save results plot
```

### Expected Performance

On DroneCaptures dataset (1280×720, flat terrain):

- **Accuracy**: 1-5m RMSE
- **Speed**: 30-60 FPS (Intel i7, Release build)
- **Mode**: GEOM (geometry-based) after initialization

---

## Coordinate Frame Conventions

| Frame | Axes | Convention |
|-------|------|------------|
| **World (W)** | x=North, y=East, z=Down | NED |
| **Body (B)** | x=Forward, y=Right, z=Down | FRD |
| **Camera (C)** | x=Right, y=Down, z=Forward | OpenCV |

**Rotation naming**: `R_CW` = World→Camera, `C_W` = camera position in world

**RPY input**:
- Roll: rotation about forward axis (positive = right wing down)
- Pitch: rotation about right axis (positive = nose up)
- Yaw: rotation about down axis (positive = clockwise from above)

**All angles in RADIANS** for `process()` API.

---

## Limitations

- **Flat terrain assumption**: Ground must be approximately planar
- **Minimum altitude**: < 5m may be unreliable
- **Texture required**: Ground must have trackable features
- **IMU required**: Needs roll/pitch/yaw at frame rate
- **Initialization**: Requires 2+ frames with known altitude

---

## Troubleshooting

### No altitude estimate (mode = HOLD/LOST)

**Causes**:
1. Insufficient features tracked (<20 inliers)
2. Homography quality gates failing
3. Non-planar or occluded ground
4. Extreme altitude changes (>3x between frames)

**Solutions**:
- Ensure good lighting and texture on ground
- Check IMU data is valid and time-synchronized
- Increase `init_frames` if initialization fails
- Verify camera calibration (intrinsics + tilt)

### Large errors

**Causes**:
1. Incorrect camera calibration (fx, fy, tilt)
2. IMU sign convention mismatch
3. Poor initialization (bad altitude anchors)

**Solutions**:
- Recalibrate camera intrinsics
- Check RPY sign conventions match your autopilot
- Use more init frames with accurate altitude

### Low FPS

**Causes**:
1. Debug build (use `-DCMAKE_BUILD_TYPE=Release`)
2. Too many features (`max_features`)
3. Large image size

**Solutions**:
- Rebuild in Release mode for 10-20x speedup
- Reduce `max_features` to 300-400
- Downscale images to 640×480 if acceptable

---

## Project Structure

```
altitude_estimator_cpp/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── include/                 # Public headers
│   └── altitude_estimator/
│       ├── realtime_altimeter.hpp      # Production API
│       ├── calibration.hpp
│       ├── config.hpp
│       ├── data_types.hpp
│       ├── homography_altimeter.hpp
│       ├── smoother.hpp
│       └── ...
├── src/                     # Implementation
│   ├── realtime_altimeter.cpp
│   ├── homography_altimeter.cpp
│   ├── smoother.cpp
│   └── ...
├── examples/                # Usage examples
│   └── example_simple.cpp
└── tests/                   # Test applications
    ├── test_drone_captures.cpp
    └── imu_noise_simulator.cpp
```

---

## License

MIT License - see LICENSE file for details.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{altitude_estimator,
  title = {Monocular Visual Altimeter for UAVs},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/altitude_estimator_cpp}
}
```

---

## Acknowledgments

- OpenCV library for computer vision primitives
- Eigen library for efficient linear algebra
- Original Python implementation and design

---

## Contact

- **Issues**: https://github.com/yourusername/altitude_estimator_cpp/issues
- **Email**: your.email@example.com

---

**Happy Flying! 🚁**

