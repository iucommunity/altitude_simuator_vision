# Quick Start Guide

Get the Altitude Estimator running in 5 minutes.

---

## Prerequisites

- C++17 compiler (GCC/Clang/MSVC)
- CMake 3.15+
- OpenCV 4.0+
- Eigen3 3.3+

---

## 1. Install Dependencies

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libeigen3-dev libopencv-dev
```

### macOS
```bash
brew install cmake eigen opencv
```

### Windows (vcpkg)
```cmd
vcpkg install opencv4:x64-windows eigen3:x64-windows
```

---

## 2. Build

```bash
# Clone
git clone <your-repo-url>
cd altitude_estimator_cpp

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

---

## 3. Run Example

```bash
./example_simple
```

Expected output:
```
=== Altitude Estimator Simple Example ===
Creating altimeter...
Altimeter created!
  Image size: 1280x720
  Camera tilt: 60°

--- INITIALIZATION PHASE ---
Frame 0: Alt=100.00m ± inf m [INIT]
...
Frame 9: Alt=82.00m ± inf m [INIT]

--- RUNTIME PHASE ---
Frame 10: Alt = 80.50 ± 2.3 m [GEOM]
...

=== Example Complete ===
```

---

## 4. Use in Your Code

```cpp
#include <altitude_estimator/realtime_altimeter.hpp>
#include <opencv2/opencv.hpp>

using namespace altitude_estimator;

int main() {
    // Create altimeter
    auto altimeter = createAltimeter(
        739.0, 739.0,      // fx, fy
        640.0, 360.0,      // cx, cy
        1280, 720,         // image size
        60.0,              // camera tilt (degrees)
        10                 // init frames
    );
    
    // PHASE 1: Init (with known altitude)
    for (int i = 0; i < 10; i++) {
        cv::Mat img = cv::imread("frame_" + std::to_string(i) + ".png");
        auto rpy = std::make_tuple(0.0, 0.0, 0.5);  // roll, pitch, yaw (radians)
        double known_alt = 100.0 - i * 2.0;         // meters
        
        auto result = altimeter->process(img, rpy, known_alt);
    }
    
    // PHASE 2: Runtime (no ground truth needed)
    while (true) {
        cv::Mat img = getNextImage();
        auto rpy = getIMU();  // (roll, pitch, yaw) in radians
        
        auto result = altimeter->process(img, rpy);
        
        if (result.is_valid) {
            printf("Altitude: %.2f ± %.2f m [%s]\n",
                   result.altitude_m, result.sigma_m, result.mode.c_str());
        }
    }
    
    return 0;
}
```

Compile:
```bash
g++ -std=c++17 myapp.cpp -o myapp \
    -I/path/to/include \
    -L/path/to/lib -laltitude_estimator \
    `pkg-config --cflags --libs opencv4 eigen3`
```

---

## Troubleshooting

**"Could not find OpenCV"**
```bash
cmake .. -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
```

**Slow performance**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release  # Not Debug!
```

**Linking errors**
- Ensure 64-bit consistency (x64 not x86)
- Check OpenCV version matches (4.x not 3.x)

---

## Next Steps

1. Read [README.md](README.md) for full documentation
2. See [BUILD.md](BUILD.md) for platform-specific build details
3. Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for architecture overview
4. Review [PORTING_NOTES.md](PORTING_NOTES.md) for Python→C++ comparison

---

**Questions?** Open a GitHub issue!

