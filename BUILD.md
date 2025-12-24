# Building Altitude Estimator C++

Complete guide to building the Altitude Estimator on different platforms.

---

## Prerequisites

### All Platforms

- **C++17** compiler
- **CMake 3.15+**
- **OpenCV 4.0+**
- **Eigen3 3.3+**
- **nlohmann_json 3.2+** (optional, for test application)

---

## Ubuntu/Debian

### Install Dependencies

```bash
# Update package lists
sudo apt-get update

# Install build tools
sudo apt-get install -y build-essential cmake git

# Install libraries
sudo apt-get install -y \
    libeigen3-dev \
    libopencv-dev \
    nlohmann-json3-dev
```

### Build

```bash
# Clone repository
git clone https://github.com/yourusername/altitude_estimator_cpp.git
cd altitude_estimator_cpp

# Create build directory
mkdir build && cd build

# Configure (Release mode recommended for performance)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build (use all CPU cores)
cmake --build . -j$(nproc)

# Run tests
./example_simple

# Optional: Install system-wide
sudo cmake --install .
```

---

## Windows

### Option 1: Visual Studio + vcpkg (Recommended)

#### Install vcpkg

```cmd
REM Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg

REM Bootstrap
bootstrap-vcpkg.bat

REM Install dependencies
vcpkg install opencv4:x64-windows eigen3:x64-windows nlohmann-json:x64-windows
```

#### Build with Visual Studio

```cmd
REM Clone repository
git clone https://github.com/yourusername/altitude_estimator_cpp.git
cd altitude_estimator_cpp

REM Create build directory
mkdir build
cd build

REM Configure with vcpkg toolchain
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
         -A x64

REM Build (Release mode)
cmake --build . --config Release

REM Run
Release\example_simple.exe
```

### Option 2: MinGW + MSYS2

```bash
# In MSYS2 terminal
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake \
          mingw-w64-x86_64-opencv mingw-w64-x86_64-eigen3 \
          mingw-w64-x86_64-nlohmann-json

# Build
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

---

## macOS

### Install Dependencies

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake eigen opencv nlohmann-json
```

### Build

```bash
# Clone repository
git clone https://github.com/yourusername/altitude_estimator_cpp.git
cd altitude_estimator_cpp

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(sysctl -n hw.ncpu)

# Run
./example_simple
```

---

## Build Options

### CMake Configuration Options

```bash
# Debug build (with symbols, slower)
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release build (optimized, 10-20x faster)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Specify custom OpenCV path
cmake .. -DOpenCV_DIR=/path/to/opencv/build

# Specify custom Eigen3 path
cmake .. -DEigen3_DIR=/path/to/eigen3/share/eigen3/cmake
```

### Build Targets

```bash
# Build library only
cmake --build . --target altitude_estimator

# Build examples
cmake --build . --target example_simple

# Build tests (requires nlohmann_json)
cmake --build . --target test_drone_captures

# Build everything
cmake --build .
```

---

## Troubleshooting

### "Could not find OpenCV"

**Solution**: Install OpenCV 4.0+ or specify path:

```bash
cmake .. -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
```

### "Could not find Eigen3"

**Solution**: Install Eigen3 or specify path:

```bash
cmake .. -DEigen3_DIR=/usr/local/share/eigen3/cmake
```

### "nlohmann_json not found"

This is optional. If you don't need `test_drone_captures`, you can skip it. Otherwise:

```bash
# Ubuntu/Debian
sudo apt-get install nlohmann-json3-dev

# macOS
brew install nlohmann-json

# Windows (vcpkg)
vcpkg install nlohmann-json
```

### Slow Performance

**Cause**: Debug build

**Solution**: Use Release mode:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Release builds are typically 10-20x faster than Debug.

### Linking Errors

**Windows**: Make sure architecture matches (x64 vs x86)

```cmd
cmake .. -A x64  REM Force 64-bit
```

**Linux**: Check OpenCV was compiled with same compiler

```bash
pkg-config --modversion opencv4
```

---

## Cross-Compilation

### For Raspberry Pi (from Ubuntu host)

```bash
# Install cross-compiler
sudo apt-get install g++-arm-linux-gnueabihf

# Configure for ARM
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/arm-linux.cmake \
         -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)
```

### For NVIDIA Jetson

Build natively on Jetson or use JetPack SDK.

---

## IDE Integration

### Visual Studio Code

1. Install "CMake Tools" extension
2. Open project folder
3. Configure with CMake: Ctrl+Shift+P → "CMake: Configure"
4. Build: Ctrl+Shift+P → "CMake: Build"

### CLion

1. Open project folder
2. CLion automatically detects CMakeLists.txt
3. Select build configuration (Debug/Release)
4. Build: Ctrl+F9

### Visual Studio

1. Open Folder → Select project directory
2. VS automatically configures CMake
3. Select build configuration
4. Build: Ctrl+Shift+B

---

## Docker Build

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libeigen3-dev libopencv-dev nlohmann-json3-dev

COPY . /app
WORKDIR /app/build

RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j$(nproc)

CMD ["./example_simple"]
```

Build and run:

```bash
docker build -t altitude_estimator .
docker run --rm altitude_estimator
```

---

## Installation

```bash
# Install to /usr/local (requires sudo)
sudo cmake --install . --prefix /usr/local

# Install to custom directory
cmake --install . --prefix /opt/altitude_estimator

# Use in other CMake projects
find_package(AltitudeEstimator REQUIRED)
target_link_libraries(myapp AltitudeEstimator::altitude_estimator)
```

---

## Performance Tuning

### Compiler Optimization Flags

```bash
# Maximum optimization (may take longer to compile)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native"

# With OpenMP parallelization (if supported)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -fopenmp"
```

### OpenCV Build Flags

For maximum performance, build OpenCV from source with optimizations:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_TBB=ON \
      -DENABLE_AVX=ON \
      -DENABLE_AVX2=ON \
      ...
```

---

## Next Steps

After building successfully:

1. Run `./example_simple` to verify installation
2. See [README.md](README.md) for API usage
3. Process your own dataset with `./test_drone_captures`

---

**Questions?** Open an issue on GitHub.

