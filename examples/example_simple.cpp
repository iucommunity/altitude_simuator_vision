/**
 * @file example_simple.cpp
 * @brief Simple example showing basic altitude estimator usage
 */

#include <altitude_estimator/realtime_altimeter.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace altitude_estimator;

int main() {
    std::cout << "=== Altitude Estimator Simple Example ===" << std::endl;
    std::cout << std::endl;
    
    // Camera parameters (example: 1280x720, 60° vertical FOV)
    double fx = 739.0, fy = 739.0;
    double cx = 640.0, cy = 360.0;
    int width = 1280, height = 720;
    
    // Create altimeter
    std::cout << "Creating altimeter..." << std::endl;
    auto altimeter = createAltimeter(
        fx, fy, cx, cy,
        width, height,
        60.0,  // camera_tilt_deg (60° down from horizontal)
        10,    // init_frames (need known altitude for first 10 frames)
        30.0   // fps
    );
    
    std::cout << "Altimeter created!" << std::endl;
    std::cout << "  Image size: " << width << "x" << height << std::endl;
    std::cout << "  Camera tilt: 60°" << std::endl;
    std::cout << "  Init frames: 10" << std::endl;
    std::cout << std::endl;
    
    // Simulate data (in real application, load from camera + IMU)
    std::cout << "Simulating data..." << std::endl;
    
    // Create synthetic images (checkerboard pattern for feature tracking)
    auto createTestImage = [](int frame_num, int w, int h) {
        cv::Mat img(h, w, CV_8UC3);
        int cell_size = 50;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                bool black = ((x / cell_size) + (y / cell_size)) % 2 == 0;
                cv::Vec3b color = black ? cv::Vec3b(0, 0, 0) : cv::Vec3b(255, 255, 255);
                img.at<cv::Vec3b>(y, x) = color;
            }
        }
        // Add some noise for realism
        cv::Mat noise(h, w, CV_8UC3);
        cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(10));
        cv::add(img, noise, img);
        return img;
    };
    
    // PHASE 1: Initialization (10 frames with known altitude)
    std::cout << "\n--- INITIALIZATION PHASE ---" << std::endl;
    std::cout << "Feeding " << altimeter->initFramesRemaining() << " frames with known altitude..." << std::endl;
    for (int i = 0; i < 10; i++) {
        cv::Mat image = createTestImage(i, width, height);
        
        // Simulate IMU data (radians)
        double roll = 0.0;                    // Level flight
        double pitch = 0.0;                   // Level flight
        double yaw = M_PI / 4.0 + i * 0.01;  // Slowly rotating
        
        // Known altitude from GPS/barometer (simulated descent)
        double known_altitude = 100.0 - i * 2.0;  // Descending from 100m to 82m
        
        try {
            // Use addInitFrame() for initialization
            bool init_complete = altimeter->addInitFrame(image, {roll, pitch, yaw}, known_altitude);
            
            std::cout << "Frame " << i << ": altitude=" << known_altitude 
                      << "m (init: " << (init_complete ? "complete" : "ongoing") << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error on frame " << i << ": " << e.what() << std::endl;
            return 1;
        }
    }
    
    std::cout << "\nInitialization complete!" << std::endl;
    std::cout << "Initialized: " << (altimeter->isInitialized() ? "YES" : "NO") << std::endl;
    
    // PHASE 2: Runtime (estimate altitude without ground truth)
    std::cout << "\n--- RUNTIME PHASE ---" << std::endl;
    for (int i = 10; i < 50; i++) {
        cv::Mat image = createTestImage(i, width, height);
        
        // Simulate IMU data
        double roll = 0.0;
        double pitch = 0.0;
        double yaw = M_PI / 4.0 + i * 0.01;
        
        // NO known altitude - system estimates independently
        auto result = altimeter->process(image, {roll, pitch, yaw});
        
        // Print every 10th frame
        if (i % 10 == 0) {
            std::cout << "Frame " << i << ": ";
            if (result.is_valid) {
                std::cout << "Alt = " << result.altitude_m << " ± " 
                         << result.sigma_m << " m [" << result.mode << "]" << std::endl;
            } else {
                std::cout << "INVALID [" << result.mode << "]" << std::endl;
            }
        }
    }
    
    // Print final status
    std::cout << "\n--- FINAL STATUS ---" << std::endl;
    std::cout << "Total frames processed: " << altimeter->frameCount() << std::endl;
    
    auto status = altimeter->getStatus();
    std::cout << "System status:" << std::endl;
    for (const auto& [key, value] : status) {
        std::cout << "  " << key << ": " << value << std::endl;
    }
    
    std::cout << "\n=== Example Complete ===" << std::endl;
    std::cout << "\nNote: This example uses synthetic images. For real usage," << std::endl;
    std::cout << "replace with actual camera images and IMU data." << std::endl;
    
    return 0;
}

