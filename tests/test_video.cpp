/**
 * @file test_video.cpp
 * @brief Video inference for altitude estimation
 * 
 * Processes a video file (e.g., output_video.mp4 created from DroneCaptures/images)
 * and outputs altitude estimates to console.
 * 
 * Usage:
 *   ./test_video --video output_video.mp4 --metadata DroneCaptures/metadata.json
 *   ./test_video --video output_video.mp4 --folder DroneCaptures
 */

#include <altitude_estimator/realtime_altimeter.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace altitude_estimator;

// ============================================================================
// Data Structures
// ============================================================================

struct MetadataEntry {
    std::string filename;
    double height;
    double pitch;
    double yaw;
    double roll;
};

struct FrameResult {
    int frame_idx;
    double gt_altitude;
    double est_altitude;
    double error;
    std::string mode;
    double sigma;
    double time_ms;
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Load metadata.json with random noise on yaw and roll (±3 degrees)
 */
std::vector<MetadataEntry> loadMetadata(const fs::path& metadata_path) {
    if (!fs::exists(metadata_path)) {
        throw std::runtime_error("metadata.json not found: " + metadata_path.string());
    }
    
    std::ifstream file(metadata_path);
    json j;
    file >> j;
    
    // Random noise generator: ±3 degrees
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> noise_dist(-3.0, 3.0);
    
    std::vector<MetadataEntry> entries;
    for (const auto& item : j) {
        MetadataEntry entry;
        entry.filename = item["filename"];
        entry.height = item["height"];
        entry.pitch = item["pitch"];
        
        // Base yaw and roll from metadata
        double base_yaw = item["yaw"];
        double base_roll = item.value("roll", 0.0);  // Default roll to 0 if not present
        
        // Add random noise ±3 degrees
        entry.yaw = base_yaw + noise_dist(gen);
        entry.roll = base_roll + noise_dist(gen);
        
        entries.push_back(entry);
    }
    
    // Sort by filename to ensure correct frame order
    std::sort(entries.begin(), entries.end(), 
              [](const MetadataEntry& a, const MetadataEntry& b) {
                  return a.filename < b.filename;
              });
    
    return entries;
}

/**
 * Compute camera intrinsics from Unity Physical Camera model
 */
void computeIntrinsics(int width, int height, 
                       double& fx, double& fy, double& cx, double& cy) {
    const double sensor_width_mm = 36.0;
    const double focal_length_mm = 20.78461;
    
    double sensor_height_eff = sensor_width_mm * height / width;
    fx = focal_length_mm * width / sensor_width_mm;
    fy = focal_length_mm * height / sensor_height_eff;
    cx = width / 2.0;
    cy = height / 2.0;
}

/**
 * Print statistics summary
 */
void printStatistics(const std::vector<FrameResult>& results) {
    std::vector<double> errors;
    std::vector<double> pct_errors;
    
    for (const auto& r : results) {
        if (!std::isnan(r.error) && r.gt_altitude > 0) {
            errors.push_back(r.error);
            pct_errors.push_back(100.0 * r.error / r.gt_altitude);
        }
    }
    
    if (errors.empty()) {
        std::cout << "No valid estimates produced\n";
        return;
    }
    
    double sum = 0, sum_sq = 0, sum_abs = 0;
    for (double e : errors) {
        sum += e;
        sum_sq += e * e;
        sum_abs += std::abs(e);
    }
    
    double mean_error = sum / errors.size();
    double std_error = std::sqrt(sum_sq / errors.size() - mean_error * mean_error);
    double mae = sum_abs / errors.size();
    double rmse = std::sqrt(sum_sq / errors.size());
    
    double mean_pct = 0;
    for (double p : pct_errors) mean_pct += p;
    mean_pct /= pct_errors.size();
    
    // Mode breakdown
    std::map<std::string, int> mode_counts;
    for (const auto& r : results) {
        mode_counts[r.mode]++;
    }
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "SUMMARY\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Frames processed: " << results.size() << "\n";
    std::cout << "Valid estimates:  " << errors.size() << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Mean Error:       " << std::showpos << mean_error << " m\n";
    std::cout << "Std Error:        " << std::noshowpos << std_error << " m\n";
    std::cout << "MAE:              " << mae << " m\n";
    std::cout << "RMSE:             " << rmse << " m\n";
    std::cout << "Mean % Error:     " << std::showpos << mean_pct << "%\n";
    std::cout << std::noshowpos;
    
    std::cout << "\nMode breakdown:\n";
    for (const auto& [mode, count] : mode_counts) {
        double pct = 100.0 * count / results.size();
        std::cout << "  " << mode << ": " << count << " frames (" 
                  << std::fixed << std::setprecision(1) << pct << "%)\n";
    }
}

/**
 * Print usage
 */
void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --video <path>       Path to video file (required)\n"
              << "  --metadata <path>    Path to metadata.json\n"
              << "  --folder <path>      Folder containing metadata.json (alternative)\n"
              << "  --init-frames <n>    Number of init frames with known altitude (default: 10)\n"
              << "  --max-frames <n>     Maximum frames to process (default: all)\n"
              << "  --start-frame <n>    Start from frame N (default: 0)\n"
              << "  --verbose            Print every frame (default: first 30 + every 50th)\n"
              << "  --help               Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " --video output_video.mp4 --folder DroneCaptures\n"
              << "  " << prog << " --video output_video.mp4 --metadata DroneCaptures/metadata.json\n";
}

// ============================================================================
// Main Video Inference
// ============================================================================

int runVideoInference(
    const fs::path& video_path,
    const std::vector<MetadataEntry>& metadata,
    int init_frames,
    int max_frames,
    int start_frame,
    bool verbose
) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "VIDEO ALTITUDE ESTIMATION\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    // Open video
    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open video: " << video_path << "\n";
        return 1;
    }
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Video: " << video_path.filename().string() << "\n";
    std::cout << "Resolution: " << width << "x" << height << "\n";
    std::cout << "Total frames: " << total_frames << "\n";
    std::cout << "FPS: " << fps << "\n";
    std::cout << "Metadata entries: " << metadata.size() << "\n";
    
    if (metadata.size() < static_cast<size_t>(total_frames)) {
        std::cout << "WARNING: Fewer metadata entries than video frames!\n";
    }
    
    // Determine frames to process
    int n_frames = total_frames - start_frame;
    if (max_frames > 0) {
        n_frames = std::min(n_frames, max_frames);
    }
    n_frames = std::min(n_frames, static_cast<int>(metadata.size()) - start_frame);
    
    if (n_frames <= 0) {
        std::cerr << "ERROR: No frames to process!\n";
        return 1;
    }
    
    std::cout << "Processing frames: " << start_frame << " to " << (start_frame + n_frames - 1) << "\n";
    std::cout << "Init frames: " << init_frames << " (known altitude)\n";
    
    // Get camera tilt from first metadata entry
    double camera_tilt = 60.0;  // Default
    if (!metadata.empty()) {
        camera_tilt = metadata[start_frame].pitch;
    }
    std::cout << "Camera tilt: " << camera_tilt << "°\n";
    
    // Compute intrinsics
    double fx, fy, cx, cy;
    computeIntrinsics(width, height, fx, fy, cx, cy);
    std::cout << "Camera Intrinsics: fx=" << std::fixed << std::setprecision(3) << fx 
              << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << "\n";
    
    // Create altimeter
    auto altimeter = createAltimeter(fx, fy, cx, cy, width, height,
                                     camera_tilt, init_frames, fps);
    
    // Seek to start frame if needed
    if (start_frame > 0) {
        cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
    }
    
    // Track results
    std::vector<FrameResult> results;
    std::vector<double> processing_times;
    
    // Print header
    std::cout << "\n" << std::left 
              << std::setw(8) << "Frame"
              << std::setw(10) << "GT Alt"
              << std::setw(10) << "Est Alt"
              << std::setw(10) << "Error"
              << std::setw(8) << "Mode"
              << std::setw(8) << "σ"
              << std::setw(8) << "ms"
              << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < n_frames; ++i) {
        int frame_idx = start_frame + i;
        
        // Read frame from video
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "WARNING: Could not read frame " << frame_idx << "\n";
            break;
        }
        
        // Get metadata for this frame
        const MetadataEntry& meta = metadata[frame_idx];
        double gt_altitude = meta.height;
        
        // RPY in radians (level flight assumption, use metadata yaw)
        double roll_rad = 0.0;
        double pitch_rad = 0.0;
        double yaw_rad = meta.yaw * M_PI / 180.0;
        
        // Process frame
        bool is_init = i < init_frames;
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        AltimeterResult result;
        try {
            if (is_init) {
                // INIT PHASE: feed frames with known altitude
                altimeter->addInitFrame(frame, {roll_rad, pitch_rad, yaw_rad}, gt_altitude);
                result.altitude_m = gt_altitude;
                result.sigma_m = std::numeric_limits<double>::infinity();
                result.is_valid = true;
                result.mode = "INIT";
            } else {
                // RUNTIME PHASE: just image + RPY
                result = altimeter->process(frame, {roll_rad, pitch_rad, yaw_rad});
            }
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV ERROR on frame " << frame_idx << ": " << e.what() << "\n";
            continue;
        } catch (const std::exception& e) {
            std::cerr << "ERROR on frame " << frame_idx << ": " << e.what() << "\n";
            continue;
        }
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        double frame_time_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        processing_times.push_back(frame_time_ms);
        
        // Compute error
        double error = result.is_valid && gt_altitude > 0 ? 
                      result.altitude_m - gt_altitude : NAN;
        
        FrameResult fr;
        fr.frame_idx = frame_idx;
        fr.gt_altitude = gt_altitude;
        fr.est_altitude = result.altitude_m;
        fr.error = error;
        fr.mode = result.mode;
        fr.sigma = result.sigma_m;
        fr.time_ms = frame_time_ms;
        results.push_back(fr);
        
        // Print progress
        bool should_print = verbose || i < 30 || i % 50 == 0 || i == n_frames - 1;
        if (should_print) {
            std::cout << std::left << std::setw(8) << frame_idx
                      << std::setw(10) << std::fixed << std::setprecision(1) << gt_altitude
                      << std::setw(10) << result.altitude_m
                      << std::setw(10) << std::showpos << error << std::noshowpos
                      << std::setw(8) << result.mode
                      << std::setw(8) << std::setprecision(1) << result.sigma_m
                      << std::setw(8) << std::setprecision(1) << frame_time_ms
                      << "\n";
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    // Print statistics
    printStatistics(results);
    
    // Performance statistics
    if (!processing_times.empty()) {
        double sum = 0, sum_sq = 0;
        double min_t = processing_times[0], max_t = processing_times[0];
        for (double t : processing_times) {
            sum += t;
            sum_sq += t * t;
            min_t = std::min(min_t, t);
            max_t = std::max(max_t, t);
        }
        double avg = sum / processing_times.size();
        double std_t = std::sqrt(sum_sq / processing_times.size() - avg * avg);
        
        std::cout << "\nPerformance:\n";
        std::cout << "  Total time:     " << std::fixed << std::setprecision(2) << total_time << " s\n";
        std::cout << "  Avg per frame:  " << avg << " ms\n";
        std::cout << "  Min per frame:  " << min_t << " ms\n";
        std::cout << "  Max per frame:  " << max_t << " ms\n";
        std::cout << "  Std per frame:  " << std_t << " ms\n";
        std::cout << "  Throughput:     " << std::setprecision(1) << (results.size() / total_time) << " FPS\n";
    }
    
    cap.release();
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Parse arguments
    fs::path video_path;
    fs::path metadata_path;
    fs::path folder_path;
    int init_frames = 10;
    int max_frames = -1;
    int start_frame = 0;
    bool verbose = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            metadata_path = argv[++i];
        } else if (arg == "--folder" && i + 1 < argc) {
            folder_path = argv[++i];
        } else if (arg == "--init-frames" && i + 1 < argc) {
            init_frames = std::stoi(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--start-frame" && i + 1 < argc) {
            start_frame = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Validate arguments
    if (video_path.empty()) {
        std::cerr << "ERROR: --video is required\n\n";
        printUsage(argv[0]);
        return 1;
    }
    
    if (!fs::exists(video_path)) {
        std::cerr << "ERROR: Video file not found: " << video_path << "\n";
        return 1;
    }
    
    // Resolve metadata path
    if (metadata_path.empty()) {
        if (!folder_path.empty()) {
            metadata_path = folder_path / "metadata.json";
        } else {
            // Try to find metadata.json in same folder as video
            metadata_path = video_path.parent_path() / "metadata.json";
            if (!fs::exists(metadata_path)) {
                std::cerr << "ERROR: --metadata or --folder is required\n\n";
                printUsage(argv[0]);
                return 1;
            }
        }
    }
    
    // Load metadata
    std::vector<MetadataEntry> metadata;
    try {
        std::cout << "Loading metadata from: " << metadata_path << "\n";
        metadata = loadMetadata(metadata_path);
        std::cout << "Loaded " << metadata.size() << " entries\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR loading metadata: " << e.what() << "\n";
        return 1;
    }
    
    // Run inference
    return runVideoInference(video_path, metadata, init_frames, max_frames, start_frame, verbose);
}

