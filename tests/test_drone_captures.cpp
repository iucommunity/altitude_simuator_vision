/**
 * @file test_drone_captures.cpp
 * @brief Test application for altitude estimation on DroneCaptures dataset
 * 
 * Equivalent to Python's test_drone_captures.py
 */

#include <altitude_estimator/realtime_altimeter.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <map>
#include <cmath>

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

struct ProcessingResult {
    int frame;
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
 * Load metadata.json from folder
 */
std::vector<MetadataEntry> loadMetadata(const fs::path& folder) {
    fs::path meta_path = folder / "metadata.json";
    
    if (!fs::exists(meta_path)) {
        throw std::runtime_error("metadata.json not found in " + folder.string());
    }
    
    std::ifstream file(meta_path);
    json j;
    file >> j;
    
    std::vector<MetadataEntry> entries;
    for (const auto& item : j) {
        MetadataEntry entry;
        entry.filename = item["filename"];
        entry.height = item["height"];
        entry.pitch = item["pitch"];
        entry.yaw = item["yaw"];
        entry.roll = item.value("roll", item["yaw"]);  // Fallback if roll missing
        entries.push_back(entry);
    }
    
    return entries;
}

/**
 * Find all images in folder/images/
 */
std::vector<fs::path> findImages(const fs::path& folder) {
    fs::path img_folder = folder / "images";
    
    if (!fs::exists(img_folder)) {
        throw std::runtime_error("images/ folder not found in " + folder.string());
    }
    
    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(img_folder)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".PNG" || ext == ".JPG") {
                images.push_back(entry.path());
            }
        }
    }
    
    std::sort(images.begin(), images.end());
    return images;
}

/**
 * Match images to metadata by filename
 */
std::vector<std::pair<fs::path, MetadataEntry>> matchImagesToMetadata(
    const std::vector<fs::path>& images,
    const std::vector<MetadataEntry>& metadata
) {
    // Build lookup
    std::map<std::string, MetadataEntry> meta_by_filename;
    for (const auto& entry : metadata) {
        meta_by_filename[entry.filename] = entry;
    }
    
    // Match
    std::vector<std::pair<fs::path, MetadataEntry>> matched;
    for (const auto& img_path : images) {
        std::string filename = img_path.filename().string();
        if (meta_by_filename.count(filename)) {
            matched.push_back({img_path, meta_by_filename[filename]});
        }
    }
    
    return matched;
}

/**
 * Compute camera intrinsics from Unity Physical Camera model
 */
void computeIntrinsics(int width, int height, 
                       double& fx, double& fy, double& cx, double& cy) {
    // Unity Physical Camera with Gate Fit = Horizontal
    const double sensor_width_mm = 36.0;
    const double focal_length_mm = 20.78461;
    
    double sensor_height_eff = sensor_width_mm * height / width;
    fx = focal_length_mm * width / sensor_width_mm;
    fy = focal_length_mm * height / sensor_height_eff;
    cx = width / 2.0;
    cy = height / 2.0;
}

/**
 * Print statistics
 */
void printStatistics(const std::vector<ProcessingResult>& results) {
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
    
    // Compute statistics
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
    for (double p : pct_errors) {
        mean_pct += p;
    }
    mean_pct /= pct_errors.size();
    
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
}

// ============================================================================
// Main Test Function
// ============================================================================

int runTest(const fs::path& folder, int init_frames = 10, int max_frames = -1) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "ALTITUDE ESTIMATION TEST - " << folder.filename().string() << "\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    // Load data
    std::cout << "Loading metadata...\n";
    auto metadata = loadMetadata(folder);
    std::cout << "Found " << metadata.size() << " metadata entries\n";
    
    std::cout << "Finding images...\n";
    auto images = findImages(folder);
    std::cout << "Found " << images.size() << " images\n";
    
    if (images.empty()) {
        std::cerr << "ERROR: No images found!\n";
        return 1;
    }
    
    // Match
    auto matched_data = matchImagesToMetadata(images, metadata);
    std::cout << "Matched " << matched_data.size() << " image-metadata pairs\n";
    
    if (matched_data.empty()) {
        std::cerr << "ERROR: No images matched to metadata!\n";
        return 1;
    }
    
    int n_frames = matched_data.size();
    if (max_frames > 0) {
        n_frames = std::min(n_frames, max_frames);
    }
    std::cout << "Processing " << n_frames << " frames\n\n";
    
    // Load first image to get dimensions
    cv::Mat sample_img = cv::imread(matched_data[0].first.string());
    if (sample_img.empty()) {
        std::cerr << "ERROR: Could not load " << matched_data[0].first << "\n";
        return 1;
    }
    
    int width = sample_img.cols;
    int height = sample_img.rows;
    std::cout << "Image size: " << width << "x" << height << "\n";
    
    // Get camera tilt from metadata
    double camera_tilt = matched_data[0].second.pitch;
    std::cout << "Camera tilt: " << camera_tilt << "°\n";
    
    // Compute intrinsics
    double fx, fy, cx, cy;
    computeIntrinsics(width, height, fx, fy, cx, cy);
    std::cout << "Camera Intrinsics: fx=" << fx << ", fy=" << fy 
              << ", cx=" << cx << ", cy=" << cy << "\n";
    
    std::cout << "Init frames: " << init_frames << " (known altitude)\n";
    
    // Create altimeter
    double fps = 30.0;
    auto altimeter = createAltimeter(fx, fy, cx, cy, width, height,
                                     camera_tilt, init_frames, fps);
    
    // Track results
    std::vector<ProcessingResult> results;
    std::vector<double> processing_times;
    
    std::cout << "\n" << std::left << std::setw(8) << "Frame"
              << std::setw(10) << "GT Alt"
              << std::setw(10) << "Est Alt"
              << std::setw(10) << "Error"
              << std::setw(8) << "Mode"
              << std::setw(8) << "σ"
              << std::setw(8) << "ms\n";
    std::cout << std::string(70, '-') << "\n";
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < n_frames; ++i) {
        const auto& [img_path, meta] = matched_data[i];
        std::cout << "Processing frame " << i << " (" << img_path.filename().string() << ")..." << std::flush;
        
        // Load image
        cv::Mat image = cv::imread(img_path.string());
        if (image.empty()) {
            std::cerr << "\nWARNING: Could not load " << img_path << "\n";
            continue;
        }
        
        // Get ground truth
        double gt_altitude = meta.height;
        
        // DATASET-SPECIFIC: Level flight assumption
        double roll_deg_gt = 0.0;
        double pitch_deg_gt = 0.0;
        double yaw_deg_gt = meta.yaw;
        
        // Convert to radians
        double roll_rad = roll_deg_gt * M_PI / 180.0;
        double pitch_rad = pitch_deg_gt * M_PI / 180.0;
        double yaw_rad = yaw_deg_gt * M_PI / 180.0;
        
        // Process frame
        bool is_init = i < init_frames;
        std::optional<double> known_alt = is_init ? std::make_optional(gt_altitude) : std::nullopt;
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        AltimeterResult result;
        try {
            result = altimeter->process(image, {roll_rad, pitch_rad, yaw_rad}, known_alt);
            std::cout << " done" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV ERROR on frame " << i << ": " << e.what() << "\n";
            std::cerr << "  Error code: " << e.code << "\n";
            std::cerr << "  File: " << e.file << ":" << e.line << "\n";
            return 1;  // Exit to see the error
        } catch (const std::exception& e) {
            std::cerr << "ERROR on frame " << i << ": " << e.what() << "\n";
            return 1;  // Exit to see the error
        } catch (...) {
            std::cerr << "UNKNOWN ERROR on frame " << i << "\n";
            return 1;  // Exit to see the error
        }
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        double frame_time_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        processing_times.push_back(frame_time_ms);
        
        // Compute error
        double error = result.is_valid && gt_altitude > 0 ? 
                      result.altitude_m - gt_altitude : NAN;
        
        ProcessingResult proc_result;
        proc_result.frame = i;
        proc_result.gt_altitude = gt_altitude;
        proc_result.est_altitude = result.altitude_m;
        proc_result.error = error;
        proc_result.mode = result.mode;
        proc_result.sigma = result.sigma_m;
        proc_result.time_ms = frame_time_ms;
        results.push_back(proc_result);
        
        // Debug: print homography scale info to diagnose "stuck at anchor"
        // (Uses RealtimeAltimeter::getStatus() → AltitudeEstimationSystem::getStatus())
        auto status = altimeter->getStatus();
        double s = status.count("homography_s") ? status["homography_s"] : 1.0;
        double log_s = status.count("homography_log_s") ? status["homography_log_s"] : 0.0;
        double dot_nu = status.count("homography_metric_dot_nu") ? status["homography_metric_dot_nu"] : 0.0;
        double rmse_px = status.count("homography_metric_rmse_px") ? status["homography_metric_rmse_px"] : 0.0;
        double nin = status.count("homography_metric_n_inliers") ? status["homography_metric_n_inliers"] : 0.0;
        double gate = status.count("homography_metric_gate_failed") ? status["homography_metric_gate_failed"] : 0.0;
        double trk = status.count("track_total") ? status["track_total"] : 0.0;
        double trk_inl = status.count("track_inliers") ? status["track_inliers"] : 0.0;
        double h_att = status.count("homography_attempted") ? status["homography_attempted"] : 0.0;
        double h_ok = status.count("homography_succeeded") ? status["homography_succeeded"] : 0.0;

        // Print progress
        if (i < 30 || i % 50 == 0 || i == n_frames - 1) {
            std::cout << std::left << std::setw(8) << i
                      << std::setw(10) << std::fixed << std::setprecision(1) << gt_altitude
                      << std::setw(10) << result.altitude_m
                      << std::setw(10) << std::showpos << error << std::noshowpos
                      << std::setw(8) << result.mode
                      << std::setw(8) << result.sigma_m
                      << std::setw(8) << frame_time_ms
                      << "  s=" << std::fixed << std::setprecision(4) << s
                      << " log_s=" << std::fixed << std::setprecision(6) << log_s
                      << " dot=" << std::fixed << std::setprecision(5) << dot_nu
                      << " rmse=" << std::fixed << std::setprecision(3) << rmse_px
                      << " inl=" << int(nin)
                      << " gate=" << std::fixed << std::setprecision(1) << gate
                      << " trk=" << int(trk)
                      << " trk_inl=" << int(trk_inl)
                      << " h_att=" << int(h_att)
                      << " h_ok=" << int(h_ok)
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
        double min_time = processing_times[0];
        double max_time = processing_times[0];
        
        for (double t : processing_times) {
            sum += t;
            sum_sq += t * t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        
        double avg_time = sum / processing_times.size();
        double std_time = std::sqrt(sum_sq / processing_times.size() - avg_time * avg_time);
        double fps_achieved = 1000.0 / avg_time;
        
        std::cout << "\nPerformance:\n";
        std::cout << "  Total time:     " << std::fixed << std::setprecision(2) << total_time << " s\n";
        std::cout << "  Avg per frame:  " << avg_time << " ms\n";
        std::cout << "  Min per frame:  " << min_time << " ms\n";
        std::cout << "  Max per frame:  " << max_time << " ms\n";
        std::cout << "  Std per frame:  " << std_time << " ms\n";
        std::cout << "  Throughput:     " << std::setprecision(1) << fps_achieved << " FPS\n";
    }
    
    // Mode breakdown
    std::map<std::string, int> mode_counts;
    for (const auto& r : results) {
        mode_counts[r.mode]++;
    }
    
    std::cout << "\nMode breakdown:\n";
    for (const auto& [mode, count] : mode_counts) {
        double pct = 100.0 * count / results.size();
        std::cout << "  " << mode << ": " << count << " frames (" 
                  << std::fixed << std::setprecision(1) << pct << "%)\n";
    }
    
    return 0;
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    // Simple argument parsing
    std::string folder = "DroneCaptures";
    int init_frames = 10;
    int max_frames = -1;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--folder" && i + 1 < argc) {
            folder = argv[++i];
        } else if (arg == "--init-frames" && i + 1 < argc) {
            init_frames = std::stoi(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --folder PATH        Dataset folder (default: DroneCaptures)\n";
            std::cout << "  --init-frames N      Number of init frames (default: 10)\n";
            std::cout << "  --max-frames N       Limit processing (default: all)\n";
            std::cout << "  --help, -h           Show this help\n";
            return 0;
        }
    }
    
    fs::path folder_path(folder);
    if (!fs::exists(folder_path)) {
        std::cerr << "ERROR: Folder not found: " << folder << "\n";
        return 1;
    }
    
    try {
        return runTest(folder_path, init_frames, max_frames);
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}

