#include "utils.h"

#include <chrono>  // for high_resolution_clock
#include <iomanip>
#include <iostream>

std::string get_current_time_string() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm timeinfo = *std::localtime(&time);
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << timeinfo.tm_hour << "-"
       << std::setfill('0') << std::setw(2) << timeinfo.tm_min << "-"
       << std::setfill('0') << std::setw(2) << timeinfo.tm_sec;
    return ss.str();
}

std::string removeFileExtension(const std::string& filename) {
    size_t lastDotPos = filename.find_last_of(".");
    if (lastDotPos == std::string::npos) {
        // No file extension found
        return filename;
    } else {
        return filename.substr(0, lastDotPos);
    }
}

std::string removePath(std::string filepath) {
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash == std::string::npos) {
        // No path separator found, return the original string
        return filepath;
    } else {
        return filepath.substr(lastSlash + 1);
    }
}
