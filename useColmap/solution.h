// CaptureProc.h
#ifndef CAPTUREPROC_H
#define CAPTUREPROC_H

#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <thread>
#include <chrono>

namespace fs = std::experimental::filesystem;

class CaptureVideo {
public:
    static std::string to_upper(const std::string& str);

    static int execute_colmap_commands(const std::string& colmap_path,
        const std::string& database_path,
        const std::string& image_path,
        const std::string& output_path);

    static std::vector<std::vector<double>> quaternionToRotationMatrix(double w, double x, double y, double z);

    static void save_lines(const std::string& input_file_path, const std::string& output_file_path);

    static void save_lines_containing_jpg(const std::string& input_file_path, const std::string& output_xml_path);

    static void capture_and_save_images(const std::string& camera_index, const std::string& save_path, int interval_seconds);
};

#endif // CAPTUREPROC_H
#pragma once
