#include <iostream>
#include <cstdlib> // 包含std::system
#include <fstream> // 包含std::ofstream
#include <algorithm>
#include <string>

// 转换字符串为大写
std::string to_upper(const std::string& str) {
    std::string upper_str = str;
    std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), ::toupper);
    return upper_str;
}

int execute_colmap_commands(const std::string& colmap_path,
                            const std::string& database_path,
                            const std::string& image_path,
                            const std::string& output_path) {
    // 调用COLMAP feature_extractor
    std::string feature_extractor_command = colmap_path + " feature_extractor --database_path " + database_path + " --image_path " + image_path;
    int return_code = std::system(feature_extractor_command.c_str());
    if (return_code != 0) {
        std::cerr << "Error executing feature_extractor command!" << std::endl;
        return return_code;
    }

    // 调用COLMAP exhaustive_matcher
    std::string exhaustive_matcher_command = colmap_path + " exhaustive_matcher --database_path " + database_path;
    return_code = std::system(exhaustive_matcher_command.c_str());
    if (return_code != 0) {
        std::cerr << "Error executing exhaustive_matcher command!" << std::endl;
        return return_code;
    }

    // 调用COLMAP mapper
    std::string mapper_command = colmap_path + " mapper --database_path " + database_path + " --image_path " + image_path + " --output_path " + output_path;
    return_code = std::system(mapper_command.c_str());
    if (return_code != 0) {
        std::cerr << "Error executing mapper command!" << std::endl;
        return return_code;
    }

    // 调用COLMAP导出模型为文本
    std::string export_model_command = colmap_path + " model_converter --input_path " + output_path + "\\0 --output_path " + output_path + "\\0 --output_type TXT";
    return_code = std::system(export_model_command.c_str());
    if (return_code != 0) {
        std::cerr << "Error exporting model as text!" << std::endl;
        return return_code;
    }

    std::cout << "COLMAP commands executed successfully!" << std::endl;
    return 0;
}

int main() {
    
    // 定义COLMAP命令
    std::string database_path = "D:\\practice\\test_scene\\database.db";
    std::string image_path = "D:\\practice\\test_scene\\images";
    std::string output_path = "D:\\practice\\test_scene\\output";
    std::string colmap_path = "D:\\桌面\\COLMAP-3.9.1-windows-cuda\\COLMAP-3.9.1-windows-cuda\\colmap.bat";
    // 调用COLMAP  
    int result = execute_colmap_commands(colmap_path, database_path, image_path, output_path);
    return result;

    std::string file_path = output_path + "\\0\\images.txt ";

    // 打开文件
    std::ifstream file(file_path);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return 1;
    }

    std::string line;
    // 逐行读取文件
    while (std::getline(file, line)) {
        // 转换行字符串为大写
        std::string upper_line = to_upper(line);
        // 检查行中是否包含 "JPG"
        if (upper_line.find("JPG") != std::string::npos) {
            // 输出包含 "JPG" 的行
            std::cout << line << std::endl;
        }
    }

    // 关闭文件
    file.close();

    return 0;
}
