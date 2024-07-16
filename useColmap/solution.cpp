#include <iostream>
#include <cstdlib> // ����std::system
#include <fstream> // ����std::ofstream
#include <algorithm>
#include <string>

// ת���ַ���Ϊ��д
std::string to_upper(const std::string& str) {
    std::string upper_str = str;
    std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), ::toupper);
    return upper_str;
}

int execute_colmap_commands(const std::string& colmap_path,
                            const std::string& database_path,
                            const std::string& image_path,
                            const std::string& output_path) {
    // ����COLMAP feature_extractor
    std::string feature_extractor_command = colmap_path + " feature_extractor --database_path " + database_path + " --image_path " + image_path;
    int return_code = std::system(feature_extractor_command.c_str());
    if (return_code != 0) {
        std::cerr << "Error executing feature_extractor command!" << std::endl;
        return return_code;
    }

    // ����COLMAP exhaustive_matcher
    std::string exhaustive_matcher_command = colmap_path + " exhaustive_matcher --database_path " + database_path;
    return_code = std::system(exhaustive_matcher_command.c_str());
    if (return_code != 0) {
        std::cerr << "Error executing exhaustive_matcher command!" << std::endl;
        return return_code;
    }

    // ����COLMAP mapper
    std::string mapper_command = colmap_path + " mapper --database_path " + database_path + " --image_path " + image_path + " --output_path " + output_path;
    return_code = std::system(mapper_command.c_str());
    if (return_code != 0) {
        std::cerr << "Error executing mapper command!" << std::endl;
        return return_code;
    }

    // ����COLMAP����ģ��Ϊ�ı�
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
    
    // ����COLMAP����
    std::string database_path = "D:\\practice\\test_scene\\database.db";
    std::string image_path = "D:\\practice\\test_scene\\images";
    std::string output_path = "D:\\practice\\test_scene\\output";
    std::string colmap_path = "D:\\����\\COLMAP-3.9.1-windows-cuda\\COLMAP-3.9.1-windows-cuda\\colmap.bat";
    // ����COLMAP  
    int result = execute_colmap_commands(colmap_path, database_path, image_path, output_path);
    return result;

    std::string file_path = output_path + "\\0\\images.txt ";

    // ���ļ�
    std::ifstream file(file_path);

    // ����ļ��Ƿ�ɹ���
    if (!file.is_open()) {
        std::cerr << "�޷����ļ�: " << file_path << std::endl;
        return 1;
    }

    std::string line;
    // ���ж�ȡ�ļ�
    while (std::getline(file, line)) {
        // ת�����ַ���Ϊ��д
        std::string upper_line = to_upper(line);
        // ��������Ƿ���� "JPG"
        if (upper_line.find("JPG") != std::string::npos) {
            // ������� "JPG" ����
            std::cout << line << std::endl;
        }
    }

    // �ر��ļ�
    file.close();

    return 0;
}
