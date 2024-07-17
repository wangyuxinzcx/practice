#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <cstdlib>

namespace fs = std::experimental::filesystem;

class CaptureVideo {
public:
    static std::string to_upper(const std::string& str) {
        std::string upper_str = str;
        std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), ::toupper);
        return upper_str;
    }

    static int execute_colmap_commands(const std::string& colmap_path,
        const std::string& database_path,
        const std::string& image_path,
        const std::string& output_path) {

        std::string database_creator_command = colmap_path + " database_creator --database_path " + database_path;
        int return_code = std::system(database_creator_command.c_str());
        if (return_code != 0) {
            std::cerr << "Error executing database_creator command!" << std::endl;
            return return_code;
        }

        std::string feature_extractor_command = colmap_path + " feature_extractor --database_path " + database_path + " --image_path " + image_path;
        return_code = std::system(feature_extractor_command.c_str());
        if (return_code != 0) {
            std::cerr << "Error executing feature_extractor command!" << std::endl;
            return return_code;
        }

        std::string exhaustive_matcher_command = colmap_path + " exhaustive_matcher --database_path " + database_path;
        return_code = std::system(exhaustive_matcher_command.c_str());
        if (return_code != 0) {
            std::cerr << "Error executing exhaustive_matcher command!" << std::endl;
            return return_code;
        }

        std::string mapper_command = colmap_path + " mapper --database_path " + database_path + " --image_path " + image_path + " --output_path " + output_path;
        return_code = std::system(mapper_command.c_str());
        if (return_code != 0) {
            std::cerr << "Error executing mapper command!" << std::endl;
            return return_code;
        }

        std::string export_model_command = colmap_path + " model_converter --input_path " + output_path + "\\0 --output_path " + output_path + "\\0 --output_type TXT";
        return_code = std::system(export_model_command.c_str());
        if (return_code != 0) {
            std::cerr << "Error exporting model as text!" << std::endl;
            return return_code;
        }

        std::cout << "COLMAP commands executed successfully!" << std::endl;
        return 0;
    }

    static std::vector<std::vector<double>> quaternionToRotationMatrix(double w, double x, double y, double z) {
        std::vector<std::vector<double>> R(3, std::vector<double>(3));

        R[0][0] = 1 - 2 * y * y - 2 * z * z;
        R[0][1] = 2 * x * y - 2 * w * z;
        R[0][2] = 2 * x * z + 2 * w * y;

        R[1][0] = 2 * x * y + 2 * w * z;
        R[1][1] = 1 - 2 * x * x - 2 * z * z;
        R[1][2] = 2 * y * z - 2 * w * x;

        R[2][0] = 2 * x * z - 2 * w * y;
        R[2][1] = 2 * y * z + 2 * w * x;
        R[2][2] = 1 - 2 * x * x - 2 * y * y;

        return R;
    }

    static void save_lines(const std::string& input_file_path, const std::string& output_file_path) {
        std::ifstream input_file(input_file_path);
        if (!input_file.is_open()) {
            std::cerr << "无法打开输入文件: " << input_file_path << std::endl;
            return;
        }

        std::ofstream output_file(output_file_path);
        if (!output_file.is_open()) {
            std::cerr << "无法打开输出文件: " << output_file_path << std::endl;
            input_file.close();
            return;
        }

        std::string line;
        while (std::getline(input_file, line)) {
            std::string upper_line = to_upper(line);
            if (upper_line.find("JPG") != std::string::npos) {
                output_file << line << std::endl;
            }
        }

        input_file.close();
        output_file.close();
    }

    static void save_lines_containing_jpg(const std::string& input_file_path, const std::string& output_xml_path) {
        std::ifstream input_file(input_file_path);
        if (!input_file.is_open()) {
            std::cerr << "无法打开输入文件: " << input_file_path << std::endl;
            return;
        }

        std::ofstream output_xml(output_xml_path);
        if (!output_xml.is_open()) {
            std::cerr << "无法创建 XML 文件: " << output_xml_path << std::endl;
            input_file.close();
            return;
        }

        output_xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
        output_xml << "<opencv_storage>" << std::endl;

        std::string line;
        int jpg_count = 0;

        while (std::getline(input_file, line)) {
            std::string upper_line = line;
            std::transform(upper_line.begin(), upper_line.end(), upper_line.begin(), ::toupper);

            if (upper_line.find("JPG") != std::string::npos) {
                std::vector<double> doubles;
                std::istringstream iss(line);
                double num;
                int count = 0;
                int pre_jpg_num = 0;
                bool pre_jpg_num_found = false;

                while (iss >> num) {
                    count++;
                    if (count == 9) {
                        pre_jpg_num = num;
                        pre_jpg_num_found = true;
                        break;
                    }
                    else if (count >= 2 && count <= 5) {
                        doubles.push_back(num);
                    }
                }

                if (doubles.size() == 4 && pre_jpg_num_found) {
                    std::vector<std::vector<double>> rotation_matrix = quaternionToRotationMatrix(doubles[0], doubles[1], doubles[2], doubles[3]);

                    output_xml << "<H_mat" << pre_jpg_num << " type_id=\"opencv-matrix\">" << std::endl;
                    output_xml << "  <pre_jpg_num>" << pre_jpg_num << "</pre_jpg_num>" << std::endl;
                    output_xml << "  <rows>3</rows>" << std::endl;
                    output_xml << "  <cols>3</cols>" << std::endl;
                    output_xml << "  <dt>d</dt>" << std::endl;
                    output_xml << "  <data>" << std::endl;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            output_xml << "    " << rotation_matrix[i][j];
                        }
                        output_xml << std::endl;
                    }
                    output_xml << "  </data>" << std::endl;
                    output_xml << "</H_mat" << pre_jpg_num << ">" << std::endl;
                    jpg_count++;
                }
            }

            if (jpg_count >= 5) {
                break;
            }
        }

        output_xml << "</opencv_storage>" << std::endl;

        input_file.close();
        output_xml.close();

        std::cout << "XML 文件保存成功: " << output_xml_path << std::endl;
    }
};

int main() {
    std::string database_path = "D:\\practice\\database.db";
    std::string image_path = "D:\\practice\\test_scene\\images";
    std::string output_path = "D:\\practice\\test_scene\\output_2";
    std::string colmap_path = "D:\\桌面\\COLMAP-3.9.1-windows-cuda\\COLMAP-3.9.1-windows-cuda\\colmap.bat";

    int result = CaptureVideo::execute_colmap_commands(colmap_path, database_path, image_path, output_path);

    CaptureVideo::save_lines_containing_jpg(output_path + "\\0\\images.txt", output_path + "\\0\\images_simplified.xml");

    return 0;
}
