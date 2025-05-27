#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../semcv/semcv.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_lst_file>" << std::endl;
        return 1;
    }

    std::filesystem::path lst_path(argv[1]);
    auto file_paths = semcv::get_list_of_file_paths(lst_path);

    for (const auto& file_path : file_paths) {
        cv::Mat img = cv::imread(file_path.string(), cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Could not read image: " << file_path << std::endl;
            continue;
        }

        std::string expected_format = semcv::strid_from_mat(img);
        std::string actual_format = file_path.filename().string();

        if (actual_format.find(expected_format) != std::string::npos) {
            std::cout << file_path.filename() << "\tgood" << std::endl;
        }
        else {
            std::cout << file_path.filename() << "\tbad, should be " << expected_format << std::endl;
        }
    }

    return 0;
}