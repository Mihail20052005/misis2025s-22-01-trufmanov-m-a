#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>

int main() {
    std::filesystem::path test_images_dir = "/Users/mtrufmanov/MisisProject/misis2025s-22-01-trufmanov-m-a/prj.lab/lab01/test_images";
    std::filesystem::create_directories(test_images_dir); 

    std::vector<cv::Size> sizes = {
        cv::Size(100, 100),
        cv::Size(200, 200),
        cv::Size(300, 300)
    };

    std::vector<int> types = {
        CV_8UC1,  // Single-channel 8-bit unsigned
        CV_8UC3,  // Three-channel 8-bit unsigned
        CV_16UC1, // Single-channel 16-bit unsigned
        CV_32FC1  // Single-channel 32-bit floating point
    };

    std::vector<std::string> formats = { "png", "tiff", "jpg" };

    std::ofstream lst_file(test_images_dir / "task01.lst");
    if (!lst_file.is_open()) {
        std::cerr << "Failed to open task01.lst for writing!" << std::endl;
        return 1;
    }

    for (const auto& size : sizes) {
        for (int type : types) {
            cv::Mat img(size, type);

            if (type == CV_8UC1 || type == CV_16UC1 || type == CV_32FC1) {
                cv::randu(img, cv::Scalar(0), cv::Scalar(255));
            }
            else if (type == CV_8UC3) {
                cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255)); 
            }

            std::string type_str;
            switch (type) {
            case CV_8UC1: type_str = "uint08"; break;
            case CV_8UC3: type_str = "uint08x3"; break;
            case CV_16UC1: type_str = "uint16"; break;
            case CV_32FC1: type_str = "real32"; break;
            default: type_str = "unknown"; break;
            }

            for (const auto& format : formats) {
                std::string filename = cv::format("%04dx%04d.%d.%s.%s",
                    size.width, size.height, img.channels(), type_str.c_str(), format.c_str());
                std::filesystem::path file_path = test_images_dir / filename;

                if (format == "jpg") {
                    cv::imwrite(file_path.string(), img, { cv::IMWRITE_JPEG_QUALITY, 95 });
                }
                else {
                    cv::imwrite(file_path.string(), img);
                }

                lst_file << filename << std::endl;
            }
        }
    }

    lst_file.close();
    std::cout << "Images generated and task01.lst file created." << std::endl;
    return 0;
}