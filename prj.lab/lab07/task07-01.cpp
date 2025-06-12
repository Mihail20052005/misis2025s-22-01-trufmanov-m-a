#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

void ensure_directory_exists(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}

struct DetectedEllipse {
    double angle;
    int height;
    int width;
    int x;
    int y;
    std::vector<cv::Point> contour;
};

std::vector<DetectedEllipse> detectEllipses(const cv::Mat& src) {
    std::vector<DetectedEllipse> ellipses;

    cv::Mat gray = src.clone();
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    cv::Mat binary;
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    cv::Mat morph;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, morph, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;

        double area = cv::contourArea(contour);
        if (area < 100 || area > 20000) continue;

        cv::RotatedRect rect = cv::fitEllipse(contour);

        DetectedEllipse ellipse;
        ellipse.angle = rect.angle;
        ellipse.height = cvRound(rect.size.height);
        ellipse.width = cvRound(rect.size.width);
        ellipse.x = cvRound(rect.center.x);
        ellipse.y = cvRound(rect.center.y);

        std::vector<cv::Point> approx_contour;
        double epsilon = 0.01 * cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx_contour, epsilon, true);

        if (approx_contour.size() > 4) {
            std::vector<cv::Point> smooth_contour;
            cv::convexHull(approx_contour, smooth_contour);
            ellipse.contour = smooth_contour;
        }
        else {
            ellipse.contour = approx_contour;
        }

        bool duplicate = false;
        for (const auto& existing : ellipses) {
            if (abs(existing.x - ellipse.x) < 20 && abs(existing.y - ellipse.y) < 20) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) {
            ellipses.push_back(ellipse);
        }
    }

    return ellipses;
}

json ellipses_to_json(const std::vector<DetectedEllipse>& ellipses) {
    json j = json::array();
    for (const auto& ellipse : ellipses) {
        json contour_json = json::array();
        for (const auto& point : ellipse.contour) {
            contour_json.push_back({ {"x", point.x}, {"y", point.y} });
        }

        j.push_back({
            {"angle", ellipse.angle},
            {"height", ellipse.height},
            {"width", ellipse.width},
            {"x", ellipse.x},
            {"y", ellipse.y},
            {"contour", contour_json}
            });
    }
    return j;
}

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc < 2) {
        std::cerr << "Usage: task07-01 <input_image> [output_dir]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_dir = argc > 2 ? argv[2] : "/Users/mtrufmanov/MisisProject/misis2025s-22-01-trufmanov-m-a/prj.lab";

    std::string boundaries_dir = output_dir + "/images_with_detected_boundaries";
    std::string json_dir = output_dir + "/image_boundaries";

    ensure_directory_exists(boundaries_dir);
    ensure_directory_exists(json_dir);

    cv::Mat image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: cannot load image " << input_path << "\n";
        return 1;
    }

    cv::Mat output_img;
    cv::cvtColor(image, output_img, cv::COLOR_GRAY2BGR);

    const int grid_size = 5;
    const int segment_size = 256;
    std::vector<DetectedEllipse> all_ellipses;

    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            cv::Rect roi(col * segment_size, row * segment_size, segment_size, segment_size);
            cv::Mat segment = image(roi).clone();
            auto ellipses = detectEllipses(segment);
            for (auto& e : ellipses) {
                for (auto& p : e.contour) {
                    p.x += col * segment_size;
                    p.y += row * segment_size;
                }
                e.x += col * segment_size;
                e.y += row * segment_size;
                all_ellipses.push_back(e);
            }
        }
    }

    for (const auto& ellipse : all_ellipses) {
        if (ellipse.contour.size() > 2) {
            cv::polylines(output_img, ellipse.contour, true, cv::Scalar(0, 0, 255), 2);
        }
    }

    std::string output_img_path = boundaries_dir + "\\" + fs::path(input_path).stem().string() + "_boundaries.png";
    if (!cv::imwrite(output_img_path, output_img)) {
        std::cerr << "Error: cannot write output image to " << output_img_path << "\n";
        return 1;
    }

    json ellipses_json = ellipses_to_json(all_ellipses);
    std::string json_path = json_dir + "\\" + fs::path(input_path).stem().string() + "_boundaries.json";

    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
        std::cerr << "Error: cannot write JSON to " << json_path << "\n";
        return 1;
    }
    json_file << ellipses_json.dump(4);
    json_file.close();

    std::cout << "Processing completed:\n";
    std::cout << " - Image with boundaries: " << output_img_path << "\n";
    std::cout << " - Boundaries description: " << json_path << "\n";

    return 0;
}