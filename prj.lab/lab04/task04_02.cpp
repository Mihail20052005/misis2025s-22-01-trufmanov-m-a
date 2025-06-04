#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <algorithm>

using json = nlohmann::json;

struct DetectedEllipse {
    double angle;
    int height;
    int width;
    int x;
    int y;
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

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image.png> <output_result.json>\n";
        return 1;
    }

    try {
        cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("Could not read input image");
        }

        const int grid_size = 5;
        const int segment_size = 256;
        std::vector<DetectedEllipse> all_ellipses;

        for (int row = 0; row < grid_size; ++row) {
            for (int col = 0; col < grid_size; ++col) {
                cv::Rect roi(col * segment_size, row * segment_size, segment_size, segment_size);
                cv::Mat segment = image(roi).clone();
                auto ellipses = detectEllipses(segment);
                for (auto& e : ellipses) {
                    e.x += col * segment_size;
                    e.y += row * segment_size;
                    all_ellipses.push_back(e);
                }
            }
        }

        json result;
        result["detected_objects"] = json::array();
        for (const auto& ellipse : all_ellipses) {
            result["detected_objects"].push_back({
                {"angle", ellipse.angle},
                {"height", ellipse.height},
                {"width", ellipse.width},
                {"x", ellipse.x},
                {"y", ellipse.y}
                });
        }

        std::ofstream out(argv[2]);
        out << result.dump(4);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
