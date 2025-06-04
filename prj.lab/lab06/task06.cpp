#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <filesystem>
#include <cmath>

struct DetectedObject {
    float angle;
    float width;
    float height;
    int x;
    int y;
};

void saveDetectionsToJson(const std::string& filename, const std::vector<DetectedObject>& detections) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    outFile << "{\n";
    outFile << "  \"detected_objects\": [\n";

    for (size_t i = 0; i < detections.size(); ++i) {
        const DetectedObject& detection = detections[i];

        outFile << "    {\n";
        outFile << "      \"angle\": " << detection.angle << ",\n";
        outFile << "      \"height\": " << detection.height << ",\n";
        outFile << "      \"width\": " << detection.width << ",\n";
        outFile << "      \"x\": " << detection.x << ",\n";
        outFile << "      \"y\": " << detection.y << "\n";
        outFile << "    }";

        if (i != detections.size() - 1) {
            outFile << ",\n";
        }
    }

    outFile << "\n  ]\n";
    outFile << "}\n";

    outFile.close();
}

std::vector<cv::KeyPoint> detectBlobs(const cv::Mat& img) {
    cv::Mat processed;
    cv::GaussianBlur(img, processed, cv::Size(9, 9), 0);
    cv::normalize(processed, processed, 0, 255, cv::NORM_MINMAX, CV_32F);
    cv::morphologyEx(processed, processed, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

    std::vector<cv::KeyPoint> keypoints;
    const int num_levels = 3;
    const float scale_factor = 2.0f;
    const int num_scales = 3;
    const float sigma = 5.0f;
    const float k = 1.414f;
    const float min_response = 0.04f * 255;
    const float min_diameter = 50.0f;
    const float overlap_threshold = 0.7f;

    std::vector<cv::Mat> pyramid;
    pyramid.push_back(processed);
    for (int i = 1; i < num_levels; ++i) {
        cv::Mat scaled;
        float scale = static_cast<float>(std::pow(scale_factor, i));
        cv::resize(processed, scaled, cv::Size(), 1.0 / scale, 1.0 / scale, cv::INTER_LINEAR);
        pyramid.push_back(scaled);
    }

    for (int level = 0; level < num_levels; ++level) {
        float scale = static_cast<float>(std::pow(scale_factor, level));
        cv::Mat prev_gaussian;

        for (int i = 0; i < num_scales; ++i) {
            float current_sigma = static_cast<float>(sigma * std::pow(k, i));
            cv::Mat gaussian;
            cv::GaussianBlur(pyramid[level], gaussian, cv::Size(0, 0), current_sigma, current_sigma, cv::BORDER_REPLICATE);

            if (i > 0) {
                cv::Mat dog;
                cv::subtract(gaussian, prev_gaussian, dog, cv::noArray(), CV_32F);

                cv::Mat dog_abs;
                cv::absdiff(dog, cv::Scalar(0), dog_abs);

                cv::Mat dilated;
                cv::dilate(dog_abs, dilated, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
                cv::Mat local_max = (dog_abs >= dilated) & (dog_abs > min_response);

                std::vector<cv::Point> points;
                cv::findNonZero(local_max, points);

                for (const auto& p : points) {
                    float blob_size = static_cast<float>(current_sigma * scale * std::sqrt(2));
                    if (blob_size < min_diameter) continue;

                    cv::KeyPoint kp;
                    kp.pt = cv::Point2f(p.x * scale, p.y * scale);
                    kp.size = blob_size;
                    kp.response = dog_abs.at<float>(p);

                    bool is_duplicate = false;
                    for (const auto& existing_kp : keypoints) {
                        float dist = static_cast<float>(cv::norm(kp.pt - existing_kp.pt));
                        if (dist < std::min(kp.size, existing_kp.size) * overlap_threshold) {
                            is_duplicate = true;
                            break;
                        }
                    }

                    if (!is_duplicate) {
                        keypoints.push_back(kp);
                    }
                }
            }
            prev_gaussian = gaussian.clone();
        }
    }

    std::vector<cv::KeyPoint> filtered_keypoints;
    float response_threshold = min_response * 2.0f;
    for (const auto& kp : keypoints) {
        if (kp.response > response_threshold) {
            filtered_keypoints.push_back(kp);
        }
    }

    return filtered_keypoints;
}

std::vector<DetectedObject> detectEllipses(const cv::Mat& image) {
    std::vector<DetectedObject> detections;

    std::vector<cv::KeyPoint> keypoints = detectBlobs(image);

    for (const auto& kp : keypoints) {
        float size = kp.size * 2.0f;
        cv::RotatedRect ellipse(kp.pt, cv::Size2f(size, size), 0.0f);

        double area = ellipse.size.width * ellipse.size.height * CV_PI / 4.0;
        if (area < 500 || area > 100000) continue;

        DetectedObject obj;
        obj.x = static_cast<int>(ellipse.center.x);
        obj.y = static_cast<int>(ellipse.center.y);
        obj.width = ellipse.size.width;
        obj.height = ellipse.size.height;
        obj.angle = ellipse.angle;

        detections.push_back(obj);
    }

    return detections;
}

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc < 3) {
        std::cerr << "Usage: task06 <image_path> <output_json>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    std::string outputJson = argv[2];

    cv::Mat imageGray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (imageGray.empty()) {
        std::cerr << "Error loading image! Check file path: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat imageColor = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (imageColor.empty()) {
        std::cerr << "Error loading color image! Check file path: " << imagePath << std::endl;
        return -1;
    }

    std::vector<DetectedObject> detections = detectEllipses(imageGray);
    saveDetectionsToJson(outputJson, detections);

    for (const DetectedObject& obj : detections) {
        cv::Point center(obj.x, obj.y);
        int radius = static_cast<int>((obj.width + obj.height) / 4.0);
        cv::Scalar color(0, 0, 255);
        cv::circle(imageColor, center, radius, color, 2);
    }

    std::filesystem::path jsonPath(outputJson);
    std::string outputImagePath = jsonPath.replace_extension(".png").string();
    cv::imwrite(outputImagePath, imageColor);

    std::cout << "Detection results saved to " << outputJson << std::endl;
    std::cout << "Visualization saved to " << outputImagePath << std::endl;

    return 0;
}
