#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <filesystem>
#include <cmath>
#include <algorithm>

using json = nlohmann::json;

struct Ellipse {
    double angle;
    int height;
    int width;
    int x;
    int y;
    int row;
    int col;
};

std::vector<std::string> readFileList(const std::string& filePath) {
    std::vector<std::string> files;
    std::ifstream file(filePath);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            files.push_back(line);
        }
    }
    return files;
}

double ellipseSimilarity(const Ellipse& gt, const Ellipse& det) {
    double dist = std::sqrt(std::pow(gt.x - det.x, 2) + std::pow(gt.y - det.y, 2)) / 256.0;
    double width_diff = std::abs(gt.width - det.width) / (double)std::max(gt.width, det.width);
    double height_diff = std::abs(gt.height - det.height) / (double)std::max(gt.height, det.height);
    double angle_diff = std::min(std::abs(gt.angle - det.angle), 360 - std::abs(gt.angle - det.angle)) / 180.0;
    return 0.5 * dist + 0.3 * (width_diff + height_diff) + 0.2 * angle_diff;
}

json evaluateQuality(const std::vector<Ellipse>& groundTruth, const std::vector<Ellipse>& detected) {
    json metrics;
    const double SIMILARITY_THRESHOLD = 0.6;

    std::vector<bool> gt_matched(groundTruth.size(), false);
    std::vector<bool> det_matched(detected.size(), false);
    int true_positives = 0;

    for (size_t i = 0; i < groundTruth.size(); ++i) {
        double best_sim = SIMILARITY_THRESHOLD;
        size_t best_j = -1;

        for (size_t j = 0; j < detected.size(); ++j) {
            if (det_matched[j]) continue;

            double sim = ellipseSimilarity(groundTruth[i], detected[j]);
            if (sim < best_sim) {
                best_sim = sim;
                best_j = j;
            }
        }

        if (best_j != -1) {
            true_positives++;
            gt_matched[i] = true;
            det_matched[best_j] = true;
        }
    }

    int false_positives = static_cast<int>(detected.size()) - true_positives;
    int false_negatives = static_cast<int>(groundTruth.size()) - true_positives;

    metrics["true_positives"] = true_positives;
    metrics["false_positives"] = false_positives;
    metrics["false_negatives"] = false_negatives;
    metrics["object_precision"] = (true_positives + false_positives) > 0 ?
        static_cast<double>(true_positives) / (true_positives + false_positives) : 0.0;
    metrics["object_recall"] = (true_positives + false_negatives) > 0 ?
        static_cast<double>(true_positives) / (true_positives + false_negatives) : 0.0;

    return metrics;
}

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <gt_list.lst> <detected_list.lst> <report.json>\n";
        return 1;
    }

    try {
        auto gt_files = readFileList(argv[1]);
        auto detected_files = readFileList(argv[2]);

        if (gt_files.size() != detected_files.size()) {
            throw std::runtime_error("Number of ground truth and detected files must match");
        }

        json report;
        report["evaluations"] = json::array();

        for (size_t i = 0; i < gt_files.size(); ++i) {
            std::ifstream gt_file(gt_files[i]);
            std::ifstream det_file(detected_files[i]);

            if (!gt_file.is_open() || !det_file.is_open()) {
                throw std::runtime_error("Could not open input files");
            }

            json gt_json = json::parse(gt_file);
            json det_json = json::parse(det_file);

            std::vector<Ellipse> groundTruth;
            for (const auto& obj : gt_json["objects"]) {
                Ellipse ellipse;
                ellipse.x = obj["elps_parameters"]["elps_x"];
                ellipse.y = obj["elps_parameters"]["elps_y"];
                ellipse.width = obj["elps_parameters"]["elps_width"];
                ellipse.height = obj["elps_parameters"]["elps_height"];
                ellipse.angle = obj["elps_parameters"]["elps_angle"];
                ellipse.row = obj["pic_coordinates"]["row"];
                ellipse.col = obj["pic_coordinates"]["col"];
                groundTruth.push_back(ellipse);
            }

            std::vector<Ellipse> detected;
            for (const auto& obj : det_json["detected_objects"]) {
                Ellipse ellipse;
                ellipse.x = obj["x"];
                ellipse.y = obj["y"];
                ellipse.width = obj["width"];
                ellipse.height = obj["height"];
                ellipse.angle = obj["angle"];
                ellipse.row = ellipse.y / 256;
                ellipse.col = ellipse.x / 256;
                detected.push_back(ellipse);
            }

            report["evaluations"].push_back({
                {"file", gt_files[i]},
                {"metrics", evaluateQuality(groundTruth, detected)}
                });
        }

        std::ofstream out(argv[3]);
        out << report.dump(4);

        std::cout << "Report generated successfully!\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
