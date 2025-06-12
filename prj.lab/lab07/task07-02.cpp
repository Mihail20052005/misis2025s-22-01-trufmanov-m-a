#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Ellipse {
    double x, y;
    double width, height;
    double angle;
};

std::vector<std::string> read_file_list(const std::string& file_path) {
    std::vector<std::string> files;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file list: " + file_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            files.push_back(line);
        }
    }

    return files;
}

std::vector<Ellipse> read_ground_truth(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open ground truth file: " + file_path);
    }

    json data = json::parse(file);
    std::vector<Ellipse> ellipses;

    for (const auto& obj : data["objects"]) {
        const auto& params = obj["elps_parameters"];
        Ellipse e;
        e.x = params["elps_x"];
        e.y = params["elps_y"];
        e.width = params["elps_width"];
        e.height = params["elps_height"];
        e.angle = params["elps_angle"];
        ellipses.push_back(e);
    }

    return ellipses;
}

std::vector<Ellipse> read_detected(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open detected file: " + file_path);
    }

    json data = json::parse(file);
    std::vector<Ellipse> ellipses;

    for (const auto& obj : data) {
        Ellipse e;
        e.x = obj["x"];
        e.y = obj["y"];
        e.width = obj["width"];
        e.height = obj["height"];
        e.angle = obj["angle"];
        ellipses.push_back(e);
    }

    return ellipses;
}

double ellipse_overlap(const Ellipse& e1, const Ellipse& e2) {
    double dx = e1.x - e2.x;
    double dy = e1.y - e2.y;
    double distance = std::sqrt(dx * dx + dy * dy);
    double avg_size = (e1.width + e1.height + e2.width + e2.height) / 4.0;
    double normalized_distance = distance / avg_size;

    double size_similarity = 1.0 - (std::abs(e1.width - e2.width) + std::abs(e1.height - e2.height)) /
        (e1.width + e1.height + e2.width + e2.height);

    double angle_diff = std::abs(e1.angle - e2.angle);
    if (angle_diff > 180) angle_diff = 360 - angle_diff;
    double angle_similarity = 1.0 - angle_diff / 180.0;

    return 0.4 * (1.0 - normalized_distance) + 0.3 * size_similarity + 0.3 * angle_similarity;
}

void compare_ellipses(const std::vector<Ellipse>& gt, const std::vector<Ellipse>& detected,
    int& true_positives, int& false_positives, int& false_negatives) {
    const double MIN_OVERLAP = 0.6;

    std::vector<bool> gt_matched(gt.size(), false);
    std::vector<bool> det_matched(detected.size(), false);

    for (size_t i = 0; i < gt.size(); ++i) {
        for (size_t j = 0; j < detected.size(); ++j) {
            if (!det_matched[j]) {
                double overlap = ellipse_overlap(gt[i], detected[j]);
                if (overlap > MIN_OVERLAP) {
                    true_positives++;
                    gt_matched[i] = true;
                    det_matched[j] = true;
                    break;
                }
            }
        }
    }

    false_negatives = static_cast<int>(std::count(gt_matched.begin(), gt_matched.end(), false));
    false_positives = static_cast<int>(std::count(det_matched.begin(), det_matched.end(), false));
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: task07-02 <gt_list.lst> <detected_list.lst> <output_report.json>\n";
        return 1;
    }

    try {
        std::string gt_list_path = argv[1];
        std::string detected_list_path = argv[2];
        std::string output_report_path = argv[3];

        auto gt_files = read_file_list(gt_list_path);
        auto detected_files = read_file_list(detected_list_path);

        if (gt_files.size() != detected_files.size()) {
            throw std::runtime_error("Number of ground truth and detected files doesn't match");
        }

        json report;
        report["evaluations"] = json::array();

        for (size_t i = 0; i < gt_files.size(); ++i) {
            auto gt_ellipses = read_ground_truth(gt_files[i]);
            auto detected_ellipses = read_detected(detected_files[i]);

            int true_positives = 0;
            int false_positives = 0;
            int false_negatives = 0;

            compare_ellipses(gt_ellipses, detected_ellipses,
                true_positives, false_positives, false_negatives);

            double precision = (true_positives + false_positives) > 0 ?
                static_cast<double>(true_positives) / (true_positives + false_positives) : 1.0;
            double recall = (true_positives + false_negatives) > 0 ?
                static_cast<double>(true_positives) / (true_positives + false_negatives) : 1.0;

            json evaluation;
            evaluation["file"] = gt_files[i];
            evaluation["metrics"]["true_positives"] = true_positives;
            evaluation["metrics"]["false_positives"] = false_positives;
            evaluation["metrics"]["false_negatives"] = false_negatives;
            evaluation["metrics"]["object_precision"] = precision;
            evaluation["metrics"]["object_recall"] = recall;

            report["evaluations"].push_back(evaluation);
        }

        fs::path output_path(output_report_path);
        if (!output_path.parent_path().empty()) {
            fs::create_directories(output_path.parent_path());
        }

        std::ofstream out_file(output_report_path);
        if (!out_file.is_open()) {
            throw std::runtime_error("Could not open output file: " + output_report_path);
        }
        out_file << report.dump(4);
        out_file.close();

        std::cout << "Report successfully generated: " << output_report_path << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}