#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct EllipseParameters {
    int x;
    int y;
    int width;
    int height;
    double angle;
};

struct Config {
    std::string output_path;
    int n;
    int bg_color;
    int elps_color;
    int noise_std;
    int blur_size;
    int min_elps_width;
    int max_elps_width;
    int min_elps_height;
    int max_elps_height;
    int seed;
};

void loadConfig(const std::string& configPath, Config& config) {
    std::ifstream configFile(configPath);
    json configJson;
    configFile >> configJson;
    config.output_path = configJson["output_path"];
    config.n = configJson["n"];
    config.bg_color = configJson["bg_color"];
    config.elps_color = configJson["elps_color"];
    config.noise_std = configJson["noise_std"];
    config.blur_size = configJson["blur_size"];
    config.min_elps_width = configJson["min_elps_width"];
    config.max_elps_width = configJson["max_elps_width"];
    config.min_elps_height = configJson["min_elps_height"];
    config.max_elps_height = configJson["max_elps_height"];
    config.seed = configJson.value("seed", 0);
}

cv::Mat generateCollage(const Config& config, std::vector<EllipseParameters>& allParams) {
    std::mt19937 gen(config.seed);
    const int margin = 32;
    const int singleImageSize = 256;
    const int collageSize = config.n * singleImageSize;
    cv::Mat collage(collageSize, collageSize, CV_8UC1, cv::Scalar(config.bg_color));

    for (int row = 0; row < config.n; ++row) {
        for (int col = 0; col < config.n; ++col) {
            EllipseParameters params;
            params.width = std::uniform_int_distribution<int>(config.min_elps_width, config.max_elps_width)(gen);
            params.height = std::uniform_int_distribution<int>(config.min_elps_height, config.max_elps_height)(gen);

            int minX = margin + params.width / 2;
            int maxX = singleImageSize - margin - params.width / 2;
            int minY = margin + params.height / 2;
            int maxY = singleImageSize - margin - params.height / 2;

            params.x = std::uniform_int_distribution<int>(minX, maxX)(gen);
            params.y = std::uniform_int_distribution<int>(minY, maxY)(gen);
            params.angle = std::uniform_real_distribution<double>(0.0, 360.0)(gen);

            cv::Mat singleImage(singleImageSize, singleImageSize, CV_8UC1, cv::Scalar(config.bg_color));
            cv::ellipse(singleImage, cv::Point(params.x, params.y), cv::Size(params.width / 2, params.height / 2), params.angle, 0, 360, cv::Scalar(config.elps_color), cv::FILLED);

            cv::GaussianBlur(singleImage, singleImage, cv::Size(config.blur_size, config.blur_size), 0);

            cv::Mat noise(singleImage.size(), CV_8UC1);
            cv::randn(noise, 0, config.noise_std);
            singleImage = cv::max(cv::min(singleImage + noise, 255), 0);

            cv::Rect roi(col * singleImageSize, row * singleImageSize, singleImageSize, singleImageSize);
            singleImage.copyTo(collage(roi));

            params.x += col * singleImageSize;
            params.y += row * singleImageSize;
            allParams.push_back(params);
        }
    }
    return collage;
}

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json> [<output_image.png> <output_gt.json>] [--seed <value>]\n";
        return 1;
    }

    try {
        Config config;
        loadConfig(argv[1], config);

        std::string image_path = config.output_path + ".png";
        std::string gt_path = config.output_path + "_gt.json";

        int arg_pos = 2;
        while (arg_pos < argc && std::string(argv[arg_pos]) != "--seed") {
            if (arg_pos == 2 && arg_pos + 1 < argc) {
                image_path = argv[arg_pos];
                gt_path = argv[arg_pos + 1];
                arg_pos += 2;
            }
            else {
                std::cerr << "Invalid arguments\n";
                return 1;
            }
        }

        if (arg_pos < argc && std::string(argv[arg_pos]) == "--seed") {
            if (arg_pos + 1 < argc) {
                config.seed = std::stoi(argv[arg_pos + 1]);
            }
        }

        std::vector<EllipseParameters> allParams;
        cv::Mat collage = generateCollage(config, allParams);

        std::filesystem::create_directories(std::filesystem::path(image_path).parent_path());
        cv::imwrite(image_path, collage);

        json groundTruth;
        groundTruth["blur_size"] = config.blur_size;
        groundTruth["colors"]["bg_color"] = config.bg_color;
        groundTruth["colors"]["elps_color"] = config.elps_color;
        groundTruth["noise_std"] = config.noise_std;
        groundTruth["size_of_collage"] = config.n;

        for (size_t i = 0; i < allParams.size(); ++i) {
            const auto& params = allParams[i];
            json obj;
            obj["pic_coordinates"]["row"] = i / config.n;
            obj["pic_coordinates"]["col"] = i % config.n;
            obj["elps_parameters"]["elps_x"] = params.x;
            obj["elps_parameters"]["elps_y"] = params.y;
            obj["elps_parameters"]["elps_width"] = params.width;
            obj["elps_parameters"]["elps_height"] = params.height;
            obj["elps_parameters"]["elps_angle"] = params.angle;
            groundTruth["objects"].push_back(obj);
        }

        std::ofstream gtFile(gt_path);
        gtFile << groundTruth.dump(4);

        std::cout << "Collage generated successfully!\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}