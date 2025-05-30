#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "../semcv/semcv.hpp"
#include <filesystem>
#include <iomanip>
#include <vector>

std::string get_image_label(size_t row, size_t col) {
    return "(" + std::to_string(row + 1) + ", " + std::to_string(col + 1) + ")";
}

void save_statistics_to_csv(const std::string& filename, const std::vector<std::vector<std::string>>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "Image,Region,Mean,StdDev\n";  

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Statistics saved to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::string output_path = "output_collage.png";
    std::string hist_output_path = "output_histogram.png";
    std::string stats_output_path = "statistics.csv";

    if (argc == 2) {
        output_path = argv[1];
        std::filesystem::path path(output_path);
        std::filesystem::path directory = path.parent_path();
        std::string filename = path.stem().string();
        std::string extension = path.extension().string();
        hist_output_path = (directory / "output_histogram.png").string();
        stats_output_path = (directory / (filename + "_stats.csv")).string();
    }

    std::vector<int> noise_levels = { 3, 7, 15 };
    std::vector<std::vector<int>> levels = {
        {0, 127, 255},
        {20, 127, 235},
        {55, 127, 200},
        {90, 127, 165}
    };

    std::vector<cv::Mat> original_imgs;
    for (const auto& level : levels) {
        original_imgs.push_back(semcv::gen_tgtimg00(level[0], level[1], level[2]));
    }

    cv::Mat hor_concat;
    cv::hconcat(original_imgs, hor_concat);

    std::vector<cv::Mat> noisy_imgs;
    for (int std : noise_levels) {
        noisy_imgs.push_back(semcv::add_noise_gau(hor_concat, std));
    }

    std::vector<cv::Mat> final_imgs = { hor_concat };
    final_imgs.insert(final_imgs.end(), noisy_imgs.begin(), noisy_imgs.end());

    cv::Mat final_result;
    cv::vconcat(final_imgs, final_result);

    if (cv::imwrite(output_path, final_result)) {
        std::cout << "Collage saved to: " << output_path << std::endl;
    }
    else {
        std::cerr << "Failed to save the collage." << std::endl;
        return 1;
    }

    std::vector<cv::Mat> hist_rows;
    for (const auto& row_img : final_imgs) {
        std::vector<cv::Mat> hist_row;
        for (int i = 0; i < 4; ++i) {
            int start_x = i * 256;
            cv::Rect roi(start_x, 0, 256, 256);
            cv::Mat single_img = row_img(roi);
            cv::Mat hist = semcv::create_histogram(single_img);
            hist_row.push_back(hist);
        }
        cv::Mat hist_row_concat;
        cv::hconcat(hist_row, hist_row_concat);
        hist_rows.push_back(hist_row_concat);
    }

    cv::Mat hist_final_result;
    cv::vconcat(hist_rows, hist_final_result);

    if (cv::imwrite(hist_output_path, hist_final_result)) {
        std::cout << "Histogram saved to: " << hist_output_path << std::endl;
    }
    else {
        std::cerr << "Failed to save histogram." << std::endl;
        return 1;
    }

    std::vector<std::vector<std::string>> stats_data;

    for (size_t img_idx = 0; img_idx < original_imgs.size(); ++img_idx) {
        const cv::Mat& img = original_imgs[img_idx];

        cv::Mat mask_background = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::Mat mask_square = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::Mat mask_circle = cv::Mat::zeros(img.size(), CV_8UC1);

        cv::rectangle(mask_square, cv::Point(23, 23), cv::Point(232, 232), cv::Scalar(255), -1);
        cv::circle(mask_circle, cv::Point(128, 128), 83, cv::Scalar(255), -1);
        mask_background = 255 - (mask_square + mask_circle);

        double mean_background, stddev_background;
        double mean_square, stddev_square;
        double mean_circle, stddev_circle;

        semcv::calculate_distribution_params(img, mask_background, mean_background, stddev_background);
        semcv::calculate_distribution_params(img, mask_square, mean_square, stddev_square);
        semcv::calculate_distribution_params(img, mask_circle, mean_circle, stddev_circle);

        std::string image_label = "Image " + std::to_string(img_idx + 1);
        stats_data.push_back({ image_label, "Background", std::to_string(mean_background), std::to_string(stddev_background) });
        stats_data.push_back({ image_label, "Square", std::to_string(mean_square), std::to_string(stddev_square) });
        stats_data.push_back({ image_label, "Circle", std::to_string(mean_circle), std::to_string(stddev_circle) });
    }

    for (size_t noise_idx = 0; noise_idx < noise_levels.size(); ++noise_idx) {
        for (size_t img_idx = 0; img_idx < original_imgs.size(); ++img_idx) {
            cv::Mat noisy_img = semcv::add_noise_gau(original_imgs[img_idx], noise_levels[noise_idx]);

            cv::Mat mask_background = cv::Mat::zeros(noisy_img.size(), CV_8UC1);
            cv::Mat mask_square = cv::Mat::zeros(noisy_img.size(), CV_8UC1);
            cv::Mat mask_circle = cv::Mat::zeros(noisy_img.size(), CV_8UC1);

            cv::rectangle(mask_square, cv::Point(23, 23), cv::Point(232, 232), cv::Scalar(255), -1);
            cv::circle(mask_circle, cv::Point(128, 128), 83, cv::Scalar(255), -1);
            mask_background = 255 - (mask_square + mask_circle);

            double mean_background, stddev_background;
            double mean_square, stddev_square;
            double mean_circle, stddev_circle;

            semcv::calculate_distribution_params(noisy_img, mask_background, mean_background, stddev_background);
            semcv::calculate_distribution_params(noisy_img, mask_square, mean_square, stddev_square);
            semcv::calculate_distribution_params(noisy_img, mask_circle, mean_circle, stddev_circle);

            std::string image_label = "Noisy Image " + std::to_string(img_idx + 1) + " (std=" + std::to_string(noise_levels[noise_idx]) + ")";
            stats_data.push_back({ image_label, "Background", std::to_string(mean_background), std::to_string(stddev_background) });
            stats_data.push_back({ image_label, "Square", std::to_string(mean_square), std::to_string(stddev_square) });
            stats_data.push_back({ image_label, "Circle", std::to_string(mean_circle), std::to_string(stddev_circle) });
        }
    }

    save_statistics_to_csv(stats_output_path, stats_data);

    return 0;
}