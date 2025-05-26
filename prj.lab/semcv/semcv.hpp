#ifndef SEMCV_HPP
#define SEMCV_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>

namespace semcv {

    std::string strid_from_mat(const cv::Mat& img, const int n = 4);
    std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst);
    cv::Mat generate_striped_image();
    cv::Mat gamma_correction(const cv::Mat& img, double gamma);

    cv::Mat gen_tgtimg00(const int lev0, const int lev1, const int lev2);
    cv::Mat add_noise_gau(const cv::Mat& img, const int std);
    cv::Mat create_histogram(const cv::Mat& img);
    void calculate_distribution_params(const cv::Mat& img, const cv::Mat& mask, double& mean, double& stddev);

    cv::Mat autocontrast(const cv::Mat& img, const double q_black, const double q_white);
    cv::Mat autocontrast_rgb(const cv::Mat& img, const double q_black, const double q_white);

} // namespace semcv

#endif // SEMCV_HPP