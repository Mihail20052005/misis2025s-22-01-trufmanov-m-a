#include "semcv.hpp"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>
#include <fstream>  
#include <filesystem>

namespace semcv {

    std::string strid_from_mat(const cv::Mat& img, const int n) {
        std::ostringstream oss;
        oss << std::setw(n) << std::setfill('0') << img.cols << "x"
            << std::setw(n) << std::setfill('0') << img.rows << "."
            << img.channels() << "."
            << (img.depth() == CV_8U ? "uint08" :
                img.depth() == CV_8S ? "sint08" :
                img.depth() == CV_16U ? "uint16" :
                img.depth() == CV_16S ? "sint16" :
                img.depth() == CV_32S ? "sint32" :
                img.depth() == CV_32F ? "real32" :
                img.depth() == CV_64F ? "real64" : "unknown");
        return oss.str();
    }

    std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst) {
        std::vector<std::filesystem::path> file_paths;
        std::ifstream file(path_lst);
        std::string line;

        std::filesystem::path base_dir = path_lst.parent_path();

        while (std::getline(file, line)) {
            file_paths.push_back(base_dir / line);
        }
        return file_paths;
    }

    cv::Mat generate_striped_image() {
        cv::Mat img(30, 768, CV_8UC1);
        for (int i = 0; i < img.cols; i += 3) {
            img.col(i).setTo(cv::Scalar((i / 3) % 256));
            img.col(i + 1).setTo(cv::Scalar((i / 3) % 256));
            img.col(i + 2).setTo(cv::Scalar((i / 3) % 256));
        }
        return img;
    }

    cv::Mat gamma_correction(const cv::Mat& img, double gamma) {
        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        cv::Mat res;
        cv::LUT(img, lookUpTable, res);
        return res;
    }

    cv::Mat gen_tgtimg00(const int lev0, const int lev1, const int lev2) {
        const int img_size = 256;
        const int square_size = 209;
        const int circle_rad = 83;

        cv::Mat img(img_size, img_size, CV_8UC1, cv::Scalar(lev0));

        int start = (img_size - square_size) / 2;
        cv::rectangle(img, cv::Rect(start, start, square_size, square_size), cv::Scalar(lev1), cv::FILLED);

        cv::circle(img, cv::Point(img_size / 2, img_size / 2), circle_rad, cv::Scalar(lev2), cv::FILLED);

        return img;
    }

    cv::Mat add_noise_gau(const cv::Mat& img, const int std) {
        cv::Mat noise(img.size(), CV_16SC1);
        cv::randn(noise, cv::Scalar(0), cv::Scalar(std));
        cv::Mat noisy_img;
        cv::add(img, noise, noisy_img, cv::noArray(), CV_8UC1);
        return noisy_img;
    }

    cv::Mat compute_histogram(const cv::Mat& img) {
        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        cv::Mat hist;
        cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

        int hist_h = 256;
        int hist_w = 256;
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::normalize(hist, hist, 0, 250, cv::NORM_MINMAX, -1, cv::Mat());

        int bin_w = cvRound((double)hist_w / histSize);
        for (int i = 1; i < histSize; i++) {
            cv::line(histImage,
                cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
                cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
                cv::Scalar(0, 0, 0), 2, 8, 0);
        }
        return histImage;
    }

    cv::Mat create_histogram(const cv::Mat& img) {
        const int HIST_SIZE = 256;
        const float RANGE[] = { 0, 256 };
        const float* HIST_RANGE = { RANGE };

        cv::Mat hist;
        cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &HIST_SIZE, &HIST_RANGE);

        int hist_w = 512;
        int hist_h = 400;
        int bin_w = cvRound((double)hist_w / HIST_SIZE);

        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

        cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

        for (int i = 1; i < HIST_SIZE; i++) {
            cv::line(histImage,
                cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
                cv::Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
                cv::Scalar(0, 0, 0), 2, 8, 0);
        }

        return histImage;
    }

    void calculate_distribution_params(const cv::Mat& img, const cv::Mat& mask, double& mean, double& stddev) {
        cv::Scalar mean_scalar, stddev_scalar;
        cv::meanStdDev(img, mean_scalar, stddev_scalar, mask);

        mean = mean_scalar[0];
        stddev = stddev_scalar[0];
    }

    cv::Mat autocontrast(const cv::Mat& img, const double q_black, const double q_white) {
        CV_Assert(img.type() == CV_8UC1); 

        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        cv::Mat hist;
        cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        double total_pixels = img.rows * img.cols;
        double black_threshold = q_black * total_pixels;
        double white_threshold = q_white * total_pixels;

        int black_level = 0;
        int white_level = 255;

        double sum = 0;
        for (int i = 0; i < histSize; ++i) {
            sum += hist.at<float>(i) * total_pixels;
            if (sum >= black_threshold && black_level == 0) {
                black_level = i;
            }
            if (sum >= white_threshold) {
                white_level = i;
                break;
            }
        }

        cv::Mat result;
        cv::Mat lut(1, 256, CV_8U);
        uchar* p = lut.ptr();
        for (int i = 0; i < 256; ++i) {
            if (i <= black_level) {
                p[i] = 0;
            }
            else if (i >= white_level) {
                p[i] = 255;
            }
            else {
                p[i] = cv::saturate_cast<uchar>(255.0 * (i - black_level) / (white_level - black_level));
            }
        }

        cv::LUT(img, lut, result);
        return result;
    }

    cv::Mat autocontrast_rgb(const cv::Mat& img, const double q_black, const double q_white) {
        CV_Assert(img.type() == CV_8UC3); 

        std::vector<cv::Mat> channels;
        cv::split(img, channels);

        for (int i = 0; i < 3; ++i) {
            channels[i] = autocontrast(channels[i], q_black, q_white);
        }

        cv::Mat result;
        cv::merge(channels, result);
        return result;
    }

} // namespace semcv