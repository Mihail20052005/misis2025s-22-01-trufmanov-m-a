#include <iostream>
#include <opencv2/opencv.hpp>
#include "../semcv/semcv.hpp"

int main(int argc, char** argv) {
    std::string output_path = "/Users/mtrufmanov/Desktop/output_collage.png";

    if (argc == 2) {
        output_path = argv[1];
    }

    cv::Mat striped_img = semcv::generate_striped_image();

    std::vector<double> gammas = { 1.8, 2.0, 2.2, 2.4, 2.6 };
    std::vector<cv::Mat> gamma_imgs;

    for (double gamma : gammas) {
        gamma_imgs.push_back(semcv::gamma_correction(striped_img, gamma));
    }

    cv::Mat output_collage;
    cv::vconcat(striped_img, gamma_imgs[0], output_collage);
    for (size_t i = 1; i < gamma_imgs.size(); ++i) {
        cv::vconcat(output_collage, gamma_imgs[i], output_collage);
    }

    if (cv::imwrite(output_path, output_collage)) {
        std::cout << "Collage saved to: " << output_path << std::endl;
    }
    else {
        std::cerr << "Error: Failed to save collage to " << output_path << std::endl;
    }

    return 0;
}