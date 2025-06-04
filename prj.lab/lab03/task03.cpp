#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

cv::Mat createSingleHistogram(const cv::Mat& channel, const cv::Scalar& color) {
    const int histSize = 256;
    const float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;

    int hist_w = 512, hist_h = 200;
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(240, 240, 240));

    cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    int bin_w = cvRound((double)hist_w / histSize);
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
            color, 2, 8, 0);
    }

    return histImage;
}

void saveHistogramCollage(const cv::Mat& original, const cv::Mat& processed,
    const std::string& output_image_path) {
    fs::path output_dir = fs::path("C:\\Users\\user\\Desktop\\misis2025s-3-nurgaliev-r-d\\prj.lab\\lab03\\output_histograms");
    fs::create_directories(output_dir);

    fs::path output_path(output_image_path);
    std::string hist_filename = output_path.stem().string() + "_histogram.png";
    fs::path full_path = output_dir / hist_filename;

    std::vector<cv::Mat> collage_rows;

    if (original.channels() == 3) {
        std::vector<cv::Mat> orig_channels, proc_channels;
        cv::split(original, orig_channels);
        cv::split(processed, proc_channels);

        for (int i = 0; i < 3; i++) {
            cv::Mat orig_hist = createSingleHistogram(orig_channels[i],
                i == 0 ? cv::Scalar(255, 0, 0) :
                i == 1 ? cv::Scalar(0, 255, 0) :
                cv::Scalar(0, 0, 255));

            cv::Mat proc_hist = createSingleHistogram(proc_channels[i],
                i == 0 ? cv::Scalar(255, 0, 0) :
                i == 1 ? cv::Scalar(0, 255, 0) :
                cv::Scalar(0, 0, 255));

            cv::Mat row;
            cv::hconcat(orig_hist, proc_hist, row);
            collage_rows.push_back(row);
        }
    }
    else {
        cv::Mat orig_hist = createSingleHistogram(original, cv::Scalar(0, 0, 0));
        cv::Mat proc_hist = createSingleHistogram(processed, cv::Scalar(0, 0, 0));

        cv::Mat row;
        cv::hconcat(orig_hist, proc_hist, row);
        collage_rows.push_back(row);
    }

    cv::Mat collage;
    cv::vconcat(collage_rows, collage);

    if (cv::imwrite(full_path.string(), collage)) {
        std::cout << "Histogram saved to: " << full_path.string() << std::endl;
    }
    else {
        std::cerr << "Failed to save histogram" << std::endl;
    }
}

cv::Mat autocontrast(const cv::Mat& img, double q_black, double q_white) {
    CV_Assert(img.type() == CV_8UC1);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    double total_pixels = img.rows * img.cols;
    double black_threshold = q_black * total_pixels;
    double white_threshold = (1.0 - q_white) * total_pixels;

    int black_level = 0;
    int white_level = 255;

    double sum = 0;
    for (int i = 0; i < histSize; ++i) {
        sum += hist.at<float>(i);
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

cv::Mat naive_autocontrast_rgb(const cv::Mat& img, double q_black, double q_white) {
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    for (int i = 0; i < 3; ++i) {
        channels[i] = autocontrast(channels[i], q_black, q_white);
    }

    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

cv::Mat autocontrast_rgb(const cv::Mat& img, double blackQuantile, double whiteQuantile) {
    if (img.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return img.clone();
    }

    if (blackQuantile < 0 || blackQuantile > 1 || whiteQuantile < 0 || whiteQuantile > 1) {
        std::cerr << "Error: Quantiles must be in the range [0, 1]." << std::endl;
        return img.clone();
    }

    if (img.channels() != 3) {
        std::cerr << "Error: Function supports only 3-channel RGB images." << std::endl;
        return img.clone();
    }

    cv::Mat lab_img;
    cv::cvtColor(img, lab_img, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_channels;
    cv::split(lab_img, lab_channels);

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    double totalPixels = gray.rows * gray.cols;
    int minThreshold = 0;
    int maxThreshold = 255;

    double sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += hist.at<float>(i);
        if (sum / totalPixels > blackQuantile) {
            minThreshold = i;
            break;
        }
    }

    sum = 0;
    for (int i = 255; i >= 0; --i) {
        sum += hist.at<float>(i);
        if (sum / totalPixels > whiteQuantile) {
            maxThreshold = i;
            break;
        }
    }

    if (minThreshold >= maxThreshold) {
        std::cerr << "Warning: Invalid thresholds (min=" << minThreshold << ", max=" << maxThreshold << ")" << std::endl;
        return img.clone();
    }

    cv::Mat lut(1, 256, CV_8U);
    double scale = 255.0 / (maxThreshold - minThreshold);

    for (int i = 0; i < 256; ++i) {
        if (i <= minThreshold) {
            lut.at<uchar>(i) = 0;
        }
        else if (i >= maxThreshold) {
            lut.at<uchar>(i) = 255;
        }
        else {
            lut.at<uchar>(i) = cv::saturate_cast<uchar>((i - minThreshold) * scale);
        }
    }

    std::vector<cv::Mat> bgr_channels;
    cv::split(img, bgr_channels);

    for (int i = 0; i < 3; ++i) {
        cv::LUT(bgr_channels[i], lut, bgr_channels[i]);
    }

    cv::Mat balanced_rgb;
    cv::merge(bgr_channels, balanced_rgb);

    cv::Mat balanced_lab;
    cv::cvtColor(balanced_rgb, balanced_lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> balanced_lab_channels;
    cv::split(balanced_lab, balanced_lab_channels);

    balanced_lab_channels[1] = lab_channels[1].clone();
    balanced_lab_channels[2] = lab_channels[2].clone();

    cv::Mat result;
    cv::merge(balanced_lab_channels, balanced_lab);
    cv::cvtColor(balanced_lab, result, cv::COLOR_Lab2BGR);

    return result;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <naive|rgb> <input_image> <q_black> <q_white> <output_image>" << std::endl;
        return 1;
    }

    std::string type = argv[1];
    std::string input_path = argv[2];
    double q_black = std::stod(argv[3]);
    double q_white = std::stod(argv[4]);
    std::string output_path = argv[5];

    cv::Mat img = cv::imread(input_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Could not open or find the image: " << input_path << std::endl;
        return 1;
    }

    cv::Mat result;
    if (type == "naive") {
        if (img.channels() == 3) {
            result = naive_autocontrast_rgb(img, q_black, q_white);
        }
        else {
            result = autocontrast(img, q_black, q_white);
        }
    }
    else if (type == "rgb") {
        if (img.channels() == 3) {
            result = autocontrast_rgb(img, q_black, q_white);
        }
        else {
            std::cerr << "RGB autocontrast requires a color image" << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << "Invalid type: " << type << std::endl;
        return 1;
    }

    if (cv::imwrite(output_path, result)) {
        std::cout << "Image saved to: " << output_path << std::endl;
    }
    else {
        std::cerr << "Failed to save image" << std::endl;
        return 1;
    }

    saveHistogramCollage(img, result, output_path);
    return 0;
}