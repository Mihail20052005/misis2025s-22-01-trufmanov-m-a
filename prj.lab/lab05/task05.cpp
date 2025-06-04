#include "task05.hpp"
#include <vector>
#include <iostream>
#include <cmath>

cv::Mat createSingleImage(const cv::Vec3b& squareColor, const cv::Vec3b& circleColor) {
    cv::Mat image(SQUARE_SIZE, SQUARE_SIZE, CV_8UC3, squareColor);
    cv::circle(image, cv::Point(SQUARE_SIZE / 2, SQUARE_SIZE / 2), CIRCLE_RADIUS, cv::Scalar(circleColor), cv::FILLED);
    return image;
}

cv::Mat createCollage() {
    std::vector<std::pair<cv::Vec3b, cv::Vec3b>> combinations = {
        {cv::Vec3b(255, 255, 255), cv::Vec3b(0, 0, 0)},
        {cv::Vec3b(0, 0, 0), cv::Vec3b(127, 127, 127)},
        {cv::Vec3b(127, 127, 127), cv::Vec3b(255, 255, 255)},
        {cv::Vec3b(127, 127, 127), cv::Vec3b(0, 0, 0)},
        {cv::Vec3b(255, 255, 255), cv::Vec3b(127, 127, 127)},
        {cv::Vec3b(0, 0, 0), cv::Vec3b(255, 255, 255)}
    };

    cv::Mat collage(SQUARE_SIZE * 2, SQUARE_SIZE * 3, CV_8UC3);

    for (int i = 0; i < 6; ++i) {
        int row = i / 3;
        int col = i % 3;
        cv::Rect roi(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE);
        createSingleImage(combinations[i].first, combinations[i].second).copyTo(collage(roi));
    }

    return collage;
}

cv::Mat filter1(const cv::Mat& input) {
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        1, 0, -1,
        2, 0, -2,
        1, 0, -1);

    cv::Mat gray, floatResult, result;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    cv::filter2D(gray, floatResult, CV_32F, kernel);
    floatResult.convertTo(result, CV_8U, 127.0, 128); 

    return result;
}

cv::Mat filter2(const cv::Mat& input) {
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1);

    cv::Mat gray, floatResult, result;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    cv::filter2D(gray, floatResult, CV_32F, kernel);
    floatResult.convertTo(result, CV_8U, 127.0, 128); 

    return result;
}

cv::Mat filter3(const cv::Mat& f1, const cv::Mat& f2) {
    cv::Mat f1_float, f2_float;
    f1.convertTo(f1_float, CV_32F, 1.0, -128.0); 
    f2.convertTo(f2_float, CV_32F, 1.0, -128.0); 

    cv::Mat magnitude;
    cv::magnitude(f1_float, f2_float, magnitude);

    magnitude.convertTo(magnitude, CV_8U, 1.0, 128); 

    return magnitude;
}

cv::Mat filter_rgb(const cv::Mat& f1, const cv::Mat& f2, const cv::Mat& f3) {
    std::vector<cv::Mat> channels = { f1, f2, f3 };
    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

int main() {
    try {
        cv::Mat collage = createCollage();
        std::string outputPath = "C:/Users/user/Desktop/misis2025s-3-nurgaliev-r-d/prj.lab/lab05/result_collage.png";
        cv::imwrite(outputPath, collage);

        cv::Mat filtered1 = filter1(collage);
        cv::Mat filtered2 = filter2(collage);
        cv::Mat filtered3 = filter3(filtered1, filtered2);
        cv::Mat filtered_rgb = filter_rgb(filtered1, filtered2, filtered3);

        cv::imwrite("C:/Users/user/Desktop/misis2025s-3-nurgaliev-r-d/prj.lab/lab05/filters/filter1.png", filtered1);
        cv::imwrite("C:/Users/user/Desktop/misis2025s-3-nurgaliev-r-d/prj.lab/lab05/filters/filter2.png", filtered2);
        cv::imwrite("C:/Users/user/Desktop/misis2025s-3-nurgaliev-r-d/prj.lab/lab05/filters/filter3.png", filtered3);
        cv::imwrite("C:/Users/user/Desktop/misis2025s-3-nurgaliev-r-d/prj.lab/lab05/filters/filter_rgb.png", filtered_rgb);

        cv::Mat f1_color, f2_color, f3_color;
        cv::cvtColor(filtered1, f1_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(filtered2, f2_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(filtered3, f3_color, cv::COLOR_GRAY2BGR);

        cv::Mat visualization(f1_color.rows * 2, f1_color.cols * 2, CV_8UC3, cv::Scalar(128, 128, 128));

        f1_color.copyTo(visualization(cv::Rect(0, 0, f1_color.cols, f1_color.rows)));
        f2_color.copyTo(visualization(cv::Rect(f1_color.cols, 0, f2_color.cols, f2_color.rows)));
        f3_color.copyTo(visualization(cv::Rect(0, f1_color.rows, f3_color.cols, f3_color.rows)));
        filtered_rgb.copyTo(visualization(cv::Rect(f1_color.cols, f1_color.rows, filtered_rgb.cols, filtered_rgb.rows)));

        std::string visualizationPath = "C:/Users/user/Desktop/misis2025s-3-nurgaliev-r-d/prj.lab/lab05/result_visualization.png";
        cv::imwrite(visualizationPath, visualization);

        std::cout << "All images saved successfully!" << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}