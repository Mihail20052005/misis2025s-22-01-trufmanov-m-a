#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

cv::Mat opencv_resize(const cv::Mat& image, double ratio) {
    int width = static_cast<int>(image.cols * ratio);
    int height = static_cast<int>(image.rows * ratio);
    cv::Size dim(width, height);
    cv::Mat resized;
    cv::resize(image, resized, dim, 0, 0, cv::INTER_AREA);
    return resized;
}

void plot_gray(const cv::Mat& image, const std::string& winname = "Gray Image") {
    cv::imshow(winname, image);
    cv::waitKey(0);
}

void plot_rgb(const cv::Mat& image, const std::string& winname = "RGB Image") {
    cv::imshow(winname, image);
    cv::waitKey(0);
}

cv::Mat dilate_image(const cv::Mat& image, int kernel_size=9) {
    cv::Mat dilated;
    cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::dilate(image, dilated, rectKernel);
    return dilated;
}

double pointDist(const cv::Point& a, const cv::Point& b) {
    return std::sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

void connect_close_contours(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours, double max_dist = 20.0) {
    for (size_t i = 0; i < contours.size(); ++i) {
        const cv::Point& start_i = contours[i].front();
        const cv::Point& end_i = contours[i].back();

        for (size_t j = i + 1; j < contours.size(); ++j) {
            const cv::Point& start_j = contours[j].front();
            const cv::Point& end_j = contours[j].back();

            std::vector<std::pair<cv::Point, cv::Point>> pairs = {
                    {start_i, start_j},
                    {start_i, end_j},
                    {end_i, start_j},
                    {end_i, end_j}
            };

            for (auto& p : pairs) {
                if (pointDist(p.first, p.second) < max_dist) {
                    cv::line(image, p.first, p.second, cv::Scalar(0, 0, 255), 2);
                }
            }
        }
    }
}

void draw_normal(cv::Mat& image, const cv::Point& p1, const cv::Point& p2, double length = 30.0, cv::Scalar color = cv::Scalar(0,0,255)) {
    cv::Point2f v = p2 - p1;

    cv::Point2f normal(-v.y, v.x);

    double norm_length = std::sqrt(normal.x*normal.x + normal.y*normal.y);
    normal.x /= norm_length;
    normal.y /= norm_length;

    cv::Point2f mid = (p1 + p2) * 0.5;


    cv::Point2f normal_end = mid + normal * static_cast<float>(length);


    cv::arrowedLine(image, mid, normal_end, color, 2, cv::LINE_AA);
}

int main() {
    std::string file_name = "/Users/mtrufmanov/MisisProject/misis2025s-22-01-trufmanov-m-a/prj.cw/photo_example.jpg"; // Укажи путь к своему изображению
    cv::Mat image = cv::imread(file_name);
    if (image.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение!" << std::endl;
        return -1;
    }


    double resize_ratio = 500.0 / image.rows;
    cv::Mat resized = opencv_resize(image, resize_ratio);


    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    plot_gray(gray, "Gray");

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(11, 11), 1);
    plot_gray(blurred, "Blurred");


    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(blurred, blurred, cv::MORPH_CLOSE, kernel);
    plot_gray(blurred, "Morph Close");


    cv::Mat dilated = dilate_image(blurred, 9);
    plot_gray(dilated, "Dilated");


    cv::Mat edged;
    cv::Canny(dilated, edged, 50, 125, 3);
    plot_gray(edged, "Edged");


    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edged, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


    cv::Mat image_with_contours = resized.clone();
    cv::drawContours(image_with_contours, contours, -1, cv::Scalar(0, 255, 0), 3);


    connect_close_contours(image_with_contours, contours, 20.0);

    plot_rgb(image_with_contours, "All Contours with Connections");


    if (!contours.empty()) {
        std::sort(contours.begin(), contours.end(),
                  [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                      return cv::contourArea(c1, false) > cv::contourArea(c2, false);
                  });
        std::vector<cv::Point> largest_contour = contours[0];

        cv::Mat image_with_largest_contour = resized.clone();
        cv::drawContours(image_with_largest_contour, std::vector<std::vector<cv::Point>>{largest_contour}, -1, cv::Scalar(0, 255, 0), 3);


        cv::RotatedRect minRect = cv::minAreaRect(largest_contour);


        double angle = minRect.angle;
        cv::Point2f rect_points[4];
        minRect.points(rect_points);
        for (int j = 0; j < 4; j++) {
            cv::line(image_with_largest_contour, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }

        std::cout << "Угол поворота чека: " << angle << " градусов" << std::endl;


        for (int j = 0; j < 4; j++) {
            draw_normal(image_with_largest_contour, rect_points[j], rect_points[(j + 1) % 4], 30.0, cv::Scalar(0, 0, 255));
        }

        plot_rgb(image_with_largest_contour, "Largest Contour with MinAreaRect and Normals");
    } else {
        std::cout << "Контуры не найдены." << std::endl;
    }

    return 0;
}
