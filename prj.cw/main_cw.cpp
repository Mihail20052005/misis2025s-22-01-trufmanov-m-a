#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

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
    cv::destroyWindow(winname);
}


void plot_rgb(const cv::Mat& image, const std::string& winname = "RGB Image") {
    cv::imshow(winname, image);
    cv::waitKey(0);
    cv::destroyWindow(winname);
}


cv::Mat dilate_image(const cv::Mat& image, int kernel_size = 9) {
    cv::Mat dilated;
    cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::dilate(image, dilated, rectKernel);
    return dilated;
}

double pointDist(const cv::Point& a, const cv::Point& b) {
    return std::sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

void draw_normal(cv::Mat& image, const cv::Point& p1, const cv::Point& p2, double length = 30.0, cv::Scalar color = cv::Scalar(0, 0, 255)) {
    cv::Point2f v = p2 - p1;
    cv::Point2f normal(-v.y, v.x);
    double norm_length = std::sqrt(normal.x * normal.x + normal.y * normal.y);
    normal.x /= norm_length;
    normal.y /= norm_length;
    cv::Point2f mid = (p1 + p2) * 0.5;
    cv::Point2f normal_end = mid + normal * static_cast<float>(length);
    cv::arrowedLine(image, mid, normal_end, color, 2, cv::LINE_AA);
}


void draw_coordinate_axes(cv::Mat& image, int length = 50) {
    cv::Point origin(40, image.rows - 40);
    cv::arrowedLine(image, origin, cv::Point(origin.x + length, origin.y), cv::Scalar(255, 0, 0), 2); // X axis - Red
    cv::putText(image, "X", cv::Point(origin.x + length + 10, origin.y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

    cv::arrowedLine(image, origin, cv::Point(origin.x, origin.y - length), cv::Scalar(0, 255, 0), 2); // Y axis - Green
    cv::putText(image, "Y", cv::Point(origin.x - 10, origin.y - length - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}


bool compareContourAreas(const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
    return cv::contourArea(c1, false) > cv::contourArea(c2, false);
}


double calculateLineAngle(const cv::Point& p1, const cv::Point& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return std::atan2(dy, dx) * 180.0 / CV_PI;
}


double calculateCheckAngle(const std::vector<cv::Point>& contour) {

    cv::RotatedRect minRect = cv::minAreaRect(contour);


    cv::Point2f rect_points[4];
    minRect.points(rect_points);


    double max_length = 0;
    int longest_side_idx = 0;

    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;
        double length = pointDist(rect_points[i], rect_points[j]);
        if (length > max_length) {
            max_length = length;
            longest_side_idx = i;
        }
    }

    cv::Point p1 = rect_points[longest_side_idx];
    cv::Point p2 = rect_points[(longest_side_idx + 1) % 4];
    double angle = calculateLineAngle(p1, p2);

    return angle;
}

double process_image_interactive(const std::string& file_path) {
    cv::Mat image = cv::imread(file_path);
    if (image.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение: " << file_path << std::endl;
        return -1.0;
    }


    double resize_ratio = 500.0 / image.rows;
    cv::Mat resized = opencv_resize(image, resize_ratio);
    plot_rgb(resized, "1. Input image");


    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    plot_gray(gray, "2. Gray");


    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(15, 15), 3);
    plot_gray(blurred, "3. Blurred");

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(blurred, blurred, cv::MORPH_CLOSE, kernel);
    plot_gray(blurred, "4. Morfologic");

    cv::Mat dilated = dilate_image(blurred, 9);
    plot_gray(dilated, "5. Dilate");


    cv::Mat edged;
    cv::Canny(dilated, edged, 50, 125, 3);
    plot_gray(edged, "6. Edge");

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edged, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {

        std::sort(contours.begin(), contours.end(), compareContourAreas);


        std::vector<cv::Point> largest_contour = contours[0];

        double angle = calculateCheckAngle(largest_contour);
        if (angle > 0) {
            angle = 90 - angle;
        }
        if (angle < 0) {
            angle = -angle;
        }

        cv::Mat image_with_largest_contour = resized.clone();
        std::vector<std::vector<cv::Point>> draw_contours = {largest_contour};
        cv::drawContours(image_with_largest_contour, draw_contours, -1, cv::Scalar(0, 255, 0), 3);
        draw_coordinate_axes(image_with_largest_contour);

        cv::RotatedRect minRect = cv::minAreaRect(largest_contour);
        cv::Point2f rect_points[4];
        minRect.points(rect_points);

        for (int j = 0; j < 4; j++) {
            cv::line(image_with_largest_contour, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            draw_normal(image_with_largest_contour, rect_points[j], rect_points[(j + 1) % 4], 30.0, cv::Scalar(0, 0, 255));
        }

        cv::Point2f center = minRect.center;
        std::string angle_text = "Angle " + std::to_string(angle).substr(0, 6) + " degr";
        cv::putText(image_with_largest_contour, angle_text, center, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);

        plot_rgb(image_with_largest_contour, "7. With angle");

        return angle;
    } else {
        std::cout << "Контуры не найдены в файле: " << file_path << std::endl;
        return -1.0;
    }
}

double process_image_batch(const std::string& file_path) {
    cv::Mat image = cv::imread(file_path);
    if (image.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение: " << file_path << std::endl;
        return -1.0;
    }

    double resize_ratio = 500.0 / image.rows;
    cv::Mat resized = opencv_resize(image, resize_ratio);


    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(15, 15), 3);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(blurred, blurred, cv::MORPH_CLOSE, kernel);

    cv::Mat dilated = dilate_image(blurred, 9);

    cv::Mat edged;
    cv::Canny(dilated, edged, 50, 125, 3);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edged, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        std::sort(contours.begin(), contours.end(), compareContourAreas);

        std::vector<cv::Point> largest_contour = contours[0];

        double angle = calculateCheckAngle(largest_contour);
        return angle;
    } else {
        std::cout << "Контуры не найдены в файле: " << file_path << std::endl;
        return -1.0;
    }
}

void process_batch_mode() {
    std::string res_folder = "/Users/mtrufmanov/MisisProject/misis2025s-22-01-trufmanov-m-a/prj.cw/res";
    std::string output_file = "/Users/mtrufmanov/MisisProject/misis2025s-22-01-trufmanov-m-a/prj.cw/res.txt";

    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл для записи результатов: " << output_file << std::endl;
        return;
    }

    outfile << "Filename,Angle (degrees)" << std::endl;

    for (const auto& entry : fs::directory_iterator(res_folder)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::string filename = entry.path().filename().string();


            double angle = process_image_batch(file_path);


            outfile << filename << "," << angle << std::endl;
            std::cout << "Обработан файл: " << filename << " - угол: " << angle << " градусов" << std::endl;
        }
    }

    outfile.close();
    std::cout << "Обработка завершена. Результаты сохранены в: " << output_file << std::endl;
}



void process_interactive_mode() {
    std::string file_path;
    std::cout << "Введите путь к изображению: ";
    std::getline(std::cin, file_path);

    double angle = process_image_interactive(file_path);
    std::cout << "Вычисленный угол поворота чека: " << angle << " градусов" << std::endl;
}

int main() {
    int mode = 0;
    std::cout << "Выберите режим работы:\n";
    std::cout << "1 - Интерактивный режим (обработка одного изображения с отображением этапов)\n";
    std::cout << "2 - Пакетный режим (обработка всех изображений в папке res с сохранением в файл)\n";
    std::cout << "Введите 1 или 2: ";
    std::cin >> mode;
    std::cin.ignore();

    if (mode == 1) {
        process_interactive_mode();
    } else if (mode == 2) {
        process_batch_mode();
    } else {
        std::cerr << "Ошибка: неверный режим. Допустимые значения: 1 или 2" << std::endl;
        return 1;
    }

    return 0;
}