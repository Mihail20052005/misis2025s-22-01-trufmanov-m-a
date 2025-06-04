#ifndef TASK05_HPP
#define TASK05_HPP

#include <opencv2/opencv.hpp>

const int SQUARE_SIZE = 127;
const int CIRCLE_RADIUS = 40;

cv::Mat createCollage();

cv::Mat filter1(const cv::Mat& input);
cv::Mat filter2(const cv::Mat& input);
cv::Mat filter3(const cv::Mat& f1, const cv::Mat& f2);
cv::Mat filter_rgb(const cv::Mat& f1, const cv::Mat& f2, const cv::Mat& f3);

#endif