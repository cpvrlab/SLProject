#ifndef CONVERT_H
#define CONVERT_H

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat rgb_to_grayscale(cv::Mat &img);

std::vector<cv::Mat> rgb_to_luv(const cv::Mat &input_color_image);

#endif

