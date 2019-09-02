#ifndef TOOLS_H
#define TOOLS_H

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

const int PATCH_SIZE      = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD  = 19;

typedef struct PyramidParameters
{
    std::vector<float> scale_factors;
    std::vector<float> level_sigma2;
    std::vector<float> inv_scale_factors;
    std::vector<float> inv_level_sigma2;
    std::vector<int> nb_feature_per_level;
    int total_features;
}PyramidParameters;

typedef struct Descriptor
{
    uchar * p;
    uchar mem[32];
}Descriptor;

void init_patch(std::vector<int> &umax);

void init_pyramid_parameters(PyramidParameters &p, int nlevels, float scale_factor, int nfeatures);

void build_pyramid(std::vector<cv::Mat> &image_pyramid, cv::Mat &image, PyramidParameters &p);

void flatten_keypoints(std::vector<cv::KeyPoint> &keypoints, std::vector<std::vector<cv::KeyPoint>>& all_keypoints, PyramidParameters &p);

void flatten_decriptors(std::vector<Descriptor> &desc, std::vector<std::vector<Descriptor>>& all_desc, PyramidParameters &p);

cv::Mat to_grayscale(cv::Mat &img);

unsigned int hamming_distance(unsigned int a, unsigned int b);

unsigned int hamming_distance(Descriptor &a, Descriptor &b);

void print_desc(Descriptor &d);

void compute_three_maxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3);

cv::KeyPoint get_middle_keypoint(cv::Mat image);

void keypoint_angle(const cv::Mat& image, cv::KeyPoint &kp, const std::vector<int>& u_max);

#endif

