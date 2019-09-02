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
#include <opencv2/xfeatures2d.hpp>
#include "convert.h"

//const int PATCH_SIZE      = 31;
//const int HALF_PATCH_SIZE = 15;
const int PATCH_SIZE      = 61;
const int HALF_PATCH_SIZE = 30;
const int EDGE_THRESHOLD  = 34;

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

void init_patch(std::vector<int> &umax, int half_patch_size = HALF_PATCH_SIZE);

void init_pyramid_parameters(PyramidParameters &p, int nlevels, float scale_factor, int nfeatures);

void build_pyramid(std::vector<cv::Mat> &image_pyramid, cv::Mat &image, PyramidParameters &p);

void flatten_keypoints(std::vector<cv::KeyPoint> &keypoints, std::vector<std::vector<cv::KeyPoint>>& all_keypoints, PyramidParameters &p);

void flatten_decriptors(std::vector<Descriptor> &desc, std::vector<std::vector<Descriptor>>& all_desc, PyramidParameters &p);

unsigned int hamming_distance(unsigned int a, unsigned int b);

unsigned int hamming_distance(Descriptor &a, Descriptor &b);

void print_desc(Descriptor &d);

void compute_three_maxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3);

cv::Mat extract_patch(cv::Mat& image, cv::KeyPoint &kp);

void get_inverted_matching(std::vector<int> &inverted_matching, std::vector<int> &matching);

int select_closest_feature(std::vector<cv::KeyPoint> &keypoints, int x, int y);

int select_closest_feature(std::vector<cv::KeyPoint> &keypoints, std::vector<int> matches, int x, int y);

std::vector<int> select_closest_features(std::vector<cv::KeyPoint> &keypoints, float radius, int x, int y);

void compute_similarity(std::vector<cv::KeyPoint> &keypoints, std::vector<Descriptor> &descs, Descriptor &cur);

void reset_similarity_score(std::vector<cv::KeyPoint> &keypoints);

std::vector<std::string> str_split(const std::string& str, char delim = '\n');

void filters_open(std::string path, std::vector<float> &param, std::vector<float> &bias, std::vector<std::vector<float>> &coeffs, std::vector<cv::Mat> &filters, std::vector<std::string> &tokens);

std::vector<cv::Mat> image_gradient(const cv::Mat &input_rgb_image);

std::vector<cv::Point3f> NonMaxSup(const cv::Mat &response);

float angle_from_gradiant(cv::Mat &image, cv::KeyPoint &kp);

#endif

