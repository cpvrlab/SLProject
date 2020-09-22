/**
 * File: VocGenerator.cpp
 * Date: September 2019
 * Author: Luc Girod
 * Description: DBoW2 voc generator
 */

#include <iostream>
#include <vector>

#include <WAIOrbVocabulary.h>
#include <KPextractor.h>
#include <ORBextractor.h>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

int loadFeatures(std::string videoname, vector<cv::Mat>& features);
void changeStructure(const cv::Mat& plain, vector<cv::Mat>& out);
void changeStructure(const cv::Mat& plain, std::map<std::string, cv::Mat>& map,  vector<cv::Mat>& out);
void buildVoc(vector<cv::Mat>& features);
std::string desc_to_str(cv::Mat& descriptor);


int main(int argc, char ** argv)
{
    vector<cv::Mat> features;
    features.clear();

    for (int i = 1; i < argc; i++)
    {
        int nb_features = loadFeatures(argv[i], features);
        std::cout << nb_features << " features loaded" << std::endl;
    }

    buildVoc(features);

    return 0;
}

int loadFeatures(std::string videoname, vector<cv::Mat>& features)
{
    int nb_features = 0;
    int nLevels = 2;
    cv::VideoCapture vidcap;
    vidcap.open(videoname);
    if (!vidcap.isOpened())
    {
        std::cout << "Can't open video " << videoname << std::endl;
        return 0;
    }

    ORB_SLAM2::ORBextractor ext = ORB_SLAM2::ORBextractor(2000, 1.2, nLevels, 20, 7);
    cv::Mat frame;
    cv::Mat grayframe;

    std::cout << "Extracting features..." << std::endl;
    while (vidcap.read(frame))
    {
        cv::cvtColor(frame, grayframe, cv::COLOR_BGR2GRAY);

        vector<cv::KeyPoint> keypoints;
        cv::Mat              descriptors;
        cv::Mat              mask;

        ext(grayframe, keypoints, descriptors);
        features.push_back(descriptors);
        nb_features += descriptors.rows;
    }
    vidcap.release();
    return nb_features;
}

//Validate that there is a unique string per descriptor
std::string desc_to_str(cv::Mat& descriptor)
{
    uchar * ptr = descriptor.ptr();
    std::string str(ptr, ptr + descriptor.cols);

    if (str.length() != descriptor.cols)
    {
        std::cout << "Should not happend" << std::endl;
    }

    return str;
}

void buildVoc(vector<cv::Mat>& features)
{
    WAIOrbVocabulary voc;

    voc.create(features);
    cout << "Saving vocabulary..." << endl;
    voc.save("voc.bin");
    cout << "Done" << endl;
}
