/**
 * File: VocGenerator.cpp
 * Date: September 2019
 * Author: Luc Girod
 * Description: DBoW2 voc generator
 */

#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace DBoW2;
using namespace std;

int loadFeatures(std::string videoname, vector<vector<cv::Mat>>& features);
void changeStructure(const cv::Mat& plain, vector<cv::Mat>& out);
void changeStructure(const cv::Mat& plain, std::map<std::string, cv::Mat>& map,  vector<cv::Mat>& out);
void buildVoc(const vector<vector<cv::Mat>>& features);
std::string desc_to_str(cv::Mat& descriptor);


int main(int argc, char ** argv)
{
    vector<vector<cv::Mat>> features;
    features.clear();

    for (int i = 1; i < argc; i++)
    {
        int nb_features = loadFeatures(argv[i], features);
        std::cout << nb_features << " features loaded" << std::endl;
    }

    buildVoc(features);

    return 0;
}

int loadFeatures(std::string videoname, vector<vector<cv::Mat>>& features)
{
    int nb_features = 0;
    cv::VideoCapture vidcap;
    vidcap.open(videoname);
    if (!vidcap.isOpened())
    {
        std::cout << "Can't open video " << videoname << std::endl;
        return 0;
    }

    //cv::Ptr<cv::ORB> orb = cv::ORB::create();
    SURFextractor ext(1500);
    std::map<std::string, cv::Mat> map;
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
        //orb->detectAndCompute(frame, mask, keypoints, descriptors);

        features.push_back(vector<cv::Mat>());
        changeStructure(descriptors, map, features.back());
        nb_features += features.back().size();
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

void changeStructure(const cv::Mat& plain, vector<cv::Mat>& out)
{
    out.resize(plain.rows);

    for (int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}

void changeStructure(const cv::Mat& plain, std::map<std::string, cv::Mat>& map,  vector<cv::Mat>& out)
{
    for (int i = 0; i < plain.rows; ++i)
    {
        cv::Mat m = plain.row(i);
        std::string str = desc_to_str(m);
        if (map.find(str) == map.end())
        {
            out.push_back(m);
            map[str] = m;
        }
    }
}

void buildVoc(const vector<vector<cv::Mat>>& features)
{
    // branching factor and depth levels
    const int           k      = 10;
    const int           L      = 6;
    const WeightingType weight = TF_IDF;
    const ScoringType   score  = L1_NORM;

    OrbVocabulary voc(k, L, weight, score);

    cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Saving vocabulary..." << endl;
    voc.saveToBinaryFile("voc.bin");
    cout << "Done" << endl;
}
