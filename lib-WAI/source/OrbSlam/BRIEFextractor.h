/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef BRIEFEXTRACTOR_H
#define BRIEFEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include <WAIHelper.h>
#include <KPextractor.h>

namespace ORB_SLAM2
{

class WAI_API BRIEFextractor : public KPextractor
{
    public:
    enum
    {
        HARRIS_SCORE = 0,
        FAST_SCORE   = 1
    };

    BRIEFextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

    ~BRIEFextractor()
    {
    }

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()(cv::InputArray             image,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray            descriptors);
    void computeKeyPointDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    std::vector<cv::Mat> mvImagePyramid;

    protected:
    void                      ComputePyramid(cv::Mat image);
    void                      ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int& minX, const int& maxX, const int& minY, const int& maxY, const int& nFeatures, const int& level);

    void                   ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);
    std::vector<cv::Point> pattern;

    int iniThFAST;
    int minThFAST;
};

} //namespace ORB_SLAM

#endif
