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

#ifndef KPEXTRACTOR_H
#define KPEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>

namespace ORB_SLAM2
{

class KPextractor
{
    public:
    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    virtual void operator()(cv::InputArray             image,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::OutputArray            descriptors)          = 0;
    virtual void computeKeyPointDescriptors(const cv::Mat&             image,
                                            std::vector<cv::KeyPoint>& keypoints,
                                            cv::Mat&                   descriptors) = 0;

    int GetLevels()
    {
        return nlevels;
    }

    float GetScaleFactor()
    {
        return scaleFactor;
    }

    std::vector<float> GetScaleFactors()
    {
        return mvScaleFactor;
    }

    std::vector<float> GetInverseScaleFactors()
    {
        return mvInvScaleFactor;
    }

    std::vector<float> GetScaleSigmaSquares()
    {
        return mvLevelSigma2;
    }

    std::vector<float> GetInverseScaleSigmaSquares()
    {
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

    protected:
    int                nfeatures;
    double             scaleFactor;
    int                nlevels;
    std::vector<int>   mnFeaturesPerLevel;
    std::vector<int>   umax;
    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

} //namespace ORB_SLAM

#endif
