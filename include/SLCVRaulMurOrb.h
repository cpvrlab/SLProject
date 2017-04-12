#ifndef SLCVRAULMURORB_H
#define SLCVRAULMURORB_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core.hpp>
#include <iostream>

class SLCVRaulMurOrb: public cv::Feature2D
{
public:
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    SLCVRaulMurOrb(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~SLCVRaulMurOrb(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.

    void detectAndCompute( cv::InputArray image, cv::InputArray mask,
                           std::vector<cv::KeyPoint>& keypoints,
                           cv::OutputArray descriptors, bool useProvidedKeypoints);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);
    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};


#endif // SLCVRAULMURORB_H
