#ifndef GLSLEXTRACTOR
#define GLSLEXTRACTOR

#include <vector>
#include <list>
#include <KPextractor.h>
#include <WAIHelper.h>
#include <GLSLHessian.h>

class GLSLextractor : public ORB_SLAM2::KPextractor
{
public:
    GLSLextractor(int w, int h, int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma);

    ~GLSLextractor() {}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()(cv::InputArray             image,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray            descriptors);

protected:
    std::vector<cv::Point> pattern;
    cv::Mat                images[2];
    int                    idx;
    GLSLHessian            imgProc;
    cv::Ptr<cv::CLAHE>     _clahe;
};

#endif
