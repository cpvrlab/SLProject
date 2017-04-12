#ifndef SLCVDESCRIPTOR_H
#define SLCVDESCRIPTOR_H

#include <SLCV.h>
#include <opencv2/xfeatures2d.hpp>

class SLCVDescriptor
{
private:
    cv::Ptr<cv::DescriptorExtractor> _descriptor;
public:
    SLCVDescriptor(SLCVDescriptorType type);
    void compute(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors);
    void detectAndCompute(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors, cv::InputArray mask=cv::noArray());

    SLCVDescriptorType type;

    void setDescriptor(cv::Ptr<cv::DescriptorExtractor> descriptor) { _descriptor = descriptor; }
};

#endif // SLCVDESCRIPTOR_H
