#include "bit_pattern.h"
#include "orb_descriptor.h"
#define _USE_MATH_DEFINES
#include <math.h>

#define DEG2RAD M_PI / 180

static void computeOrbDescriptor(const cv::KeyPoint& kpt,
                                 const cv::Mat&      img,
                                 const cv::Point*    pattern,
                                 Descriptor&         desc)
{
    desc.p      = desc.mem;
    float angle = kpt.angle * DEG2RAD;
    float a = (float)cos(angle), b = (float)sin(angle);
    //float a = (float)sin(angle), b = (float)cos(angle);
    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int    step   = (int)img.step;

    //    int x, y;
    //#define CHECK_VALUE(idx) \
//    x = cvRound(pattern[idx].x * b + pattern[idx].y * a); \
//    y = cvRound(pattern[idx].x * a - pattern[idx].y * b); \
//    if (x < -15 || x > 15 || y < -15 || y > 15) \
//    { \
//        std::cout << "ERROR: " << x << " ," << y << std::endl; \
//        std::cout << pattern[idx].x << std::endl; \
//        std::cout << pattern[idx].y << std::endl; \
//        std::cout << a << std::endl; \
//        std::cout << b << std::endl; \
//    }

#define GET_VALUE(idx) \
    center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + \
           cvRound(pattern[idx].x * a - pattern[idx].y * b)]

    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        // CHECK_VALUE(0);
        int t0, t1, val;
        t0  = GET_VALUE(0);
        t1  = GET_VALUE(1);
        val = t0 < t1;
        t0  = GET_VALUE(2);
        t1  = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc.p[i] = (uchar)val;
    }

#undef GET_VALUE
}

static void get_pattern(std::vector<cv::Point>& pattern)
{
    const int        npoints  = 512;
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
}

void ComputeORBDescriptors(std::vector<Descriptor>& descriptors, const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::Point>* pattern)
{
    if (descriptors.size() == 0)
    {
        descriptors.resize(keypoints.size());
    }

    if (pattern == NULL)
    {
        std::vector<cv::Point> p;
        get_pattern(p);
        for (size_t i = 0; i < keypoints.size(); i++)
            computeOrbDescriptor(keypoints[i], image, &p[0], descriptors[i]);
    }
    else
    {
        for (size_t i = 0; i < keypoints.size(); i++)
            computeOrbDescriptor(keypoints[i], image, &(*pattern)[0], descriptors[i]);
    }
}

void ComputeORBDescriptors(std::vector<std::vector<Descriptor>>& descriptors, std::vector<cv::Mat> image_pyramid, PyramidParameters& p, std::vector<std::vector<cv::KeyPoint>>& allKeypoints)
{
    int nlevels = p.scale_factors.size();
    descriptors.resize(nlevels);

    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        nkeypoints += (int)allKeypoints[level].size();
        descriptors[level].resize((int)allKeypoints[level].size());
    }

    if (nkeypoints == 0)
        return;

    std::vector<cv::Point> pattern;
    get_pattern(pattern);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        std::vector<cv::KeyPoint>& keypoints       = allKeypoints[level];
        int                        nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0)
            continue;

        // preprocess the resized image
        cv::Mat workingMat = image_pyramid[level].clone();
        GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        ComputeORBDescriptors(descriptors[level], workingMat, keypoints, &pattern);
    }
}
