/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*/

#if !defined(TARGET_OS_IOS) or (TARGET_OS_IOS == 0)
#    include <vector>
#    include <AverageTiming.h>
#    include <GLSLextractor.h>
#    include <iostream>
#    include <BRIEFPattern.h>

using namespace cv;
using namespace std;

const int PATCH_SIZE      = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD  = 19;

static void computeBRIEFDescriptor(const KeyPoint&  kpt,
                                   const Mat&       img,
                                   const cv::Point* pattern,
                                   uchar*           desc)
{
    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int    step   = (int)img.step;

#    define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x) * step + cvRound(pattern[idx].y)]

    for (int i = 0; i < 32; ++i, pattern += 16)
    {
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

        desc[i] = (uchar)val;
    }

#    undef GET_VALUE
}

GLSLextractor::GLSLextractor(int w, int h, int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma)
  : KPextractor("GLSL", true),
    imgProc(w, h, nbKeypointsBigSigma, nbKeypointsSmallSigma, highThrs, lowThrs, bigSigma, smallSigma)
{
    mvScaleFactor.resize(1);
    mvLevelSigma2.resize(1);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;

    mvInvScaleFactor.resize(1);
    mvInvLevelSigma2.resize(1);
    mvInvScaleFactor[0] = 1.0f;
    mvInvLevelSigma2[0] = 1.0f;
    nlevels             = 1;
    scaleFactor         = 1.0;

    idx       = 0;
    images[1] = cv::Mat(h, w, CV_8UC1);

    const int        npoints  = 512;
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

/*
    _clahe = cv::createCLAHE();
    _clahe->setClipLimit(2.0);
    _clahe->setTilesGridSize(cv::Size(8, 8));
    */
}

static void computeDescriptors(const Mat&               image,
                               vector<KeyPoint>&        keypoints,
                               Mat&                     descriptors,
                               const vector<cv::Point>& pattern)
{
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        computeBRIEFDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
    }
}

void GLSLextractor::operator()(InputArray        _image,
                               vector<KeyPoint>& _keypoints,
                               OutputArray       _descriptors)
{
    images[idx] = _image.getMat().clone();

    if (_clahe)
    {
        //cv::imwrite("claheBefore.png", images[idx]);
        AVERAGE_TIMING_START("clahe");
        _clahe->apply(images[idx], images[idx]);
        AVERAGE_TIMING_STOP("clahe");
        //std::cout << "clahe time: " << AverageTiming::getTime("clahe") << std::endl;

        //cv::imwrite("claheAfter.png", images[idx]);
    }

    Mat m;
    Mat descriptors;

    AVERAGE_TIMING_START("GLSL Hessian");

    _keypoints.clear();

    imgProc.setInputTexture(images[idx]);
    imgProc.gpu_kp();
    imgProc.readResult(_keypoints);

    if (_keypoints.size() == 0)
    {
        _descriptors.release();
        return;
    }
    else
    {
        _descriptors.create(_keypoints.size(), 32, CV_8U);
        descriptors = _descriptors.getMat();
    }

    idx = (idx + 1) % 2;
    Mat workingMat;
    GaussianBlur(images[idx], workingMat, cv::Size(7, 7), 2, 2, BORDER_REFLECT_101);

    // Compute the descriptors
    Mat desc = descriptors.rowRange(0, _keypoints.size());
    computeDescriptors(workingMat, _keypoints, desc, pattern);

    AVERAGE_TIMING_STOP("GLSL Hessian");
}

#endif // TARGET_OS_IOS
