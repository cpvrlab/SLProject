//#############################################################################
//  File:      SLCVDescriptor.cpp
//  Purpose:   OpenCV Detector Wrapper
//  Author:    Timon Tschanz
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/

#include "SLCVDescriptor.h"
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLCVRaulMurOrb.h>
using namespace cv;

SLCVDescriptor::SLCVDescriptor(SLCVDescriptorType descriptorType)
{
    type = descriptorType;
    switch(descriptorType){
    case DESC_BRIEF:    _descriptor = xfeatures2d::BriefDescriptorExtractor::create(32, true); return;
    case DESC_ORB:      _descriptor = ORB::create(200, 1.44f, 3, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30); return;
    case DESC_FREAK:    _descriptor = xfeatures2d::FREAK::create(true, true, 22.0f, 2); return;
    case DESC_KAZE:     _descriptor = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.01f, 2, 2 ,KAZE::DIFF_PM_G2); return;
    case DESC_BRISK:    _descriptor = BRISK::create(30, 2, 1.0f); return;
    case DESC_SIFT:     _descriptor = xfeatures2d::SiftDescriptorExtractor::create(300, 2, 0.04, 10, 1.6);return;
    case DESC_SURF:     _descriptor = xfeatures2d::SurfFeatureDetector::create(100, 2, 2, false, false);return;
    case DESC_RAUL:     _descriptor = new SLCVRaulMurOrb(1500, 1.44f, 4, 30, 20); return;
    default: break;
    }
}

void SLCVDescriptor::compute(InputArray image, std::vector<KeyPoint> &keypoints, OutputArray descriptors)
{
    _descriptor->compute(image, keypoints, descriptors);
}

void SLCVDescriptor::detectAndCompute(InputArray image, std::vector<KeyPoint> &keypoints, OutputArray descriptors,InputArray mask){
    _descriptor->detectAndCompute(image, mask, keypoints, descriptors);
}
