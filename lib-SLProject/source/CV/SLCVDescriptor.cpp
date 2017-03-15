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

using namespace cv;

SLCVDescriptor::SLCVDescriptor(SLCVDescriptorType descriptorType)
{
    type = descriptorType;
    switch(descriptorType){
    case DESC_BRIEF:    _descriptor = xfeatures2d::BriefDescriptorExtractor::create(); return;
    case DESC_ORB:      _descriptor = ORB::create(); return;
    case DESC_FREAK:    _descriptor = xfeatures2d::FREAK::create(); return;
    case DESC_KAZE:     _descriptor = AKAZE::create(); return;
    case DESC_BRISK:    _descriptor = BRISK::create(); return;
    case DESC_SIFT:     _descriptor = xfeatures2d::SiftDescriptorExtractor::create();return;
    case DESC_SURF:     _descriptor = xfeatures2d::SurfFeatureDetector::create();return;
    default: break;
    }
}

void SLCVDescriptor::compute(InputArray image, std::vector<KeyPoint> &keypoints, OutputArray descriptors)
{
    _descriptor->compute(image, keypoints, descriptors);
}
