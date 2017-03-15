//#############################################################################
//  File:      SLCVDetector.cpp
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

#include "slcvdetector.h"
#include <SLScene.h>
#include <SLSceneView.h>

using namespace cv;

SLCVDetector::SLCVDetector(SLCVDetectorType DetectorType, SLbool forced)
{
    this->forced = forced;
    this->type = DetectorType;
    switch(DetectorType){

        case DT_FAST:   _detector = FastFeatureDetector::create(); return;
        case DT_AGAST:  _detector = AgastFeatureDetector::create(); return;
        case DT_BRISK:  _detector = BRISK::create(); return;
        case DT_KAZE:   _detector = AKAZE::create(); return;
        case DT_ORB:    _detector = ORB::create(); return;
        case DT_SIFT:   _detector = xfeatures2d::SIFT::create(); return;
        case DT_SURF:   _detector = xfeatures2d::SURF::create(); return;
        break;
    }
}

void SLCVDetector::detect(InputArray image, std::vector<KeyPoint> &keypoints, InputArray mask){
    _detector->detect(image, keypoints, mask);
}
