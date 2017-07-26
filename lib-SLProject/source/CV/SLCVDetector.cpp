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

#include "SLCVDetector.h"
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLCVRaulMurOrb.h>
using namespace cv;

SLCVDetector::SLCVDetector(SLCVDetectorType DetectorType, SLbool forced)
{
    this->forced = forced;
    this->type = DetectorType;
    switch(DetectorType){

        case DT_FAST:   _detector = FastFeatureDetector::create(30, true, FastFeatureDetector::TYPE_9_16); return;
        case DT_AGAST:  _detector = AgastFeatureDetector::create(30, true, AgastFeatureDetector::OAST_9_16); return;
        case DT_BRISK:  _detector = BRISK::create(30, 2, 1.0f); return;
        case DT_KAZE:   _detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 2, 2, KAZE::DIFF_PM_G2); return;
        case DT_ORB:    _detector = ORB::create(200, 1.44f, 3, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30); return;
        case DT_SIFT:   _detector = xfeatures2d::SIFT::create(300, 2, 0.04, 10, 1.6); return;
        case DT_SURF:   _detector = xfeatures2d::SURF::create(100, 2, 2, false, false); return;
        case DT_RAUL:   _detector = new SLCVRaulMurOrb(1500, 1.44f, 4, 30, 20); return;
        break;
    }
}

void SLCVDetector::detect(InputArray image, std::vector<KeyPoint> &keypoints, InputArray mask){
    _detector->detect(image, keypoints, mask);
}
