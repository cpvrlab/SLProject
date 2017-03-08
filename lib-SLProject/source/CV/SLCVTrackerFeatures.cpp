//#############################################################################
//  File:      SLCVTrackerAruco.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
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
#include <SLSceneView.h>
#include <SLCVTrackerFeatures.h>
#include <SLCVCapture.h>

using namespace cv;

//cv::Ptr<cv::FastFeatureDetector> detector;
cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> descriptor;
cv::Ptr<cv::ORB> detector;
//-----------------------------------------------------------------------------
SLCVTrackerFeatures::SLCVTrackerFeatures(SLNode* node) :
                  SLCVTracker(node)
{
//    detector = cv::FastFeatureDetector::create(30);
    descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    detector = cv::ORB::create(1000,1.44f,5,31,0,2,ORB::HARRIS_SCORE,31,20);
//    detector = cv::BRISK::create(50, 3, 1.0f);
//    detector = cv::FastFeatureDetector::create(30);
}
//-----------------------------------------------------------------------------
SLbool SLCVTrackerFeatures::track(SLCVMat imageGray,
                                  SLCVMat image,
                                  SLCVCalibration *calib,git git s
                                  SLSceneView *sv) {
    assert(!image.empty() && "Image is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");


    SLScene *scene = SLScene::current;


    // ORB feature extraction -------------------------------------------------
    SLCVVKeyPoint keypoints;
    cv::Mat descriptors;
    SLfloat detectTimeMillis = scene->timeMilliSec();
    detector->detect(imageGray, keypoints);
    scene->setDetectionTimesMS(scene->timeMilliSec()-detectTimeMillis);
    SLfloat startTimeMillis = scene->timeMilliSec();
    detector->compute(imageGray, keypoints, descriptors);
    scene->setFeatureTimesMS(scene->timeMilliSec()-startTimeMillis);
    cv::drawKeypoints(imageGray, keypoints, image, Scalar(0,0,255));

    // ------------------------------------------------------------------------

    return false;
}
//-----------------------------------------------------------------------------
