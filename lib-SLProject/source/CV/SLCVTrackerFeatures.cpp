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

//-----------------------------------------------------------------------------
SLCVTrackerFeatures::SLCVTrackerFeatures(SLNode* node) :
                  SLCVTracker(node)
{}
//-----------------------------------------------------------------------------
//! Tracks the all ArUco markers in the given image for the first sceneview
/* The tracking of all aruco markers is done only once even if multiple aruco 
markers are used for different SLNode.
*/
SLbool SLCVTrackerFeatures::track(SLCVMat imageGray,
                                  SLCVMat image,
                                  SLCVCalibration* calib,
                                  SLSceneView* sv)
{
    assert(!image.empty() && "Image is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");

    SLScene *scene = SLScene::current;

    SLfloat startTimeMillis = scene->timeMilliSec();

    // ORB feature extraction -------------------------------------------------
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    SLCVVKeyPoint keypoints;
    detector->detect(image, keypoints);

    cv::Mat descriptors;
    detector->compute(image, keypoints, descriptors);

    cv::Mat rgb;
    cv::cvtColor(image, rgb, CV_BGR2RGB);
    cv::drawKeypoints(rgb, keypoints, rgb);
    cv::cvtColor(rgb, image, CV_RGB2BGR);
    // ------------------------------------------------------------------------

    scene->setFeatureTimesMS(scene->timeMilliSec()-startTimeMillis);

    return false;
}
//-----------------------------------------------------------------------------