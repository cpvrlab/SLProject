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
SLCVTrackerFeatures::SLCVTrackerFeatures(SLNode *node) :
        SLCVTracker(node) {
    _detector = ORB::create(/* int nfeatures */ 80,
            /* float scaleFactor */ 1,
            /* int nlevels */ 1,
            /* int edgeThreshold */ 31,
            /* int firstLevel */ 0,
            /* int WTA_K */ 2,
            /* int scoreType */ ORB::HARRIS_SCORE,
            /* int patchSize */ 31,
            /* int fastThreshold */ 20);

    _matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //_lastFrameDescriptors =
}

//-----------------------------------------------------------------------------
SLbool SLCVTrackerFeatures::track(SLCVMat imageGray,
                                  SLCVMat image,
                                  SLCVCalibration *calib,
                                  SLSceneView *sv) {
    assert(!image.empty() && "Image is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");

    SLScene *scene = SLScene::current;

    // ORB feature detection -------------------------------------------------
    SLCVVKeyPoint keypoints;
    Mat descriptors;

    SLfloat detectTimeMillis = scene->timeMilliSec();
    _detector->detect(imageGray, keypoints);
    scene->setDetectionTimesMS(scene->timeMilliSec() - detectTimeMillis);
    // ------------------------------------------------------------------------

    // ORB feature descriptor extraction --------------------------------------
    SLfloat computeTimeMillis = scene->timeMilliSec();
    _detector->compute(imageGray, keypoints, descriptors);
    scene->setFeatureTimesMS(scene->timeMilliSec() - computeTimeMillis);
    // ------------------------------------------------------------------------
    drawKeypoints(imageGray, keypoints, image, Scalar(0, 0, 255));

    // Matching ---------------------------------------------------------------
    vector<vector<DMatch>> matches;
    SLfloat matchTimeMillis = scene->timeMilliSec();
    //_matcher->knnMatch(descriptors, _lastFrameDescriptors, matches, 3);
    scene->setMatchTimesMS(scene->timeMilliSec() - matchTimeMillis);
    //_lastFrameDescriptors = descriptors;
    // ------------------------------------------------------------------------
    scene->setFeatureTimesMS(scene->timeMilliSec() - computeTimeMillis);

    return false;
}
//-----------------------------------------------------------------------------
