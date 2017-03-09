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
#include <time.h>
#include <sys/stat.h>

using namespace cv;

#define SAVE_SNAPSHOTS 1
#define SAVE_SNAPSHOTS_OUTPUT "/tmp/cv_tracking/"

#define FLANN_BASED 0

//-----------------------------------------------------------------------------
SLCVTrackerFeatures::SLCVTrackerFeatures(SLNode *node) :
        SLCVTracker(node) {
    _detector = ORB::create(
            /* int nfeatures */ 100,
            /* float scaleFactor */ 1.2f,
            /* int nlevels */ 8,
            /* int edgeThreshold */ 31,
            /* int firstLevel */ 0,
            /* int WTA_K */ 2,
            /* int scoreType */ ORB::HARRIS_SCORE,
            /* int patchSize */ 31,
            /* int fastThreshold */ 20);

#if FLANN_BASED
    _matcher = new FlannBasedMatcher();
#else
    _matcher =  BFMatcher::create(BFMatcher::BRUTEFORCE_HAMMING, false);
#endif
    //TODO: Works only for Unix/Linux
    mkdir(SAVE_SNAPSHOTS_OUTPUT, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

//------------------------------------------------------------------------------
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

    // ORB feature detection --------------------------------------------------
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
    // TODO: Don't do this if-statement for every call
    if (_lastFrameDescriptors.empty()) {
        _lastFrameKeypoints = keypoints;
        _lastFrameDescriptors = descriptors;
        _lastFrameGray = imageGray;
        return false;
    }

    SLfloat matchTimeMillis = scene->timeMilliSec();

#if FLANN_BASED
    if(descriptors.type() !=CV_32F ) descriptors.convertTo(descriptors, CV_32F);
    if(_lastFrameDescriptors.type() != CV_32F) _lastFrameDescriptors.convertTo(_lastFrameDescriptors, CV_32F);

    vector< DMatch > matches;
    _matcher->match(descriptors, _lastFrameDescriptors, matches);
#else
    vector<vector<DMatch>> matches;
    int k = 1; // Draws k lines for k-best feature matches
    _matcher->knnMatch(descriptors, _lastFrameDescriptors, matches, k);
#endif

    scene->setMatchTimesMS(scene->timeMilliSec() - matchTimeMillis);

    Mat imgMatches;
    drawMatches(imageGray, keypoints, _lastFrameGray, _lastFrameKeypoints, matches, imgMatches);

#if SAVE_SNAPSHOTS
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];

    time (&rawtime);
    timeinfo = localtime (&rawtime);

    strftime(buffer ,80, "%I%M%S", timeinfo);
    imwrite(SAVE_SNAPSHOTS_OUTPUT + string(buffer) + ".png", imgMatches);
#endif

    _lastFrameKeypoints = keypoints;
    _lastFrameDescriptors = descriptors;
    _lastFrameGray = imageGray;
    // ------------------------------------------------------------------------

    return false;
}
//-----------------------------------------------------------------------------
