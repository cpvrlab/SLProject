//#############################################################################
//  File:      SLCVTrackerAruco.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
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
#include <SLCVRaulMurOrb.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace cv;

#define DEBUG 1
#define FLANN_BASED 0
#define SAVE_SNAPSHOTS_OUTPUT "/tmp/cv_tracking/"

// RANSAC configuration
const int iterations = 500;
const int reprojection_error = 2.0;
const double confidence = 0.85;

//-----------------------------------------------------------------------------
SLCVTrackerFeatures::SLCVTrackerFeatures(SLNode *node) :
        SLCVTracker(node) {

    #if FLANN_BASED
    _matcher = new FlannBasedMatcher();
    #else
    _matcher =  BFMatcher::create(BFMatcher::BRUTEFORCE_HAMMING, false);
    #endif

    #ifdef SAVE_SNAPSHOTS_OUTPUT
    #if defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS)
    mkdir(SAVE_SNAPSHOTS_OUTPUT, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    #elif defined(SL_OS_WINDOWS)
    mkdir(SAVE_SNAPSHOTS_OUTPUT);
    #endif
    #endif

    _frameCount = 0;
}

//------------------------------------------------------------------------------
void SLCVTrackerFeatures::loadModelPoints() {
    SLGLTexture* trackerTexture = new SLGLTexture("planartracking.jpg");
    SLCVImage* img = trackerTexture->images()[0];
    cvtColor(img->cvMat(), _map.frameGray, CV_RGB2GRAY);

    // Detect and compute features in marker image
     SLScene::current->_detector->detect(_map.frameGray, _map.keypoints);
     SLScene::current->_descriptor->compute(_map.frameGray, _map.keypoints, _map.descriptors);
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

    // Read reference marker ##############################################################################################
    if (_frameCount == 0) loadModelPoints();

    //  Main part: Detect, describe, match and track features #############################################################
    SLCVVKeyPoint keypoints = detectFeatures(imageGray);
    Mat descriptors = describeFeatures(imageGray , keypoints);
    vector<DMatch> matches = matchFeatures(descriptors);
    // ####################################################################################################################

    #ifdef SAVE_SNAPSHOTS_OUTPUT
    Mat imgKeypoints;
    drawKeypoints(imageGray, keypoints, imgKeypoints);
    imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(_frameCount) + "-keypoints.png", imgKeypoints);

    Mat imgMatches;
    drawMatches(imageGray, keypoints, _map.frameGray, _map.keypoints, matches, imgMatches);
    imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(_frameCount) + "-matches.png", imgMatches);
    #endif

    _frameCount++;
    return false;
}

//-----------------------------------------------------------------------------
inline SLCVVKeyPoint SLCVTrackerFeatures::detectFeatures(const Mat &imageGray) {
    SLCVVKeyPoint keypoints;
    SLfloat detectTimeMillis = SLScene::current->timeMilliSec();
    SLScene *scene = SLScene::current;
    scene->_detector->detect(imageGray, keypoints);
    SLScene::current->setDetectionTimesMS(SLScene::current->timeMilliSec() - detectTimeMillis);
    return keypoints;
}

//-----------------------------------------------------------------------------
inline Mat SLCVTrackerFeatures::describeFeatures(const Mat &imageGray, SLCVVKeyPoint &keypoints) {
    Mat descriptors;
    SLfloat computeTimeMillis = SLScene::current->timeMilliSec();
    SLScene *scene = SLScene::current;
    scene->_descriptor->compute(imageGray, keypoints, descriptors);
    SLScene::current->setFeatureTimesMS(SLScene::current->timeMilliSec() - computeTimeMillis);
    return descriptors;
}

//-----------------------------------------------------------------------------
inline vector<DMatch> SLCVTrackerFeatures::matchFeatures(const Mat &descriptors) {
    SLfloat matchTimeMillis = SLScene::current->timeMilliSec();

    // 1. Get matches with FLANN or KNN algorithm ######################################################################################
    #if FLANN_BASED
    if(descriptors.type() !=CV_32F ) descriptors.convertTo(descriptors, CV_32F);
    if(_lastFrameDescriptors.type() != CV_32F) _lastFrameDescriptors.convertTo(_lastFrameDescriptors, CV_32F);

    vector< DMatch > matches;
    _matcher->match(descriptors, _lastFrameDescriptors, matches);
    #else
    int k = 2;
    vector<vector<DMatch>> matches;
    _matcher->knnMatch(descriptors, _map.descriptors, matches, k);

    float ratio = 0.8f;
    vector<DMatch> good_matches;
    for(size_t i = 0; i < matches.size(); i++) {
        const DMatch &match1 = matches[i][0];
        const DMatch& match2 = matches[i][1];
        float inverse_ratio = match1.distance / match2.distance;
        if (inverse_ratio < ratio) good_matches.push_back(match1);
    }
    #endif

    SLScene::current->setMatchTimesMS(SLScene::current->timeMilliSec() - matchTimeMillis);
    return good_matches;
}

