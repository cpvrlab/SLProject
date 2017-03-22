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
//#define SAVE_SNAPSHOTS_OUTPUT "/tmp/cv_tracking/"

#define FLANN_BASED 0

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
    #if defined(unix)
    mkdir(SAVE_SNAPSHOTS_OUTPUT, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    #else
    mkdir(SAVE_SNAPSHOTS_OUTPUT);
    #endif
    #endif
}

//------------------------------------------------------------------------------
void SLCVTrackerFeatures::load2dReferenceFeatures() {
    Mat planartracking = imread("../_data/images/textures/planartracking.jpg");
    cvtColor(planartracking, _map.frameGray, CV_RGB2GRAY);
    SLScene *scene = SLScene::current;
    scene->_detector->detect(_map.frameGray, _map.keypoints);
    scene->_descriptor->compute(_map.frameGray, _map.keypoints, _map.descriptors);

    // Calculate 3D-Points
    const SLfloat lengthMM = 8.0;
    for (unsigned int i = 0; i< _map.keypoints.size(); i++) {
        float x = _map.keypoints[i].pt.x;	// 2D location in image
        float y = _map.keypoints[i].pt.y;
        float X = (lengthMM / _fx) * (x - _cx);
        float Y = (lengthMM / _fy) * (y - _cy);
        float Z = 0;
        _model.push_back(Point3f(X, Y, Z));
    }
}

//------------------------------------------------------------------------------
inline void SLCVTrackerFeatures::initCameraMat(SLCVCalibration *calib) {
    _fx = calib->fx();
    _fy = calib->fy();

    _cx = calib->cx();
    _cy = calib->cy();

    _intrinsics = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters
    _intrinsics.at<double>(0, 0) = _fx;                  //  [ fx   0  cx ]
    _intrinsics.at<double>(1, 1) = _fy;                  //  [  0  fy  cy ]
    _intrinsics.at<double>(0, 2) = _cx;                  //  [  0   0   1 ]
    _intrinsics.at<double>(1, 2) = _cy;
    _intrinsics.at<double>(2, 2) = 1;

    _distortion = Mat::zeros(4, 1, CV_64F);         // Distortion parameters

    load2dReferenceFeatures();
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

    // TODO: Really necessary to put this check here (instead of constructor)?
    if (_intrinsics.empty()) initCameraMat(calib);


    //  Main part: Detect, describe, match and track features #############################################################
    SLCVVKeyPoint keypoints = detectFeatures(imageGray);
    Mat descriptors = describeFeatures(imageGray , keypoints);
    vector<DMatch> matches = matchFeatures(descriptors);

    if(matches.size() >= 4)  { // RANSAC crashes if there are 0 points and we need at least 4 points to determine planarity
        Mat rvec = cv::Mat::zeros(3, 3, CV_64FC1);      // rotation matrix
        Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);      // translation matrix
        vector<Point2f> inliers = calculatePose(keypoints, matches, rvec, tvec);
        //_extrinsics = calculateExtrinsicMatrix(rvec, tvec);
        _extrinsics = createGLMatrix(tvec, rvec);

        // Update Scene Graph camera to display model correctly (positioning cam relative to world coordinates)
        sv->camera()->om(_extrinsics.inverse());

        //set node visible
        sv->camera()->setDrawBitsRec(SL_DB_HIDDEN, false);

        SLMat4f omTower = sv->camera()->om() * _extrinsics;
        _tower->om(omTower);
        sv->camera()->setDrawBitsRec(SL_DB_HIDDEN, false);
    }

    // ####################################################################################################################
    //drawObject(image);

    #if DEBUG
    //draw2DPoints(image, inliers, Scalar(0, 0, 255));

    Mat imgMatches;
    drawMatches(imageGray, keypoints, _map.frameGray, _map.keypoints, matches, imgMatches);

    #ifdef SAVE_SNAPSHOTS_OUTPUT
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];

    time (&rawtime);
    timeinfo = localtime (&rawtime);

    strftime(buffer ,80, "%I%M%S", timeinfo);
    imwrite(SAVE_SNAPSHOTS_OUTPUT + string(buffer) + ".png", imgMatches);
    #endif
    #endif

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

    float ratio = 0.7f;
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

//-----------------------------------------------------------------------------
inline vector<Point2f> SLCVTrackerFeatures::calculatePose(const SLCVVKeyPoint &keypoints, const vector<DMatch> &matches, Mat &rvec, Mat &tvec) {
    vector<Point3f> points_model(matches.size());
    vector<Point2f> points_scene(matches.size());

    for (size_t i = 0; i < matches.size(); i++) {
        points_model[i] =    _model[matches[i].trainIdx];
        points_scene[i] = keypoints[matches[i].queryIdx].pt;
    }

    Mat inliersIndex;
    cv::solvePnPRansac(points_model,
                       points_scene,
                       _intrinsics,
                       _distortion,
                       rvec, tvec,
                       false,
                       iterations,
                       reprojection_error,
                       confidence,
                       inliersIndex,
                       cv::SOLVEPNP_ITERATIVE);

    // Convert inliers from index matrix back to points
    vector<Point2f> inliers;
    for (int i = 0; i < inliersIndex.rows; i++) {
        int idx = inliersIndex.at<int>(i);
        inliers.push_back(points_scene[idx]);
    }

    #if DEBUG
    printf("We got %d inliers and %d matches overall \n", inliers.size(), matches.size());
    #endif

    //TODO: Return necessery?
    return inliers;
}

//-----------------------------------------------------------------------------
inline void SLCVTrackerFeatures::draw2DPoints(Mat image, const vector<Point2f> &list_points, Scalar color) {
    for( size_t i = 0; i < list_points.size(); i++) {
        Point2f point_2d = list_points[i];

        // Draw Selected points
        circle(image, point_2d, 4, color, -1, 8);
    }
}

inline Mat SLCVTrackerFeatures::calculateExtrinsicMatrix(Mat &rvec, Mat &tvec) {
    /*
        Rotation-Translation Matrix Definition

        [ r11 r12 r13 t1
          r21 r22 r23 t2
          r31 r32 r33 t3 ]

    */
    Mat extrinsics = Mat::zeros(3, 4, CV_64FC1);
    extrinsics.at<double>(0,0) = rvec.at<double>(0,0);
    extrinsics.at<double>(0,1) = rvec.at<double>(0,1);
    extrinsics.at<double>(0,2) = rvec.at<double>(0,2);
    extrinsics.at<double>(1,0) = rvec.at<double>(1,0);
    extrinsics.at<double>(1,1) = rvec.at<double>(1,1);
    extrinsics.at<double>(1,2) = rvec.at<double>(1,2);
    extrinsics.at<double>(2,0) = rvec.at<double>(2,0);
    extrinsics.at<double>(2,1) = rvec.at<double>(2,1);
    extrinsics.at<double>(2,2) = rvec.at<double>(2,2);
    extrinsics.at<double>(0,3) = tvec.at<double>(0);
    extrinsics.at<double>(1,3) = tvec.at<double>(1);
    extrinsics.at<double>(2,3) = tvec.at<double>(2);

    return extrinsics;
}
