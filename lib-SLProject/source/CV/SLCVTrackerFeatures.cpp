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
#define HOFF_EXAMPLE 0
//#define SAVE_SNAPSHOTS_OUTPUT "/tmp/cv_tracking/"

#define FLANN_BASED 0

// Matching configuration
const float minRatio = 0.75f;

// RANSAC configuration
const int iterations = 500;
const float reprojectionError = 5.0;
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
inline void SLCVTrackerFeatures::initCameraMat(SLCVCalibration *calib) {
    _fx = calib->fx();
    _fy = calib->fy();

    _cx = calib->cx();
    _cy = calib->cy();

    _intrinsics = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters
    _intrinsics.at<double>(0, 0) = _fx;             //  [ fx   0  cx ]
    _intrinsics.at<double>(1, 1) = _fy;             //  [  0  fy  cy ]
    _intrinsics.at<double>(0, 2) = _cx;             //  [  0   0   1 ]
    _intrinsics.at<double>(1, 2) = _cy;
    _intrinsics.at<double>(2, 2) = 1;

    _distortion = Mat::zeros(4, 1, CV_64F);         // Distortion parameters

    loadModelPoints();
}

//------------------------------------------------------------------------------
void SLCVTrackerFeatures::loadModelPoints() {
    Mat planartracking = imread("../_data/images/textures/planartracking.jpg");
    cvtColor(planartracking, _map.frameGray, CV_RGB2GRAY);
    SLScene *scene = SLScene::current;
    scene->_detector->detect(_map.frameGray, _map.keypoints);
    scene->_descriptor->compute(_map.frameGray, _map.keypoints, _map.descriptors);

    // Calculate 3D-Points based on the detected features
    // FIXME: Use correct 3D points!!!!! (Pointcloud)
    const SLfloat heightMM = 8.0;
    for (unsigned int i = 0; i< _map.keypoints.size(); i++) {
        float x = _map.keypoints[i].pt.x;	// 2D location in image
        float y = _map.keypoints[i].pt.y;   // 2D location in image
        float X = (heightMM / _fx) * (x - _cx);
        float Y = (heightMM / _fy) * (y - _cy);
        float Z = 0;
        _model.push_back(Point3f(X, Y, Z));
    }
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

    Mat rvec = cv::Mat::zeros(3, 3, CV_64FC1);      // rotation matrix
    Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);      // translation matrix
    bool foundPose = calculatePose(image, keypoints, matches, rvec, tvec);

    if (foundPose) {
        //_extrinsics = calculateExtrinsicMatrix(rvec, tvec);
        _extrinsics = createGLMatrix(tvec, rvec);

        // Update Scene Graph camera to display model correctly (positioning cam relative to world coordinates)
        sv->camera()->om(_extrinsics.inverse());

        //set node visible
        sv->camera()->setDrawBitsRec(SL_DB_HIDDEN, false);
    }

    // ####################################################################################################################

    #ifdef SAVE_SNAPSHOTS_OUTPUT
    Mat imgMatches;
    drawMatches(imageGray, keypoints, _map.frameGray, _map.keypoints, matches, imgMatches);
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];

    time (&rawtime);
    timeinfo = localtime (&rawtime);

    strftime(buffer ,80, "%I%M%S", timeinfo);
    imwrite(SAVE_SNAPSHOTS_OUTPUT + string(buffer) + ".png", imgMatches);
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

    vector<DMatch> goodMatches;
    for(size_t i = 0; i < matches.size(); i++) {
        const DMatch &match1 = matches[i][0];
        const DMatch &match2 = matches[i][1];
        float inverseRatio = match1.distance / match2.distance;
        if (inverseRatio < minRatio) goodMatches.push_back(match1);
    }
    #endif

    SLScene::current->setMatchTimesMS(SLScene::current->timeMilliSec() - matchTimeMillis);
    return goodMatches;
}

//-----------------------------------------------------------------------------
inline bool SLCVTrackerFeatures::calculatePose(const Mat &image, const SLCVVKeyPoint &keypoints, const vector<DMatch> &matches, Mat &rvec, Mat &tvec) {
    bool foundPose = 0;

    #if HOFF_EXAMPLE
    // For homography caluculations there must be at least 4 points
    if(matches.size() < 4) return 0;

    vector<Point2f> pts1(matches.size());	// Points from ref image
    vector<Point2f> pts2(matches.size());	// Points from new image
    for (size_t i = 0; i < matches.size(); i++) {
        pts1[i] = _map.keypoints[matches[i].trainIdx].pt;
        pts2[i] =      keypoints[matches[i].queryIdx].pt;
    }


    /* The following determines if there is a perspective transformation between
     * the two point features (2D).
     *
     * See http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findhomography
     *
     */
    vector<unsigned char> inliersMask(pts1.size());
    Mat homography = cv::findHomography(
        pts1, pts2,
        cv::FM_RANSAC,
        reprojectionError,    // Allowed reprojection error in pixels (default=3)
        inliersMask);

    /*
     * Convert inliers back to points. The inliersIndex matrix contais the frame location point
     */
    vector<DMatch> inliers;
    for (int i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i])
           inliers.push_back(matches[i]);
    }

    /* Find 2D/3D correspondences
     *
     *  At the moment we are using only the two correspondences like this:
     *  KeypointsOriginal <-> KeypointsActualscene
     *
     *  Train index --> "Point" in the model image
     *  Query index --> "Point" in the actual frame
     */
    vector<Point3f> modelPoints(matches.size());
    vector<Point2f> framePoints(matches.size());
    for (size_t i = 0; i < inliers.size(); i++) {
        modelPoints[i] =    _model[inliers[i].trainIdx];
        framePoints[i] = keypoints[inliers[i].queryIdx].pt;
    }

    if (inliers.size() < 5) return 0;

    foundPose = cv::solvePnP(
        modelPoints, framePoints,
        _intrinsics,                    // intrinsic camera parameter matrix
        cv::Mat::zeros(5, 1, CV_64F),	// distortion coefficients
        rvec, tvec);                    // output rotation and translation

    #else
    if (matches.size() == 0) return 0; // RANSAC crashes if 0 points are given

    /* Find 2D/3D correspondences
     *
     *  At the moment we are using only the two correspondences like this:
     *  KeypointsOriginal <-> KeypointsActualscene
     *
     *  Train index --> "Point" in the model image
     *  Query index --> "Point" in the actual frame
     */
    vector<Point3f> modelPoints(matches.size());
    vector<Point2f> framePoints(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        modelPoints[i] =    _model[matches[i].trainIdx];
        framePoints[i] = keypoints[matches[i].queryIdx].pt;
    }

    /* We execute first RANSAC to eliminate wrong feature correspondences (outliers) and only use
     * the correct ones (inliers) for PnP solving.
     *
     * RANSAC --------------------------
     * The RANdom Sample Consensus algorithm is called to remove "wrong" point correspondences
     *  which makes the solvePnP more robust. The so called inliers are used for calculation,
     *  wrong correspondences (outliers) will be ignored. Therefore the method below will first
     *  run a solvePnP with the EPNP method and returns the reprojection error.
     *
     * PnP ----------------------------- (https://en.wikipedia.org/wiki/Perspective-n-Point)
     * General problem: We have a calibrated cam and sets of corresponding 2D/3D points.
     *  We will calculate the rotation and translation in respect to world coordinates.
     *
     * Methods
     *
     * P3P: If we have 3 Points given, we have the minimal form of the PnP problem. We can
     *  treat the points as a triangle definition ABC. We have 3 corner points and 3 angles.
     *  Because we get many soulutions for the equation, there will be a fourth point which
     *  removes the ambiguity. Therefore the OpenCV implementation requires 4 points to use
     *  this method.
     *
     * EPNP: This method is used if there are n >= 4 points. The reference points are expressed
     *  as 4 virtual control points. The coordinates of these points are the unknowns for the
     *  equtation.
     *
     * ITERATIVE: Calculates pose using the DLT (Direct Linear Transform) method and
     *  makes a Levenberg-Marquardt optimization. The latter helps to decrease the reprojection
     *  error which describes how good the calculated POSE applies to the point sets.
     */
    vector<unsigned char> inliersMask(modelPoints.size());
    cv::solvePnPRansac(modelPoints,
                       framePoints,
                       _intrinsics,
                       _distortion,
                       rvec, tvec,
                       false,
                       iterations,
                       reprojectionError,
                       confidence,
                       inliersMask,
                       cv::SOLVEPNP_ITERATIVE);

    #endif

    #if DEBUG
    /*
     * Convert inliers back to points. The inliersIndex matrix contais the frame location point
     */
    vector<Point2f> inlierPoints;
    for (int i = 0; i < inliersMask.size(); i++) {
        int idx = inliersMask[i];
        inlierPoints.push_back(framePoints[idx]);
    }

    draw2DPoints(image, inlierPoints, Scalar(255, 0, 0));
    printf("Found pose: %d \n", foundPose);
    #endif

    return foundPose;
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
