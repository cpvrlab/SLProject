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
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace cv;

#define DEBUG 1
#define SAVE_SNAPSHOTS_OUTPUT "/tmp/cv_tracking/"

#define FLANN_BASED 0

//-----------------------------------------------------------------------------
SLCVTrackerFeatures::SLCVTrackerFeatures(SLNode *node) :
        SLCVTracker(node) {
    _detector = ORB::create(
            /* int nfeatures */ 500,
            /* float scaleFactor */ 1.2f,
            /* int nlevels */ 8,
            /* int edgeThreshold */ 31,
            /* int firstLevel */ 0,
            /* int WTA_K */ 2,
            /* int scoreType */ ORB::FAST_SCORE,
            /* int patchSize */ 31,
            /* int fastThreshold */ 20);

#if FLANN_BASED
    _matcher = new FlannBasedMatcher();
#else
    _matcher =  BFMatcher::create(BFMatcher::BRUTEFORCE_HAMMING, false);
#endif

#if SAVE_SNAPSHOTS
    #if defined(unix)
        mkdir(SAVE_SNAPSHOTS_OUTPUT, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    #else
        mkdir(SAVE_SNAPSHOTS_OUTPUT);
    #endif
#endif

        load2dReferenceFeatures();

        // Focal length
        fx = 530;
        fy = 530;

        // Principal point
        cx = 320;
        cy = 240;
}

//------------------------------------------------------------------------------
void SLCVTrackerFeatures::load2dReferenceFeatures() {
    Mat planartracking = imread("../_data/images/textures/planartracking.jpg");
    cvtColor(planartracking, _lastFrameGray, CV_RGB2GRAY);

    _detector->detect(_lastFrameGray, _lastFrameKeypoints);
    _detector->compute(_lastFrameGray, _lastFrameKeypoints, _lastFrameDescriptors);

    // Calculate 3D-Points
    const double heightAboveObject = 8.0;
    for (unsigned int i = 0; i<_lastFrameKeypoints.size(); i++) {
        float x = _lastFrameKeypoints[i].pt.x;	// 2D location in image
        float y = _lastFrameKeypoints[i].pt.y;
        float X = (heightAboveObject / fx)*(x - cx);
        float Y = (heightAboveObject / fy)*(y - cy);
        float Z = 0;
        _points3d_model.push_back(cv::Point3f(X, Y, Z));
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

    SLCVVKeyPoint keypoints = extractFeatures(imageGray);
    Mat descriptors = describeFeatures(imageGray, keypoints);
    vector<DMatch> matches = matchFeatures(descriptors);
    if(matches.size() < 4) return false;
    vector<Point2f> inliers = trackFeatures(keypoints, matches);

    #if DEBUG
    draw2DPoints(image, inliers, Scalar(0, 0, 255));

    Mat imgMatches;
    drawMatches(imageGray, keypoints, _lastFrameGray, _lastFrameKeypoints, matches, imgMatches);

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
inline SLCVVKeyPoint SLCVTrackerFeatures::extractFeatures(Mat& imageGray) {
    SLCVVKeyPoint keypoints;
    SLfloat detectTimeMillis = SLScene::current->timeMilliSec();
    _detector->detect(imageGray, keypoints);
    SLScene::current->setDetectionTimesMS(SLScene::current->timeMilliSec() - detectTimeMillis);
    return keypoints;
}

//-----------------------------------------------------------------------------
inline Mat SLCVTrackerFeatures::describeFeatures(Mat& imageGray, SLCVVKeyPoint& keypoints) {
    Mat descriptors;
    SLfloat computeTimeMillis = SLScene::current->timeMilliSec();
    _detector->compute(imageGray, keypoints, descriptors);
    SLScene::current->setFeatureTimesMS(SLScene::current->timeMilliSec() - computeTimeMillis);
    return descriptors;
}

//-----------------------------------------------------------------------------
inline vector<DMatch> SLCVTrackerFeatures::matchFeatures(Mat& descriptors) {
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
    _matcher->knnMatch(descriptors, _lastFrameDescriptors, matches, k);

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

//-----------------------------------------------------------------------------
inline vector<Point2f> SLCVTrackerFeatures::trackFeatures(SLCVVKeyPoint& keypoints, vector<DMatch>& matches) {
    vector<Point3f> points_model(matches.size());
    vector<Point2f> points_scene(matches.size());

    for (size_t i = 0; i < matches.size(); i++) {
        points_model[i] = _points3d_model[matches[i].trainIdx];
        points_scene[i] =       keypoints[matches[i].queryIdx].pt;
    }


    Mat cam = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters
    cam.at<double>(0, 0) = fx;                  //      [ fx   0  cx ]
    cam.at<double>(1, 1) = fy;                  //      [  0  fy  cy ]
    cam.at<double>(0, 2) = cx;                  //      [  0   0   1 ]
    cam.at<double>(1, 2) = cy;
    cam.at<double>(2, 2) = 1;

    Mat distortion = Mat::zeros(4, 1, CV_64F);


    int iterations = 50;
    int reprojection_error = 2.0;
    double confidence = 0.95;

    Mat rvec = Mat::zeros(3, 1, CV_64FC1);          // output rotation vector
    Mat tvec = Mat::zeros(3, 1, CV_64FC1);          // output translation

    Mat inliersIndex;
    cv::solvePnPRansac(points_model,
                       points_scene,
                       cam,
                       distortion,
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
        inliers.push_back(points_scene[inliersIndex.at<int>(i)]);
    }

    return inliers;
}

//-----------------------------------------------------------------------------
void SLCVTrackerFeatures::draw2DPoints(Mat image, vector<Point2f> &list_points, Scalar color) {
  for( size_t i = 0; i < list_points.size(); i++)
  {
    Point2f point_2d = list_points[i];

    // Draw Selected points
    circle(image, point_2d, 4, color, -1, 8);
  }
}

void SLCVTrackerFeatures::drawPose(cv::Mat rotVec, cv::Mat transVec, cv::Mat K, cv::Mat dist, cv::Mat image) {
    printf("Draw pose...");
}




