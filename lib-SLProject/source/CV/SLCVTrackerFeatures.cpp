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
//#define SAVE_SNAPSHOTS_OUTPUT "/tmp/cv_tracking/"
#define FLANN_BASED 0

// RANSAC configuration
const int iterations = 500;
const int reprojection_error = 2.0;
const double confidence = 0.95;

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
    cvtColor(planartracking, _lastFrameGray, CV_RGB2GRAY);

    _detector->detect(_lastFrameGray, _lastFrameKeypoints);
    _detector->compute(_lastFrameGray, _lastFrameKeypoints, _lastFrameDescriptors);

    // Calculate 3D-Points
    const SLfloat heightAboveObject = 8.0;
    for (unsigned int i = 0; i<_lastFrameKeypoints.size(); i++) {
        float x = _lastFrameKeypoints[i].pt.x;	// 2D location in image
        float y = _lastFrameKeypoints[i].pt.y;
        float X = (heightAboveObject / _fx)*(x - _cx);
        float Y = (heightAboveObject / _fy)*(y - _cy);
        float Z = 0;
        _points3d_model.push_back(cv::Point3f(X, Y, Z));
    }

    /*
          D                     C
            -------------------
            |                 |
            |                 |
            |       A4        |
            | (210mm × 297mm) |
            |                 |
            -------------------
          A                     B
    */

    _model.push_back(Point3f(0,     0, 0)); // A
    _model.push_back(Point3f(297,   0, 0)); // B
    _model.push_back(Point3f(297, 210, 0)); // C
    _model.push_back(Point3f(0,   210, 0)); // D
}

//------------------------------------------------------------------------------
inline void SLCVTrackerFeatures::initCameraMat(SLCVCalibration *calib) {
    _fx = calib->fx();
    _fy = calib->fy();

    _cx = calib->cx();
    _cy = calib->cy();

    _cam = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters
    _cam.at<double>(0, 0) = _fx;                  //  [ fx   0  cx ]
    _cam.at<double>(1, 1) = _fy;                  //  [  0  fy  cy ]
    _cam.at<double>(0, 2) = _cx;                  //  [  0   0   1 ]
    _cam.at<double>(1, 2) = _cy;
    _cam.at<double>(2, 2) = 1;

    _distortion = Mat::zeros(4, 1, CV_64F);      // Distortion parameters
    _rMatrix = cv::Mat::zeros(3, 3, CV_64FC1);   // rotation matrix
    _tMatrix = cv::Mat::zeros(3, 1, CV_64FC1);   // translation matrix
    _pMatrix = cv::Mat::zeros(3, 4, CV_64FC1);   // rotation-translation matrix

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
    if (_cam.empty()) initCameraMat(calib);

    SLCVVKeyPoint keypoints = extractFeatures(imageGray);
    Mat descriptors = describeFeatures(imageGray , keypoints);
    vector<DMatch> matches = matchFeatures(descriptors);

    if(matches.size() >= 4)  {
        vector<Point2f> inliers = trackFeatures(keypoints, matches);
        calcPMatrix();
        drawObject(image);
    }

    #if DEBUG
    //draw2DPoints(image, inliers, Scalar(0, 0, 255));

    Mat imgMatches;
    drawMatches(imageGray, keypoints, _lastFrameGray, _lastFrameKeypoints, matches, imgMatches);

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
inline SLCVVKeyPoint SLCVTrackerFeatures::extractFeatures(const Mat &imageGray) {
    SLCVVKeyPoint keypoints;
    SLfloat detectTimeMillis = SLScene::current->timeMilliSec();
    _detector->detect(imageGray, keypoints);
    SLScene::current->setDetectionTimesMS(SLScene::current->timeMilliSec() - detectTimeMillis);
    return keypoints;
}

//-----------------------------------------------------------------------------
inline Mat SLCVTrackerFeatures::describeFeatures(const Mat &imageGray, SLCVVKeyPoint &keypoints) {
    Mat descriptors;
    SLfloat computeTimeMillis = SLScene::current->timeMilliSec();
    _detector->compute(imageGray, keypoints, descriptors);
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
    _matcher->knnMatch(descriptors, _lastFrameDescriptors, matches, k);

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
inline vector<Point2f> SLCVTrackerFeatures::trackFeatures(const SLCVVKeyPoint &keypoints, const vector<DMatch> &matches) {
    vector<Point3f> points_model(matches.size());
    vector<Point2f> points_scene(matches.size());

    for (size_t i = 0; i < matches.size(); i++) {
        points_model[i] = _points3d_model[matches[i].trainIdx];
        points_scene[i] =       keypoints[matches[i].queryIdx].pt;
    }

    Mat inliersIndex, rvec;
    cv::solvePnPRansac(points_model,
                       points_scene,
                       _cam,
                       _distortion,
                       rvec, _tMatrix,
                       false,
                       iterations,
                       reprojection_error,
                       confidence,
                       inliersIndex,
                       cv::SOLVEPNP_ITERATIVE);

    cv::Rodrigues(rvec, _rMatrix);

    // Convert inliers from index matrix back to points
    vector<Point2f> inliers;
    for (int i = 0; i < inliersIndex.rows; i++) {
        int idx = inliersIndex.at<int>(i);
        inliers.push_back(points_scene[idx]);
    }

    #if DEBUG
    printf("We got %d inliers and %d matches overall \n", inliers.size(), matches.size());
    #endif

    return inliers;
}

//-----------------------------------------------------------------------------
inline void SLCVTrackerFeatures::draw2DPoints(Mat image, const vector<Point2f> &list_points, Scalar color) {
  for( size_t i = 0; i < list_points.size(); i++)
  {
    Point2f point_2d = list_points[i];

    // Draw Selected points
    circle(image, point_2d, 4, color, -1, 8);
  }
}

//-----------------------------------------------------------------------------
inline void SLCVTrackerFeatures::drawObject(const Mat &image)
{
   Point2f a = backproject3DPoint(_model[0]);
   Point2f b = backproject3DPoint(_model[1]);
   Point2f c = backproject3DPoint(_model[2]);
   Point2f d = backproject3DPoint(_model[3]);

    rectangle(image, a, c, cv::Scalar(0, 0, 255));
}

//-----------------------------------------------------------------------------
inline Point2f SLCVTrackerFeatures::backproject3DPoint(const Point3f &point3d)
{
  // 3D point vector [x y z 1]'
  cv::Mat point3d_vec = cv::Mat(4, 1, CV_64FC1);
  point3d_vec.at<double>(0) = point3d.x;
  point3d_vec.at<double>(1) = point3d.y;
  point3d_vec.at<double>(2) = point3d.z;
  point3d_vec.at<double>(3) = 1;

  // 2D point vector [u v 1]'
  cv::Mat point2d_vec = cv::Mat(4, 1, CV_64FC1);
  point2d_vec = _cam * _pMatrix * point3d_vec;

  // Normalization of [u v]'
  Point2f point2d;
  point2d.x = (float)(point2d_vec.at<double>(0) / point2d_vec.at<double>(2));
  point2d.y = (float)(point2d_vec.at<double>(1) / point2d_vec.at<double>(2));

  return point2d;
}

inline void SLCVTrackerFeatures::calcPMatrix()
{
    // Rotation-Translation Matrix Definition
    _pMatrix.at<double>(0,0) = _rMatrix.at<double>(0,0);
    _pMatrix.at<double>(0,1) = _rMatrix.at<double>(0,1);
    _pMatrix.at<double>(0,2) = _rMatrix.at<double>(0,2);
    _pMatrix.at<double>(1,0) = _rMatrix.at<double>(1,0);
    _pMatrix.at<double>(1,1) = _rMatrix.at<double>(1,1);
    _pMatrix.at<double>(1,2) = _rMatrix.at<double>(1,2);
    _pMatrix.at<double>(2,0) = _rMatrix.at<double>(2,0);
    _pMatrix.at<double>(2,1) = _rMatrix.at<double>(2,1);
    _pMatrix.at<double>(2,2) = _rMatrix.at<double>(2,2);
    _pMatrix.at<double>(0,3) = _tMatrix.at<double>(0);
    _pMatrix.at<double>(1,3) = _tMatrix.at<double>(1);
    _pMatrix.at<double>(2,3) = _tMatrix.at<double>(2);
}
