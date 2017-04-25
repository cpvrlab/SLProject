//#############################################################################
//  File:      SLCVTrackerFeatures.cpp
//  Author:    Pascal Zingg, Timon Tschanz
//  Date:      Spring 2017
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
#include <opencv2/tracking.hpp>

#if defined(SL_OS_WINDOWS)
#include <direct.h>
#endif

using namespace cv;

#define DEBUG_OUTPUT 0
#define FORCE_REPOSE 1
#define DISTINGUISH_FEATURE_DETECT_COMPUTE 0

// Settings for drawing things into current camera frame
#define DRAW_KEYPOINTS 0
#define DRAW_REPROJECTION 1
#define DRAW_REPOSE_INFO 1

#ifdef SL_SAVE_DEBUG_OUTPUT
    #if defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS)
    #define SAVE_SNAPSHOTS_OUTPUT "/tmp/cv_tracking/"
    #elif defined(SL_OS_WINDOWS)
    #define SAVE_SNAPSHOTS_OUTPUT "cv_tracking/"
    #endif
#endif
// Feature detection and extraction
const int nFeatures = 2000;
const float minRatio = 0.7f;

// RANSAC parameters
const int iterations = 500;
const float reprojectionError = 2.0f;
const double confidence = 0.95;

// Repose patch size (TODO: Adjust automatically)
const int patchSize = 20;
const int patchHalf = patchSize / 2;
const int reposeFrequency = 10;

// Benchmarking
#define TRACKING_MEASUREMENT 1
#if TRACKING_MEASUREMENT
float low_detection_milis = 1000.0f;
float sum_detection_millis;
float high_detection_milis;

float low_compute_milis = 1000.0f;
float sum_compute_millis;
float high_compute_milis;

float low_detectcompute_milis = 1000.0f;
float sum_detectcompute_millis;
float high_detectcompute_milis;

float sum_allmatches_to_inliers = 0.0f;
double sum_reprojection_error = 0.0f;
float sum_poseopt_difference = 0.0f;
#endif

//-----------------------------------------------------------------------------
SLCVTrackerFeatures::SLCVTrackerFeatures(SLNode *node) :
        SLCVTracker(node) {
    SLCVRaulMurOrb* orbSlamMatcherAndDescriptor = new SLCVRaulMurOrb(nFeatures, 1.44f, 6, 20, 10);
    SLScene::current->_detector->setDetector(orbSlamMatcherAndDescriptor);
    SLScene::current->_descriptor->setDescriptor(orbSlamMatcherAndDescriptor);

    _matcher =  BFMatcher::create(BFMatcher::BRUTEFORCE_HAMMING, false);

#ifdef SAVE_SNAPSHOTS_OUTPUT
#if defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS)
    mkdir(SAVE_SNAPSHOTS_OUTPUT, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#elif defined(SL_OS_WINDOWS)
    _mkdir(SAVE_SNAPSHOTS_OUTPUT);
#else
#undef SAVE_SNAPSHOTS_OUTPUT
#endif
#endif

    frameCount = 0;
    _prev.points2D = SLCVVPoint2f(nFeatures);
    _prev.foundPose = false;
}

SLCVTrackerFeatures::~SLCVTrackerFeatures() {
#if TRACKING_MEASUREMENT
    int firstColWidth = 40;
#if DISTINGUISH_FEATURE_DETECT_COMPUTE
    cout << endl << endl << "Detection: -------------------------------------------------------" << endl;
    cout << setw(firstColWidth) << "Min detection Time (ms): " << low_detection_milis << endl;
    cout << setw(firstColWidth) << "Avg detection Time (ms): " << sum_detection_millis / frameCount << endl;
    cout << setw(firstColWidth) << "High detection Time (ms): " << high_detection_milis << endl << endl;

    cout << "Extraction: ------------------------------------------------------" << endl;
    cout << setw(firstColWidth) << "Min compute Time (ms): " << low_compute_milis << endl;
    cout << setw(firstColWidth) << "Avg compute Time (ms): " << sum_compute_millis / frameCount << endl;
    cout << setw(firstColWidth) << "High compute Time (ms): " << high_compute_milis << endl << endl;
#else
    cout << endl << endl << "Detect and compute: -------------------------------------------------------" << endl;
    cout << setw(firstColWidth) << "Min detect & compute Time (ms): " << low_detectcompute_milis << endl;
    cout << setw(firstColWidth) << "Avg detect & compute Time (ms): " << sum_detectcompute_millis / frameCount << endl;
    cout << setw(firstColWidth) << "High detect & compute Time (ms): " << high_detectcompute_milis << endl;
#endif
    cout << "POSE calculation: -------------------------------------------------------" << endl;
    cout << setw(firstColWidth) << "Avg allmatches to inliers proposition: " << sum_allmatches_to_inliers / frameCount << endl ;
    cout << setw(firstColWidth) << "Avg reprojection error: " << sum_reprojection_error / frameCount << endl;
    cout << setw(firstColWidth) << "Avg match boost factor: " << sum_poseopt_difference / frameCount << endl;

    cout << endl;
#endif
}

//------------------------------------------------------------------------------
void SLCVTrackerFeatures::loadModelPoints()
{
    // Read reference marker
    SLGLTexture* trackerTexture = new SLGLTexture(std::string(SL_TRACKER_IMAGE_NAME) + std::string(".png"));
    SLCVImage* img = trackerTexture->images()[0];
    cvtColor(img->cvMat(), _map.frameGray, CV_RGB2GRAY);
    cv::rotate(_map.frameGray, _map.frameGray, ROTATE_180);
    cv::flip(_map.frameGray, _map.frameGray, 1);

    // Detect and compute features in marker image
     SLScene::current->_descriptor->detectAndCompute(_map.frameGray, _map.keypoints, _map.descriptors);
    // Calculates proprtion of MM and Pixel (sample measuring)
    const SLfloat lengthMM = 297.0;
    const SLfloat lengthPX = img->width();
    float pixelPerMM = lengthPX / lengthMM;

    // Calculate 3D-Points based on the detected features
    for (unsigned int i = 0; i< _map.keypoints.size(); i++) {
        Point2f refImageKeypoint = _map.keypoints[i].pt; // 2D location in image
        refImageKeypoint /= pixelPerMM;                  // Point scaling
        float Z = 0;                                     // Here we can use 0 because we expect a planar object
        _map.model.push_back(Point3f(refImageKeypoint.x, refImageKeypoint.y, Z));
    }

    /*
     * Draw the projected points and keypoints into the MODEL
     */
    _map.frameGray.copyTo(_map.imgDrawing);
    cvtColor(_map.imgDrawing, _map.imgDrawing, CV_GRAY2BGR);

    for (size_t i = 0; i < _map.model.size(); i++) {
        if (i % reposeFrequency)
            continue;

        Point2f originalModelPoint = _map.keypoints[i].pt;

        //draw all projected map features on video stream
        circle(_map.imgDrawing, originalModelPoint, 1, CV_RGB(255, 0, 0), 1, FILLED);
        putText(_map.imgDrawing, to_string(i), Point2f(originalModelPoint.x - 1, originalModelPoint.y - 1),
            FONT_HERSHEY_SIMPLEX, 0.25, CV_RGB(255, 0, 0), 1.0);
    }
}

//------------------------------------------------------------------------------
SLbool SLCVTrackerFeatures::track(SLCVMat imageGray,
                                  SLCVMat image,
                                  SLCVCalibration *calib,
                                  SLSceneView *sv)
{
    assert(!image.empty() && "Image is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");

    vector<DMatch> inlierMatches;
    vector<Point2f> points2D;
    SLCVVKeyPoint keypoints;
    SLCVMat descriptors;

    SLCVMat rvec = cv::Mat::zeros(3, 3, CV_64FC1);      // rotation matrix
    SLCVMat tvec = cv::Mat::zeros(3, 1, CV_64FC1);      // translation matrix
    bool foundPose = false;

#if DEBUG_OUTPUT
    cout << "--------------------------------------------------" << endl << "Processing frame #" << frameCount << "..." << endl;
    //cout << "Actual average amount of 2D points: " << setprecision(3) <<  _prev.points2D.size() << endl;
#endif

    // Detect and describe keypoints on model ###############################
     if (frameCount == 0) { // Load reference points at start
         _calib = calib;
         loadModelPoints();
     }
     // #####################################################################

    // TODO: Handle detecting || tracking correctly!
    if (FORCE_REPOSE || frameCount % 20 == 0) { // || lastNmatchedKeypoints * 0.6f > _prev.points2D.size()) {
#if DISTINGUISH_FEATURE_DETECT_COMPUTE
        // Detect keypoints ####################################################
        keypoints = getKeypoints(imageGray);
        // #####################################################################


        // Extract descriptors from keypoints ##################################
        descriptors = getDescriptors(imageGray , keypoints);
        // #####################################################################
#else
        getKeypointsAndDescriptors(imageGray, keypoints, descriptors);
#endif
        // Feature Matching ####################################################
        vector<DMatch> matches = getFeatureMatches(descriptors);
        // #####################################################################


        // POSE calculation ####################################################
        bool useExtrinsicGuess = false;
        if (_prev.foundPose) {
            useExtrinsicGuess = true;
            rvec = _prev.rvec;
            tvec = _prev.tvec;
        }
        foundPose = calculatePose(image, keypoints, matches, inlierMatches, points2D, rvec, tvec, useExtrinsicGuess, descriptors);
        // #####################################################################

    } else {
        // Feature tracking ####################################################
        // points2D should be empty, this feature points are the calculated points with optical flow
        //trackWithOptFlow(_prev.image, _prev.points2D, image, points2D, rvec, tvec);
        //if (foundPose) lastNmatchedKeypoints = points2D.size(); // Write actual detected points amount
        // #####################################################################
    }

    // Update camera object SLCVMatrix  ########################################
    if (foundPose) {
        // Converts calulated extrinsic camera components (translation & rotation) to OpenGL camera SLCVMatrix
        _pose = createGLMatrix(tvec, rvec);

        // Update Scene Graph camera to display model correctly (positioning cam relative to world coordinates)
        sv->camera()->om(_pose.inverse());
    }

#if DRAW_KEYPOINTS
    if (!keypoints.empty()) {
        SLCVMat imgKeypoints;
        drawKeypoints(image, keypoints, imgKeypoints);
        imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-keypoints.png", imgKeypoints);
    }
#endif

#if defined(SAVE_SNAPSHOTS_OUTPUT)
    // Draw matches
    if (!inlierMatches.empty()) {
        SLCVMat imgMatches;
        drawMatches(imageGray, keypoints, _map.frameGray, _map.keypoints, inlierMatches, imgMatches);
        imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-matching.png", imgMatches);
    }

    // Draw optical flow
    if (/*foundPose &&*/ _prev.points2D.size() == points2D.size() && points2D.size() > 0) {
        SLCVMat optFlow, rgb;
        imageGray.copyTo(optFlow);
        cvtColor(optFlow, rgb, CV_GRAY2BGR);
        for (size_t i=0; i < points2D.size(); i++) {
            cv::arrowedLine(rgb, _prev.points2D[i], points2D[i], Scalar(0, 255, 0), 1, LINE_8, 0, 0.2);
        }
        imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-optflow.png", rgb);
    }
#endif

    // Copy actual frame data to _prev struct for next frame
    _prev.imageGray = imageGray;
    _prev.image = image;
    _prev.points2D = points2D;
    _prev.rvec = rvec;
    _prev.tvec = tvec;
    _prev.foundPose = foundPose;

    frameCount++;

    return false;
}

//-----------------------------------------------------------------------------
SLCVVKeyPoint SLCVTrackerFeatures::getKeypoints(const SLCVMat &imageGray)
{
    SLCVVKeyPoint keypoints;
    SLfloat detectTimeMillis = SLScene::current->timeMilliSec();
    SLScene::current->_detector->detect(imageGray, keypoints);

    SLfloat detectionDifference = SLScene::current->timeMilliSec() - detectTimeMillis;
    SLScene::current->setDetectionTimesMS(detectionDifference);

#if TRACKING_MEASUREMENT
    if (detectionDifference > 0) {
        if (detectionDifference < low_detection_milis)
            low_detection_milis = detectionDifference;
        else if (detectionDifference > high_detection_milis)
            high_detection_milis = detectionDifference;

        if (frameCount > 0)
            sum_detection_millis += detectionDifference;
    }
#endif

    return keypoints;
}

//-----------------------------------------------------------------------------
Mat SLCVTrackerFeatures::getDescriptors(const SLCVMat &imageGray, SLCVVKeyPoint &keypoints)
{
    SLCVMat descriptors;
    SLfloat computeTimeMillis = SLScene::current->timeMilliSec();
    SLScene::current->_descriptor->compute(imageGray, keypoints, descriptors);

    SLfloat computeDifference = SLScene::current->timeMilliSec() - computeTimeMillis;
    SLScene::current->setFeatureTimesMS(computeDifference);

#if TRACKING_MEASUREMENT
    if (computeDifference > 0) {
        if (computeDifference < low_compute_milis)
            low_compute_milis = computeDifference;
        else if (computeDifference > high_compute_milis)
            high_compute_milis = computeDifference;

        if (frameCount > 0)
            sum_compute_millis += computeDifference;
    }
#endif

    return descriptors;
}

//-----------------------------------------------------------------------------
void SLCVTrackerFeatures::getKeypointsAndDescriptors(const SLCVMat &imageGray, SLCVVKeyPoint &keypoints, SLCVMat &descriptors)
{
    SLfloat detectComputeTimeMillis = SLScene::current->timeMilliSec();
    SLScene::current->_descriptor->detectAndCompute(imageGray, keypoints, descriptors);
    SLfloat detectComputeDifference = SLScene::current->timeMilliSec() - detectComputeTimeMillis;
    SLScene::current->setFeatureTimesMS(detectComputeDifference);

#if TRACKING_MEASUREMENT
    if (detectComputeDifference > 0) {
        if (detectComputeDifference < low_detectcompute_milis)
            low_detectcompute_milis = detectComputeDifference;
        else if (detectComputeDifference > high_detectcompute_milis)
            high_detectcompute_milis = detectComputeDifference;

        if (frameCount > 0)
            sum_detectcompute_millis += detectComputeDifference;
    }
#endif
}

//-----------------------------------------------------------------------------
vector<DMatch> SLCVTrackerFeatures::getFeatureMatches(const SLCVMat &descriptors)
{
    SLfloat SLCVMatchTimeMillis = SLScene::current->timeMilliSec();

    // 1. Get SLCVMatches with KNN algorithm ######################################################################################
    int k = 2;
    vector<vector<DMatch>> SLCVMatches;
    _matcher->knnMatch(descriptors, _map.descriptors, SLCVMatches, k);

    /* Perform ratio test which determines if k SLCVMatches from the knn SLCVMatcher are not too similar.
     *  If the ratio of the the distance of the two SLCVMatches is toward 1, the SLCVMatches are near identically.
     */
    vector<DMatch> goodMatches;
    for(size_t i = 0; i < SLCVMatches.size(); i++) {
        const DMatch &match1 = SLCVMatches[i][0];
        const DMatch &match2 = SLCVMatches[i][1];
        if (match2.distance == 0.0f || ( match1.distance / match2.distance) < minRatio)
            goodMatches.push_back(match1);
    }

    SLScene::current->setMatchTimesMS(SLScene::current->timeMilliSec() - SLCVMatchTimeMillis);
    return goodMatches;
}

//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::calculatePose(const SLCVMat &imageVideo, vector<KeyPoint> &keypoints, vector<DMatch> &allMatches,
    vector<DMatch> &inlierMatches, vector<Point2f> &inlierPoints, SLCVMat &rvec, SLCVMat &tvec, bool extrinsicGuess,
    const SLCVMat& descriptors)
{
    // RANSAC crashes if 0 points are given
    if (allMatches.size() == 0) return 0;

    /* Find 2D/3D correspondences
     *
     *  At the moment we are using only the two correspondences like this:
     *  KeypointsOriginal <-> KeypointsActualscene
     *
     *  Train index --> "Point" in the model
     *  Query index --> "Point" in the actual frame
     */
    vector<Point3f> modelPoints(allMatches.size());
    vector<Point2f> framePoints(allMatches.size());
    for (size_t i = 0; i < allMatches.size(); i++) {
        modelPoints[i] = _map.model[allMatches[i].trainIdx];
        framePoints[i] =  keypoints[allMatches[i].queryIdx].pt;
    }

    // Finding PnP solution
    vector<unsigned char> inliersMask(modelPoints.size());
    bool foundPose = solvePnP(modelPoints, framePoints, extrinsicGuess, rvec, tvec, inliersMask);

    for (size_t i = 0; i < inliersMask.size(); i++) {
        size_t idx = inliersMask[i];
        inlierMatches.push_back(allMatches[idx]);
        inlierPoints.push_back(framePoints[idx]);
    }

#if TRACKING_MEASUREMENT
    sum_allmatches_to_inliers += inlierMatches.size() / allMatches.size();
#endif

    // Pose optimization
    if (foundPose) {
        int matchesBefore = inlierMatches.size();
        optimizePose(imageVideo, keypoints, inlierMatches, rvec, tvec, descriptors);

        modelPoints = vector<Point3f>(inlierMatches.size());
        framePoints = vector<Point2f>(inlierMatches.size());
        for (size_t i = 0; i < inlierMatches.size(); i++) {
            modelPoints[i] = _map.model[inlierMatches[i].trainIdx];
            framePoints[i] =  keypoints[inlierMatches[i].queryIdx].pt;
        }


        foundPose = cv::solvePnP(modelPoints,
                                 framePoints,
                                 _calib->cameraMat(),
                                 _calib->distortion(),
                                 rvec, tvec,
                                 true,
                                 SOLVEPNP_ITERATIVE
        );

#if DEBUG_OUTPUT
        cout << "Optimize pose: " << inlierMatches.size() - matchesBefore << " more matches found" << endl;
#endif

#if TRACKING_MEASUREMENT
        sum_poseopt_difference += inlierMatches.size() / matchesBefore;
#endif
    }

    return foundPose;
}

//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::trackWithOptFlow(SLCVMat &previousFrame, vector<Point2f> &previousPoints,
                                           SLCVMat &currentFrame, vector<Point2f> &predPoints)
{
    if (previousPoints.size() == 0) return false;

    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::Size winSize(15, 15);
    cv::TermCriteria criteria(
    cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
        10,    // terminate after this many iterations, or
        0.03); // when the search window moves by less than this

    // Find next possible feature points based on optical flow
    cv::calcOpticalFlowPyrLK(
                previousFrame, currentFrame, // Previous and current frame
                previousPoints, predPoints,  // Previous and current keypoints coordinates.The latter will be
                                             // expanded if there are more good coordinates detected during OptFlow algorithm
                status,                      // Output vector for keypoint correspondences (1 = SLCVMatch found)
                err,                         // Errors
                winSize,                     // Search window for each pyramid level
                3,                           // Max levels of pyramid creation
                criteria,                    // Configuration from above
                0,                           // Additional flags
                0.001                        // Minimal Eigen threshold
    );

    // Only use points which are not wrong in any way (???) during the optical flow calculation
    for (size_t i = 0; i < status.size(); i++) {
        if (!status[i]) {
            previousPoints.erase(previousPoints.begin() + i);
            predPoints.erase(predPoints.begin() + i);
        }
    }

    // RANSAC crashes if 0 points are given
    if (previousPoints.size() == 0) return false;

    // Call solvePnP to get the pose from the previous known camera pose and the 3D correspondences and the predicted keypoints
    /*
    vector<Point3f> modelPoints(_prev.matches.size());
    vector<Point2f> framePoints(_prev.matches.size());
    for (size_t i = 0; i < _prev.matches.size(); i++) {
        modelPoints[i] =  _map.model[_prev.matches[i].trainIdx];
        framePoints[i] =  predPoints[_prev.matches[i].queryIdx].pt;
    }

    vector<unsigned char> inliersMask(previousPoints.size());
    foundPose = solvePnP(_map.model, predPoints, true, rvec, tvec, inliersMask);
    */

    return false; // foundPose;
}

//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::solvePnP(vector<Point3f> &modelPoints, vector<Point2f> &framePoints, bool guessExtrinsic,
                                   SLCVMat &rvec, SLCVMat &tvec, vector<unsigned char> &inliersMask)
{
    /* We execute first RANSAC to eliminate wrong feature correspondences (outliers) and only use
     * the correct ones (inliers) for PnP solving (https://en.wikipedia.org/wiki/Perspective-n-Point).
     *
     * Methods of solvePnP
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
     * ITERATIVE: Calculates pose using the DLT (Direct Linear Transform) method. If there is
     *  a homography will be much easier and no DLT will be used. Otherwise we are using the DLT
     *  and make a Levenberg-Marquardt optimization. The latter helps to decrease the reprojection
     *  error which is the sum of the squared distances between the image and object points.
     *
     *
     * 1.) Call RANSAC with EPNP ----------------------------
     * The RANdom Sample Consensus algorithm is called to remove "wrong" point correspondences
     *  which makes the solvePnP more robust. The so called inliers are used for calculation,
     *  wrong correspondences (outliers) will be ignored. Therefore the method below will first
     *  run a solvePnP with the EPNP method and returns the reprojection error. EPNP works like
     *  the following:
     *  1. Choose the 4 control pints: C0 as centroid of reference points, C1, C2 and C3 from PCA
     *      of the reference points
     *  2. Compute barycentric coordinates with the control points
     *  3. Derivate the image reference points with the above
     *  .... ???
     *
     * 2.) Call PnP ITERATIVE -------------------------------
     * General problem: We have a calibrated cam and sets of corresponding 2D/3D points.
     *  We will calculate the rotation and translation in respect to world coordinates.
     *
     *  1. If for no extrinsic guess, begin with computation
     *  2. If planarity is detected, find homography, otherwise use DLT method
     *  3. After sucessful determination of a pose, optimize it with Levenberg-Marquardt (iterative part)
     *
     */
    return cv::solvePnPRansac(modelPoints,
                              framePoints,
                              _calib->cameraMat(),
                              _calib->distortion(),
                              rvec, tvec,
                              guessExtrinsic,
                              iterations,
                              reprojectionError,
                              confidence,
                              inliersMask
    );
}

//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::optimizePose(const SLCVMat &imageVideo, vector<KeyPoint> &keypoints, vector<DMatch> &matches,
                                       SLCVMat &rvec, SLCVMat &tvec, const SLCVMat& descriptors)
{
    vector<KeyPoint> bboxFrameKeypoints;
    vector<size_t> frameIndicesInsideRect;
    double localReprojectionErrorSum = 0;

    //matches.clear();

    // 1. Reproject the model points with the calculated POSE
    vector<Point2f> projectedPoints(_map.model.size());
    cv::projectPoints(_map.model, rvec, tvec, _calib->cameraMat(), _calib->distortion(), projectedPoints);

    for (size_t i = 0; i < _map.model.size(); i++)
    {
        //only every reposeFrequency
        if (i % reposeFrequency)
            continue;

        // Get the corresponding projected point of the actual (i) modelpoint
        Point2f projectedModelPoint = projectedPoints[i];

        // 2. Select only before calculated Keypoints within patch with projected "positioning" keypoint as center
        // OpenCV: Top-left origin
        int xTopLeft = projectedModelPoint.x - patchSize / 2;
        int yTopLeft = projectedModelPoint.y - patchSize / 2;
        int xDownRight = xTopLeft + patchSize;
        int yDownRight = yTopLeft + patchSize;

        for (size_t j = 0; j < keypoints.size(); j++) {
            // bbox check
            if (keypoints[j].pt.x > xTopLeft &&
                    keypoints[j].pt.x < xDownRight &&
                    keypoints[j].pt.y > yTopLeft &&
                    keypoints[j].pt.y < yDownRight)
            {
                bboxFrameKeypoints.push_back(keypoints[j]);
                frameIndicesInsideRect.push_back(j);
            }
        }

        //3. SLCVMatch the descriptors of the keypoints inside the rectangle around the projected map point
        //with the descritor of the projected map point.
        //(du musst versuchen den einzelnen descriptor des projizierten map point und die descriptoren
        // der keypoints im aktuellen frame aus den cv::Mat's zu extrahieren und einzeln an knnMatch zu übergeben.
        // Vllt. kann man auch diesen parameter "mask" in der Methode knnmatch verwenden... Weiss ich auch nicht...)
        //todo...
        //nur symbolisch, diese descriptoren müssen wir SLCVMatchen
        //descriptors[i]  => descriptor des map points
        //for( size_t j : frameIndicesInsideRect )
        //	_map.descriptors[frameIndicesInsideRect[j]]
        // => descriptoren der keypoints im rechteck

        // This is our descriptor for the model point i
        Mat modelPointDescriptor = _map.descriptors.row(i);

        // We extract the descriptors which belong to the keypoints inside the rectangle around the projected
        // map point
        Mat bboxPointsDescriptors; //(frameIndicesInsideRect.size(), descriptors.cols, descriptors.type());
        for (size_t j : frameIndicesInsideRect) {
            bboxPointsDescriptors.push_back(descriptors.row(j));
        }

        //4. Match the frame keypoints inside the rectangle with the projected model point
        vector<DMatch> newMatches;
        _matcher->match(bboxPointsDescriptors, modelPointDescriptor, newMatches);

        int k = 0;
        for (size_t j : frameIndicesInsideRect) {
            newMatches[k].trainIdx = i;
            newMatches[k].queryIdx = j;
            k++;
        }

        if (newMatches.size() > 0) {
            //5. Only add the best new match to matches vector
            DMatch bestNewMatch; bestNewMatch.distance = 0;
            for (DMatch newMatch : newMatches) {
                if (bestNewMatch.distance < newMatch.distance) bestNewMatch = newMatch;
            }

            //bestNewMatch.trainIdx = i;
            matches.push_back(bestNewMatch);
        }

#if DRAW_REPROJECTION
        Mat imgReprojection = imageVideo;
#else
        Mat imgReprojection;
        imageVideo.copyTo(imgReprojection);
#endif

#if DRAW_REPROJECTION || defined(SAVE_SNAPSHOTS_OUTPUT)
        /*
         * Draw the projected points and keypoints into the current FRAME
         */
        //draw all projected map features on video stream
        circle(imgReprojection, projectedModelPoint, 1, CV_RGB(255, 0, 0), 1, FILLED);

        //draw the point index and reprojection error
        putText(imgReprojection, to_string(i), Point2f(projectedModelPoint.x - 1, projectedModelPoint.y - 1),
            FONT_HERSHEY_SIMPLEX, 0.25, CV_RGB(255, 0, 0), 1.0);

        Point2f originalModelPoint = _map.keypoints[i].pt;
        double reprojectionError = norm(Mat(projectedModelPoint), Mat(originalModelPoint));
        localReprojectionErrorSum += reprojectionError;

#if DRAW_REPOSE_INFO
        //draw green rectangle around every map point
        rectangle(imgReprojection,
            Point2f(projectedModelPoint.x - patchHalf, projectedModelPoint.y - patchHalf),
            Point2f(projectedModelPoint.x + patchHalf, projectedModelPoint.y + patchHalf),
            CV_RGB(0, 255, 0));
        //draw key points, that lie inside this rectangle
        for (auto kPt : bboxFrameKeypoints)
            circle(imgReprojection, kPt.pt, 1, CV_RGB(0, 0, 255), 1, FILLED);
#endif
#endif

        bboxFrameKeypoints.clear();
        frameIndicesInsideRect.clear();
    }

    sum_reprojection_error += localReprojectionErrorSum / _map.model.size();

#if DRAW_REPOSE_INFO
    double reprojectionErrorAvg = localReprojectionErrorSum / _map.model.size();
    putText(imageVideo, "Reprojection error: " + to_string(reprojectionErrorAvg), Point2f(20, 20),
            FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 2.0);
#endif

#if defined(SAVE_SNAPSHOTS_OUTPUT)
    // Abuse of the drawMatches method to simply draw the two image side by side
    SLCVMat imgOut;
    drawMatches(imageVideo, vector<KeyPoint>(), _map.imgDrawing, vector<KeyPoint>(), vector<DMatch>(), imgOut);
    imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-poseoptimization.png", imgOut);
#endif

    return 0;
}
