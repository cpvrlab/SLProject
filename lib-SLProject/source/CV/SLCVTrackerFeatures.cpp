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
//#include <opencv2/tracking.hpp>

#if defined(SL_OS_WINDOWS)
#include <direct.h>
#endif

using namespace cv;

#define DEBUG_OUTPUT 0
#define FORCE_REPOSE 0
#define OPTIMIZE_POSE 1
#define DISTINGUISH_FEATURE_DETECT_COMPUTE 0

// Settings for drawing things into current camera frame
#define DRAW_KEYPOINTS 1
#define DRAW_REPROJECTION 1
#define DRAW_REPOSE_INFO 1

// Set stones Tracker as default reference image
#ifndef SL_TRACKER_IMAGE_NAME
    #define SL_TRACKER_IMAGE_NAME "stones"
#endif

//#define SL_SAVE_DEBUG_OUTPUT
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
const float reprojection_error = 2.0f;
const double confidence = 0.95;

// Repose patch size
const int reposeFrequency = 10;
const int initialPatchSize = 2;
const int maxPatchSize = 80;

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

int frames_with_pose = 0;
int sum_matches = 0;
int sum_inlier_matches = 0;
float sum_allmatches_to_inliers = 0.0f;
double sum_reprojection_error = 0.0f;
float sum_poseopt_difference = 0.0f;
double translationError = 0;
double rotationError = 0;
#endif

// to_String method since Android does not support full C++11 support...
template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str();
}

//-----------------------------------------------------------------------------
SLCVTrackerFeatures::SLCVTrackerFeatures(SLNode *node) :
    SLCVTracker(node)
{
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

SLCVTrackerFeatures::~SLCVTrackerFeatures()
{
#if TRACKING_MEASUREMENT
    SL_LOG(" \n");
    SL_LOG(" \n");
    SL_LOG("------------------------------------------------------------------\n");
    SL_LOG("SLCVTrackerFeatures statistics \n");
    SL_LOG("------------------------------------------------------------------\n");
    SL_LOG("Avg calculation time per frame                   : %f ms\n", SLScene::current->trackingTimesMS().average());
    SL_LOG(" \n");
    SL_LOG("Settings for Pose estimation: ------------------------------------\n");
    SL_LOG("Features                                         : %d\n", nFeatures);
    SL_LOG("Minimal ratio for 2 best matches                 : %f\n", minRatio);
    SL_LOG("RANSAC iterations                                : %d\n", iterations);
    SL_LOG("RANSAC mean reprojection error                   : %f\n", reprojection_error);
    SL_LOG("RANSAC confidence                                : %d\n",confidence );
    SL_LOG("Repose frequency                                 : Each %d point\n", reposeFrequency);
    SL_LOG("Initial patch size for Pose optimization         : %d pixels\n", initialPatchSize);
    SL_LOG("Maximal patch size for Pose optimization         : %d pixels\n", maxPatchSize);
    SL_LOG(" \n");
#if DISTINGUISH_FEATURE_DETECT_COMPUTE
    SL_LOG("\nDetection: -------------------------------------------------------\n");
    SL_LOG("Min detection Time (ms)                          : %d\n", low_detection_milis);
    SL_LOG("Avg detection Time (ms)                          : %d\n", sum_detection_millis / frameCount);
    SL_LOG("High detection Time (ms)                         : %d\n", high_detection_milis);
    SL_LOG(" \n");
    SL_LOG("\nExtraction: ------------------------------------------------------\n");
    SL_LOG("Min compute Time (ms)                            : %d\n", low_compute_milis);
    SL_LOG("Avg compute Time (ms)                            : %d\n", sum_compute_millis / frameCount);
    SL_LOG("High compute Time (ms)                           : %d\n", high_compute_milis);
#else
    SL_LOG("Feature detection and description: -------------------------------\n");
    SL_LOG("Min detect & compute Time                        : %f ms\n", low_detectcompute_milis);
    SL_LOG("Avg detect & compute Time                        : %f ms\n", sum_detectcompute_millis / frameCount);
    SL_LOG("High detect & compute Time                       : %f ms\n", high_detectcompute_milis);
#endif
    SL_LOG(" \n");
    SL_LOG("Pose information: ------------------------------------------------\n");
    SL_LOG("Avg allmatches to inliers proposition            : %f\n", sum_allmatches_to_inliers / frameCount);
    SL_LOG("Avg reprojection error (only if POSE)            : %f\n", sum_reprojection_error / frames_with_pose);
    SL_LOG("Pose found                                       : %d of %d frames\n", frames_with_pose, frameCount);
    SL_LOG("Avg matches                                      : %f\n", sum_matches / frames_with_pose);
    SL_LOG("Avg inlier matches                               : %f\n", sum_inlier_matches / frames_with_pose);
    SL_LOG("Avg more matches with Pose optimization          : %f\n", sum_poseopt_difference / frames_with_pose);
    SL_LOG("Avg Rotation error                               : %f\n", rotationError / frames_with_pose);
    SL_LOG("Avg Translation error                            : %f\n", translationError / frames_with_pose);

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
#if defined(SAVE_SNAPSHOTS_OUTPUT) || defined(DRAW_REPOSE_INFO)
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
#endif
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
    SLCVVKeyPoint keypoints;
    SLCVMat descriptors;
    vector<DMatch> matches;
    float reprojectionError = 0;

    SLCVMat rvec = cv::Mat::zeros(3, 1, CV_64FC1);      // rotation matrix
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

     bool useExtrinsicGuess = false;
     if (_prev.foundPose) {
         useExtrinsicGuess = true;
         rvec = _prev.rvec;
         tvec = _prev.tvec;
     }

    // TODO: Handle detecting || tracking correctly!
    if (FORCE_REPOSE
        || frameCount % 30 == 0
        || !_prev.foundPose
        || _prev.points2D.size() < 0.8 * _inlierPoints3D.size()
        || _prev.reprojectionError > 3 * reprojectionError)
    {
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
        matches = getFeatureMatches(descriptors);
        // #####################################################################

        // POSE calculation ####################################################
        foundPose = calculatePose(image, keypoints, matches, inlierMatches, rvec, tvec, useExtrinsicGuess, descriptors);
        // #####################################################################
        cout << "Relocalisation with " << _inlierPoints2D.size() << " points ..." << endl;
        _prev.points2D = _inlierPoints2D;

    } else {
        // Feature tracking ####################################################
        // Two ways possible: Eighter track the existing keypoints with Optical Flow and calculate the
        // relative Pose or try to match the inlier features locally.

        // Optical Flow approach
        foundPose = trackWithOptFlow(_prev.imageGray, _prev.points2D, imageGray, rvec, tvec, image);

        // Track features with local matching (make use of already used function optimizePose)
        // getKeypointsAndDescriptors(imageGray, keypoints, descriptors);
        // optimizePose(image, keypoints, descriptors, matches, rvec, tvec, reprojectionError, true);
        // #####################################################################r
    }

    // Update camera object SLCVMatrix  ########################################
    if (foundPose) {
        // Converts calulated extrinsic camera components (translation & rotation) to OpenGL camera SLCVMatrix
        _pose = createGLMatrix(tvec, rvec);

        // Update Scene Graph camera to display model correctly (positioning cam relative to world coordinates)
        sv->camera()->om(_pose.inverse());

        frames_with_pose++;
    }

#if defined(SAVE_SNAPSHOTS_OUTPUT)
    // Draw keypoints
    if (!keypoints.empty()) {
        SLCVMat imgKeypoints;
        drawKeypoints(image, keypoints, imgKeypoints);
        imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-keypoints.png", imgKeypoints);
    }

    // Draw matches
    if (!inlierMatches.empty()) {
        SLCVMat imgMatches;
        drawMatches(imageGray, keypoints, _map.frameGray, _map.keypoints, inlierMatches, imgMatches,CV_RGB(255,0,0), CV_RGB(255,0,0));
        imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-matching.png", imgMatches);
    }
#endif

    // Copy actual frame data to _prev struct for next frame
    imageGray.copyTo(_prev.imageGray);
    _prev.reprojectionError = reprojectionError;
    _prev.image = image;

    _prev.points3D = _inlierPoints3D;
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
    vector<DMatch> &inlierMatches, SLCVMat &rvec, SLCVMat &tvec, bool extrinsicGuess, const SLCVMat& descriptors)
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
    vector<unsigned char> inliersMask(modelPoints.size());
    bool foundPose =  cv::solvePnPRansac(modelPoints,
                              framePoints,
                              _calib->cameraMat(),
                              _calib->distortion(),
                              rvec, tvec,
                              extrinsicGuess,
                              iterations,
                              reprojection_error,
                              confidence,
                              inliersMask
    );

    // Get matches with help of inlier indices
    _inlierPoints2D.clear();
    _inlierPoints3D.clear();
    for (size_t i = 0; i < inliersMask.size(); i++) {
        size_t idx = inliersMask[i];
        inlierMatches.push_back(allMatches[idx]);
        _inlierPoints2D.push_back(framePoints[idx]);
        _inlierPoints3D.push_back(modelPoints[idx]);
    }


    // Pose optimization
    if (foundPose) {
#if OPTIMIZE_POSE
        float matchesBefore = inlierMatches.size();
        foundPose = optimizePose(imageVideo, keypoints, descriptors, inlierMatches, rvec, tvec);

#if DEBUG_OUTPUT
        cout << "Optimize pose: " << inlierMatches.size() - matchesBefore << " more matches found" << endl;
#endif // DEBUG_OUTPUT
#endif // OPTIMIZE_POSE

#if TRACKING_MEASUREMENT
        sum_matches += allMatches.size();
        sum_inlier_matches += inlierMatches.size();
        sum_allmatches_to_inliers += inlierMatches.size() / allMatches.size();
#if OPTIMIZE_POSE
        sum_poseopt_difference += inlierMatches.size() / matchesBefore;
#endif // OPTIMIZE_POSE
#endif // TRACKING_MEASUREMENT
    }

    return foundPose;
}


//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::optimizePose(const SLCVMat &imageVideo, vector<KeyPoint> &keypoints, const SLCVMat& descriptors,
    vector<DMatch> &matches, SLCVMat &rvec, SLCVMat &tvec, float reprojectionError)
{

    // 1. Reproject the model points with the calculated POSE
    vector<Point2f> projectedPoints(_map.model.size());
    cv::projectPoints(_map.model, rvec, tvec, _calib->cameraMat(), _calib->distortion(), projectedPoints);

    vector<KeyPoint> bboxFrameKeypoints;
    vector<size_t> frameIndicesInsideRect;

    for (size_t i = 0; i < _map.model.size(); i++)
    {
        //only every reposeFrequency
        if (i % reposeFrequency)
            continue;

        // Check if this point has already a match inside matches, continue if so
        int alreadyMatched = 0;
        for (size_t j = 0; j < matches.size(); j++) {
            if (matches[j].trainIdx == i) alreadyMatched++;
        }

        if (alreadyMatched > 0) continue;

        // Get the corresponding projected point of the actual (i) modelpoint
        Point2f projectedModelPoint = projectedPoints[i];
        vector<DMatch> newMatches;

        int patchSize = initialPatchSize;

        // Adaptive patch size
        while (newMatches.size() == 0 && patchSize <= maxPatchSize)
        {
            // Increase matches by even number
            patchSize += 2;
            newMatches.clear();
            bboxFrameKeypoints.clear();
            frameIndicesInsideRect.clear();

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

            // This is our descriptor for the model point i
            Mat modelPointDescriptor = _map.descriptors.row(i);

            // We extract the descriptors which belong to the keypoints inside the rectangle around the projected
            // map point
            Mat bboxPointsDescriptors;
            for (size_t j : frameIndicesInsideRect) {
                bboxPointsDescriptors.push_back(descriptors.row(j));
            }

            //4. Match the frame keypoints inside the rectangle with the projected model point
            _matcher->match(bboxPointsDescriptors, modelPointDescriptor, newMatches);
        }

#if DEBUG_OUTPUT
        cout << "Matches inside patch: " << newMatches.size() << endl;
#endif // DEBUG_OUTPUT

        if (newMatches.size() > 0) {
            for (size_t j = 0; j < frameIndicesInsideRect.size(); j++) {
                newMatches[j].trainIdx = i;
                newMatches[j].queryIdx = frameIndicesInsideRect[j];
            }

            //5. Only add the best new match to matches vector
            DMatch bestNewMatch; bestNewMatch.distance = 0;
            for (DMatch newMatch : newMatches) {
                if (bestNewMatch.distance < newMatch.distance) bestNewMatch = newMatch;
            }

            //5. Only add the best new match to matches vector
            matches.push_back(bestNewMatch);
        }

        // Get the keypoint which was used for pose estimation
        Point2f keypointForPoseEstimation = keypoints[matches.back().queryIdx].pt;
        reprojectionError += norm(Mat(projectedModelPoint), Mat(keypointForPoseEstimation));

#if DRAW_REPROJECTION
        Mat imgReprojection = imageVideo;
#else
        Mat imgReprojection;
        imageVideo.copyTo(imgReprojection);
#endif

#if DRAW_REPROJECTION || defined(SAVE_SNAPSHOTS_OUTPUT)
#if DRAW_REPOSE_INFO
        //draw green rectangle around every map point
        rectangle(imgReprojection,
            Point2f(projectedModelPoint.x - patchSize / 2, projectedModelPoint.y - patchSize / 2),
            Point2f(projectedModelPoint.x + patchSize / 2, projectedModelPoint.y + patchSize / 2),
            CV_RGB(0, 255, 0));
        //draw key points, that lie inside this rectangle
        for (auto kPt : bboxFrameKeypoints)
            circle(imgReprojection, kPt.pt, 1, CV_RGB(0, 0, 255), 1, FILLED);
#endif
        /*
         * Draw the projected points and keypoints into the current FRAME
         */
        //draw all projected map features and the original keypoint on video stream
        circle(imgReprojection, projectedModelPoint, 2, CV_RGB(255, 0, 0), 1, FILLED);
        circle(imgReprojection, keypointForPoseEstimation, 5, CV_RGB(0, 0, 255), 1, FILLED);

        //draw the point index and reprojection error
        putText(imgReprojection, to_string(i), Point2f(projectedModelPoint.x - 2, projectedModelPoint.y - 5),
            FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(255, 0, 0), 1.0);

#endif
    }

    sum_reprojection_error += reprojectionError / _map.model.size();

#if DRAW_REPROJECTION
    // Draw the projection error for the current frame
    putText(imageVideo, "Reprojection error: " + to_string(reprojectionError / _map.model.size()), Point2f(20, 20),
            FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 2.0);
#endif
    Mat prevRmat, currRmat;
    if(_prev.foundPose){
        Rodrigues(_prev.rvec, prevRmat);
        Rodrigues(rvec, currRmat);
        double rotationError_rad = acos((trace(prevRmat * currRmat).val[0] - 1.0)/2.0);
        rotationError += rotationError_rad*180/3.14;
        translationError += cv::norm(_prev.tvec, tvec);
    }
#if defined(SAVE_SNAPSHOTS_OUTPUT)
    // Abuse of the drawMatches method to simply draw the two image side by side
    SLCVMat imgOut;
    drawMatches(imageVideo, vector<KeyPoint>(), _map.imgDrawing, vector<KeyPoint>(), vector<DMatch>(), imgOut, CV_RGB(255,0,0), CV_RGB(255,0,0));
    imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-poseoptimization.png", imgOut);
#endif

    // Optimize POSE
    vector<Point3f> modelPoints = vector<Point3f>(matches.size());
    vector<Point2f> framePoints = vector<Point2f>(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        modelPoints[i] = _map.model[matches[i].trainIdx];
        framePoints[i] =  keypoints[matches[i].queryIdx].pt;
    }

    if (modelPoints.size() == 0) return false;
    return cv::solvePnP(modelPoints,
                             framePoints,
                             _calib->cameraMat(),
                             _calib->distortion(),
                             rvec, tvec,
                             true,
                             SOLVEPNP_ITERATIVE
    );
}

//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::trackWithOptFlow(Mat &previousFrame, vector<Point2f> &prev2DPoints, Mat &currentFrame,
    Mat &rvec, Mat &tvec, SLCVMat &frame)
{
    if (prev2DPoints.size() < 4) return false;

    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::Size winSize(15, 15);
    cv::TermCriteria criteria(
    cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
        10,    // terminate after this many iterations, or
        0.03); // when the search window moves by less than this

    // Find next possible feature points based on optical flow
    vector<Point2f> pred2DPoints(prev2DPoints.size());
    cv::calcOpticalFlowPyrLK(
                previousFrame,               // Previous and current frame
                currentFrame,
                prev2DPoints,                // Previous and current keypoints coordinates.The latter will be
                pred2DPoints,                // expanded if there are more good coordinates detected during OptFlow algorithm
                status,                      // Output vector for keypoint correspondences (1 = Match found)
                err,                         // Errors
                winSize,                     // Search window for each pyramid level
                3,                           // Max levels of pyramid creation
                criteria,                    // Configuration from above
                0,                           // Additional flags
                0.001                        // Minimal Eigen threshold
    );

    // Only use points which are not wrong in any way (???) during the optical flow calculation
    vector<Point2f> frame2DPoints;
    vector<Point3f> model3DPoints;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            frame2DPoints.push_back(pred2DPoints[i]);
            model3DPoints.push_back(_inlierPoints3D[i]);
        }
    }

#if defined(SAVE_SNAPSHOTS_OUTPUT)
    // Draw optical flow
    SLCVMat optFlow, rgb;
    currentFrame.copyTo(optFlow);
    cvtColor(optFlow, rgb, CV_GRAY2BGR);
    for (size_t i=0; i < prev2DPoints.size(); i++)
        cv::arrowedLine(rgb, prev2DPoints[i], pred2DPoints[i], Scalar(0, 255, 0), 1, LINE_8, 0, 0.2);

    imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-optflow.png", rgb);
#endif

#if defined(DRAW_KEYPOINTS)
    for (size_t i=0; i < frame2DPoints.size(); i++)
        circle(frame, frame2DPoints[i], 2, Scalar(0, 255, 0));
#endif
    _prev.points2D = frame2DPoints;
    return cv::solvePnP(model3DPoints,
                        frame2DPoints,
                        _calib->cameraMat(),
                        _calib->distortion(),
                        rvec, tvec,
                        true
    );
}
