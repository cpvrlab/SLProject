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

#define DEBUG 0
#define FORCE_REPOSE 1

#if defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS)
#define SAVE_SNAPSHOTS_OUTPUT "/tmp/cv_tracking/"
#elif defined(SL_OS_WINDOWS)
#define SAVE_SNAPSHOTS_OUTPUT "cv_tracking/"
#endif
// Feature detection and extraction
const int nFeatures = 800;
const float minRatio = 0.8f;
#define FLANN_BASED 0

// RANSAC parameters
const int iterations = 400;
const float reprojectionError = 3.0f;
const double confidence = 0.95;

// Repose patch size (TODO: Adjust automatically)
const int patchSize = 30;
const int patchHalf = patchSize / 2;
const int reposeFrequency = 20;

// Benchmarking
#define TRACKING_MEASUREMENT 0
#if TRACKING_MEASUREMENT
float low_detection_milis = 1000.0f;
float avg_detection_milis;
float high_detection_milis;

float low_compute_milis = 1000.0f;
float avg_compute_milis;
float high_compute_milis;
#endif
//-----------------------------------------------------------------------------
SLCVTrackerFeatures::SLCVTrackerFeatures(SLNode *node) :
        SLCVTracker(node)
{
    SLScene::current->_detector->setDetector(new SLCVRaulMurOrb(nFeatures, 1.44f, 3, 30, 20));
    SLScene::current->_descriptor->setDescriptor(ORB::create(nFeatures, 1.44f, 3, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30));

#if FLANN_BASED
    _matcher = new FlannBasedMatcher();
#else
    _matcher =  BFMatcher::create(BFMatcher::BRUTEFORCE_HAMMING, false);
#endif

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

//------------------------------------------------------------------------------
void SLCVTrackerFeatures::loadModelPoints()
{
    // Read reference marker
    SLGLTexture* trackerTexture = new SLGLTexture("stones.jpg");
    SLCVImage* img = trackerTexture->images()[0];
    cvtColor(img->cvMat(), _map.frameGray, CV_RGB2GRAY);

    // Detect and compute features in marker image
     SLScene::current->_detector->detect(_map.frameGray, _map.keypoints);
     SLScene::current->_descriptor->compute(_map.frameGray, _map.keypoints, _map.descriptors);

    // Calculates proprtion of MM and Pixel (sample measuring)
    const SLfloat lengthMM = 297.0;
    const SLfloat lengthPX = 2 * _calib->cx();
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
        circle(_map.imgDrawing, Point2f(originalModelPoint.x, originalModelPoint.y), 1, CV_RGB(255, 0, 0), 1, FILLED);
        putText(_map.imgDrawing, to_string(i), Point2f(originalModelPoint.x - 1, originalModelPoint.y - 1),
            FONT_HERSHEY_SIMPLEX, 0.25, CV_RGB(255, 0, 0), 1.0);

        //draw green rectangle around every map point
        rectangle(_map.imgDrawing,
            Point2f(originalModelPoint.x - patchHalf, originalModelPoint.y - patchHalf),
            Point2f(originalModelPoint.x + patchHalf, originalModelPoint.y + patchHalf),
            CV_RGB(0, 255, 0));
    }
}

//------------------------------------------------------------------------------
SLbool SLCVTrackerFeatures::track(SLCVMat imageGray,
                                  SLCVMat image,
                                  SLCVCalibration *calib,
                                  SLSceneView *sv)
{
#if TRACKING_MEASUREMENT
    if (frameCount == 700){
           ofstream myfile;
           myfile.open ("/tmp/tracker_stats.txt");
           myfile << "Min Detection Time (Ms) " << low_detection_milis << "\n";
           myfile << "Avg Detection Time (Ms) " << avg_detection_milis << "\n";
           myfile << "High Detection Time (Ms) " << high_detection_milis << "\n";

           myfile << "Min Compute Time (Ms) " << low_compute_milis << "\n";
           myfile << "Avg Compute Time (Ms) " << avg_compute_milis << "\n";
           myfile << "High Compute Time (Ms) " << high_compute_milis << "\n";
           myfile.close();
    }
#endif
    assert(!image.empty() && "Image is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");

    vector<DMatch> inlierMatches;
    vector<Point2f> points2D;
    SLCVVKeyPoint keypoints;
    SLCVMat rvec = cv::Mat::zeros(3, 3, CV_64FC1);      // rotation SLCVMatrix
    SLCVMat tvec = cv::Mat::zeros(3, 1, CV_64FC1);      // translation SLCVMatrix
    bool foundPose = false;

#if DEBUG
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

        // Detect keypoints ####################################################
        keypoints = getKeypoints(imageGray);
        // #####################################################################


        // Extract descriptors from keypoints ##################################
        SLCVMat descriptors = getDescriptors(imageGray , keypoints);
        // #####################################################################


        // Feature SLCVMatching ####################################################
        vector<DMatch> SLCVMatches = getFeatureMatches(descriptors);
        // #####################################################################


        // POSE calculation ####################################################
        bool useExtrinsicGuess = false;
        if (_prev.foundPose) {
            useExtrinsicGuess = true;
            rvec = _prev.rvec;
            tvec = _prev.tvec;
        }
        foundPose = calculatePose(image, keypoints, SLCVMatches, inlierMatches, points2D, rvec, tvec, useExtrinsicGuess, descriptors );
        // #####################################################################

#if DEBUG
        cout << "RePOSE with help of " << inlierMatches.size() << " keypoints (RANSAC inliers)" << endl;
#endif
    } else {
#if DEBUG
        cout << "Going to track previous feature points..." << endl;
#endif
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

#if defined(SAVE_SNAPSHOTS_OUTPUT)
    // Draw SLCVMatches
    if (foundPose && !inlierMatches.empty()) {
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

#if TRACKING_MEASUREMENT
    SLfloat time = SLScene::current->timeMilliSec() - detectTimeMillis;
    if (time != 0){
        if (time < low_detection_milis){
            low_detection_milis = time;
        }
        else if (time > high_detection_milis){
            high_detection_milis = time;
        }
        if (frameCount > 0)
        avg_detection_milis = (frameCount*avg_detection_milis + time)/(1+frameCount);
    }
#endif
    SLScene::current->setDetectionTimesMS(SLScene::current->timeMilliSec() - detectTimeMillis);
    return keypoints;
}

//-----------------------------------------------------------------------------
Mat SLCVTrackerFeatures::getDescriptors(const SLCVMat &imageGray, SLCVVKeyPoint &keypoints)
{
    SLCVMat descriptors;
    SLfloat computeTimeMillis = SLScene::current->timeMilliSec();
    SLScene::current->_descriptor->compute(imageGray, keypoints, descriptors);
#if TRACKING_MEASUREMENT
    SLfloat time = SLScene::current->timeMilliSec() - computeTimeMillis;
    if (time != 0.0f){
        if (time < low_compute_milis){
            low_compute_milis = time;
        }
        else if (time > high_compute_milis){
            high_compute_milis = time;
        }
        if (frameCount > 0){
            avg_compute_milis = (avg_compute_milis*frameCount + time)/(1+frameCount);
        }
        else {
            avg_compute_milis = time;
        }
    }
#endif
    SLScene::current->setFeatureTimesMS(SLScene::current->timeMilliSec() - computeTimeMillis);
    return descriptors;
}

//-----------------------------------------------------------------------------
vector<DMatch> SLCVTrackerFeatures::getFeatureMatches(const SLCVMat &descriptors)
{
    SLfloat SLCVMatchTimeMillis = SLScene::current->timeMilliSec();

    // 1. Get SLCVMatches with FLANN or KNN algorithm ######################################################################################
#if FLANN_BASED
    if(descriptors.type() != CV_32F) descriptors.convertTo(descriptors, CV_32F);
    if(_map.descriptors.type() != CV_32F) _map.descriptors.convertTo(_map.descriptors, CV_32F);

    vector<DMatch> goodMatches;
    _matcher->match(descriptors, _map.descriptors, goodMatches);
#else
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
#endif

    SLScene::current->setMatchTimesMS(SLScene::current->timeMilliSec() - SLCVMatchTimeMillis);
    return goodMatches;
}

//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::calculatePose(const SLCVMat &imageVideo, vector<KeyPoint> &keypoints,
    vector<DMatch> &allMatches, vector<DMatch> &inlierMatches, vector<Point2f> &inlierPoints, SLCVMat &rvec, SLCVMat &tvec,
    bool extrinsicGuess, const SLCVMat& descriptors, int iteration)
{

    cout << "iteration "<< iteration <<endl;

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

    // Pose optimization
    if (foundPose) {
        if (iteration == 1) return foundPose;
        optimizePose(imageVideo, keypoints, inlierMatches, rvec, tvec, descriptors);
        calculatePose(imageVideo, keypoints, allMatches, inlierMatches, inlierPoints, rvec, tvec, extrinsicGuess, descriptors, ++iteration);
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
bool SLCVTrackerFeatures::solvePnP(vector<Point3f> &modelPoints, vector<Point2f> &framePoints,
                                   bool guessExtrinsic, SLCVMat &rvec, SLCVMat &tvec, vector<unsigned char> &inliersMask)
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

    int type;
    if (guessExtrinsic) {
        type = cv::SOLVEPNP_ITERATIVE;
    } else {
        type = cv::SOLVEPNP_EPNP;
    }

    //TODO: Split up
    return cv::solvePnPRansac(modelPoints,
                       framePoints,
                       _calib->cameraMat(),
                       _calib->distortion(),
                       rvec, tvec,
                       guessExtrinsic,
                       iterations,
                       reprojectionError,
                       confidence,
                       inliersMask,
                       type
    );
}

//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::optimizePose(const SLCVMat &imageVideo, vector<KeyPoint> &keypoints,
                                       vector<DMatch> &matches, SLCVMat &rvec, SLCVMat &tvec, const SLCVMat& descriptors)
{
    vector<KeyPoint> bboxFrameKeypoints;
    vector<size_t> frameIndicesInsideRect;

    // 1. Reproject the model points with the calculated POSE
    vector<Point2f> projectedPoints(_map.model.size());
    cv::projectPoints(_map.model, rvec, tvec, _calib->cameraMat(), _calib->distortion(), projectedPoints);

    for (size_t i = 0; i < _map.model.size(); i++)
    {
        //only every reposeFrequency
        if (i % reposeFrequency)
            continue;

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

#if defined(SAVE_SNAPSHOTS_OUTPUT)
        /*
         * Draw the projected points and keypoints into the current FRAME
         */
        //draw all projected map features on video stream
        circle(imageVideo, projectedModelPoint, 1, CV_RGB(255, 0, 0), 1, FILLED);
        putText(imageVideo, to_string(i), Point2f(projectedModelPoint.x - 1, projectedModelPoint.y - 1),
            FONT_HERSHEY_SIMPLEX, 0.25, CV_RGB(255, 0, 0), 1.0);

        //draw green rectangle around every map point
        rectangle(imageVideo,
            Point2f(projectedModelPoint.x - patchHalf, projectedModelPoint.y - patchHalf),
            Point2f(projectedModelPoint.x + patchHalf, projectedModelPoint.y + patchHalf),
            CV_RGB(0, 255, 0));
        //draw key points, that lie inside this rectangle
        for (auto kPt : bboxFrameKeypoints)
            circle(imageVideo, kPt.pt, 1, CV_RGB(0, 0, 255), 1, FILLED);
#endif

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
        SLCVMat modelPointDescriptor = _map.descriptors.row(i);

        // We extract the descriptors which belong to the keypoints inside the rectangle around the projected
        // map point
        SLCVMat bboxPointsDescriptors;
        for (size_t j : frameIndicesInsideRect) {
            SLCVMat descriptorInsideRectangle = descriptors.row(j);
            bboxPointsDescriptors.push_back(descriptorInsideRectangle);
        }

        //4. Match the frame keypoints inside the rectangle with the projected model point
        vector<DMatch> newMatches;
        _matcher->match(bboxPointsDescriptors, modelPointDescriptor, newMatches);

        //5. Append the new matches to the already found matches
        matches.insert(matches.end(), newMatches.begin(), newMatches.end());

#if DEBUG
        // cout << "Newly added matches: " << matches.size() - newMatches.size() << endl;
#endif

        bboxFrameKeypoints.clear();
        frameIndicesInsideRect.clear();
    }

#if defined(SAVE_SNAPSHOTS_OUTPUT)
    // Abuse of the drawMatches method to simply draw the two image side by side
    SLCVMat imgOut;
    drawMatches(imageVideo, vector<KeyPoint>(), _map.imgDrawing, vector<KeyPoint>(), vector<DMatch>(), imgOut);
    imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-poseoptimization.png", imgOut);
#endif

    return 0;
}
