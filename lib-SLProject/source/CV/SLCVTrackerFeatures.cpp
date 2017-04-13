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
#define DRAW_ALL_MAP_PTS_ON_VIDEO 1
#define FIND_AND_DRAW_KPTS_IN_ROI 1

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
        SLCVTracker(node) {
    SLScene::current->_detector->setDetector(new SLCVRaulMurOrb(nFeatures, 1.44f, 3, 30, 20));
    SLScene::current->_descriptor->setDescriptor(ORB::create(nFeatures, 1.44f, 3, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30));

    #if FLANN_BASED
    _matcher = new FlannBasedMatcher();
    #else
    _matcher =  BFMatcher::create(BFMatcher::BRUTEFORCE_HAMMING, false);
    #endif

    #ifdef SAVE_SNAPSHOTS_OUTPUT
    #if defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS)
    //TODO: Cleanup before start
    mkdir(SAVE_SNAPSHOTS_OUTPUT, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    #elif defined(SL_OS_WINDOWS)
    _mkdir(SAVE_SNAPSHOTS_OUTPUT);
    #endif
    #endif

    frameCount = 0;
    _prev.points2D = SLCVVPoint2f(nFeatures);
    _prev.foundPose = false;
}

//------------------------------------------------------------------------------
void SLCVTrackerFeatures::loadModelPoints() {
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
}

//------------------------------------------------------------------------------
SLbool SLCVTrackerFeatures::track(SLCVMat imageGray,
                                  SLCVMat image,
                                  SLCVCalibration *calib,
                                  SLSceneView *sv) {
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
    Mat rvec = cv::Mat::zeros(3, 3, CV_64FC1);      // rotation matrix
    Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);      // translation matrix
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
        Mat descriptors = getDescriptors(imageGray , keypoints);
        // #####################################################################


        // Feature matching ####################################################
        vector<DMatch> matches = getFeatureMatches(descriptors);
        // #####################################################################


        // POSE calculation ####################################################
        bool useExtrinsicGuess = false;
        if (_prev.foundPose) {
            useExtrinsicGuess = true;
            rvec = _prev.rvec;
            tvec = _prev.tvec;
        }
        foundPose = calculatePose(imageGray, image, keypoints, matches, inlierMatches, points2D, rvec, tvec, useExtrinsicGuess, descriptors );
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

    // Update camera object matrix  ########################################
    if (foundPose) {
        // Converts calulated extrinsic camera components (translation & rotation) to OpenGL camera matrix
        _pose = createGLMatrix(tvec, rvec);

        // Update Scene Graph camera to display model correctly (positioning cam relative to world coordinates)
        sv->camera()->om(_pose.inverse());
    }

    #if defined(SAVE_SNAPSHOTS_OUTPUT) && (defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS) || defined(SL_OS_WINDOWS))
    // Draw matches
    if (foundPose && !inlierMatches.empty()) {
        Mat imgMatches;
        drawMatches(imageGray, keypoints, _map.frameGray, _map.keypoints, inlierMatches, imgMatches);
        imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-matching.png", imgMatches);

		//for (size_t i = 0; i < keypoints.size(); i++) {
		//	circle(image, keypoints[i].pt, 10, Scalar(0, 255, 0), 1, FILLED);
		//}
    }

    // Draw optical flow
    if (/*foundPose &&*/ _prev.points2D.size() == points2D.size() && points2D.size() > 0) {
        Mat optFlow, rgb;
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
SLCVVKeyPoint SLCVTrackerFeatures::getKeypoints(const Mat &imageGray) {
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
Mat SLCVTrackerFeatures::getDescriptors(const Mat &imageGray, SLCVVKeyPoint &keypoints) {
    Mat descriptors;
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
vector<DMatch> SLCVTrackerFeatures::getFeatureMatches(const Mat &descriptors) {
    SLfloat matchTimeMillis = SLScene::current->timeMilliSec();

    // 1. Get matches with FLANN or KNN algorithm ######################################################################################
    #if FLANN_BASED
    if(descriptors.type() != CV_32F) descriptors.convertTo(descriptors, CV_32F);
    if(_map.descriptors.type() != CV_32F) _map.descriptors.convertTo(_map.descriptors, CV_32F);

    vector<DMatch> goodMatches;
    _matcher->match(descriptors, _map.descriptors, goodMatches);
    #else
    int k = 2;
    vector<vector<DMatch>> matches;
    _matcher->knnMatch(descriptors, _map.descriptors, matches, k);

    /* Perform ratio test which determines if k matches from the knn matcher are not too similar.
     *  If the ratio of the the distance of the two matches is toward 1, the matches are near identically.
     */
    vector<DMatch> goodMatches;
    for(size_t i = 0; i < matches.size(); i++) {
        const DMatch &match1 = matches[i][0];
        const DMatch &match2 = matches[i][1];
        if (match2.distance == 0.0f || ( match1.distance / match2.distance) < minRatio)
            goodMatches.push_back(match1);
    }
    #endif

    SLScene::current->setMatchTimesMS(SLScene::current->timeMilliSec() - matchTimeMillis);
    return goodMatches;
}

//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::calculatePose(const Mat &imageGray, const Mat &imageVideo,
	vector<KeyPoint> &keypoints, vector<DMatch> &matches, vector<DMatch> &inliers,
	vector<Point2f> &inlierPoints, Mat &rvec, Mat &tvec, bool extrinsicGuess, const Mat& descriptors) {

    // RANSAC crashes if 0 points are given
    if (matches.size() == 0) return 0;

    /* Find 2D/3D correspondences
     *
     *  At the moment we are using only the two correspondences like this:
     *  KeypointsOriginal <-> KeypointsActualscene
     *
     *  Train index --> "Point" in the model
     *  Query index --> "Point" in the actual frame
     */
    vector<Point3f> modelPoints(matches.size());
    vector<Point2f> framePoints(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        modelPoints[i] = _map.model[matches[i].trainIdx];
        framePoints[i] =  keypoints[matches[i].queryIdx].pt;
    }

    // Finding PnP solution
    vector<unsigned char> inliersMask(modelPoints.size());
    bool foundPose = solvePnP(modelPoints, framePoints, extrinsicGuess, rvec, tvec, inliersMask);

    for (size_t i = 0; i < inliersMask.size(); i++) {
        size_t idx = inliersMask[i];
        inliers.push_back(matches[idx]);
        inlierPoints.push_back(framePoints[idx]);
    }

    //TODO: Move to other image write location
    #if defined(SAVE_SNAPSHOTS_OUTPUT) && (defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS) || defined(SL_OS_WINDOWS))
    // Draw reprojection --------------------------------
    if (foundPose) {
        Mat imgReprojection;
        imageGray.copyTo(imgReprojection);
        cvtColor(imgReprojection, imgReprojection, CV_GRAY2BGR);

        // Reproject the model points with the calculated POSE
        vector<Point2f> projectedPositioningKeypoints(modelPoints.size());
        cv::projectPoints(modelPoints, rvec, tvec, _calib->cameraMat(), _calib->distortion(), projectedPositioningKeypoints);

        // Calculate the sum of euclidean distances
        double sum = 0.;
        for (size_t i = 0; i < inliers.size(); i++) {
           Point2f imgPoint = inlierPoints[i];
           Point2f refPoint(modelPoints[i].x, modelPoints[i].y);
           sum += norm(Mat(imgPoint), Mat(refPoint));
        }

        // Calculate the average of the sum of euclidean distances and print out
        if(sum > 0) std::cout << "Average euclidean distance=" << sum << std::endl;

        // Draw the reprojection
        // for(Point2f it : framePoints) circle(imgReprojection, it, 1, Scalar(0, 0, 255), 1, FILLED);             // Outliers (in- and outliers)
        for(Point2f it : inlierPoints) circle(imgReprojection, it, 1, Scalar(255, 0, 0), 1, FILLED);            // Inliers
        for(size_t i = 0; i < projectedPositioningKeypoints.size(); i++) {
            circle(imgReprojection, projectedPositioningKeypoints[i], 10, Scalar(0, 255, 0), 1, FILLED);        // Model points
            putText(imgReprojection, "bp" + to_string(i), Point2f(projectedPositioningKeypoints[i].x - 12, projectedPositioningKeypoints[i].y -12),
                    FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0), 1.0);
        }

        imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-reprojection.png", imgReprojection);
    }
    #endif

    // Pose optimization
    if (foundPose && _map.keypoints.size() > 0)
        optimizePose(imageGray, imageVideo, keypoints, inliers, rvec, tvec, descriptors);

    return foundPose;
}

//-----------------------------------------------------------------------------
bool SLCVTrackerFeatures::trackWithOptFlow(SLCVMat &previousFrame, vector<Point2f> &previousPoints, SLCVMat &currentFrame, vector<Point2f> &predPoints, Mat &rvec, Mat &tvec) {
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
                status,                      // Output vector for keypoint correspondences (1 = match found)
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
bool SLCVTrackerFeatures::solvePnP(vector<Point3f> &modelPoints, vector<Point2f> &framePoints, bool guessExtrinsic, Mat &rvec, Mat &tvec, vector<unsigned char> &inliersMask) {
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
bool SLCVTrackerFeatures::optimizePose(const Mat &imageGray, const Mat &imageVideo,
	vector<KeyPoint> &keypoints, vector<DMatch> &matches, Mat &rvec, Mat &tvec, const Mat& descriptors)
{

    int patchSize = 30;
	int patchHalf = patchSize / 2;

    vector<KeyPoint> bboxModelKeypoints, bboxFrameKeypoints;
	vector<size_t> frameIndicesInsideRect;

    for (size_t i = 0; i < _map.model.size(); i++)
	{
		//only every tenth
		if (i % 10)
			continue;

        // 1. Reproject the model points with the calculated POSE
        Point3f originalModelPoint = _map.model[i];
        Point2f projectedModelPoint = backprojectPoint(originalModelPoint, rvec, tvec); // projectedModelPoints[i];

#if DRAW_ALL_MAP_PTS_ON_VIDEO
		//draw all projected map features on video stream
		circle(imageVideo, projectedModelPoint, 1, CV_RGB(255, 0, 0), 1, FILLED);
		putText(imageVideo, to_string(i), Point2f(projectedModelPoint.x - 1, projectedModelPoint.y - 1),
			FONT_HERSHEY_SIMPLEX, 0.25, CV_RGB(255, 0, 0), 1.0);
#endif

        // 2. Select only before calculated Keypoints within patch with projected "positioning" keypoint as center
        // OpenCV: Top-left origin
        int xTopLeft = projectedModelPoint.x - patchSize / 2;
        int yTopLeft = projectedModelPoint.y - patchSize / 2;
        int xDownRight = xTopLeft + patchSize;
        int yDownRight = yTopLeft + patchSize;

        // 3. Select only the image-plane keypoints within the defined patch
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

		//Was soll das???????

        //// OpenCV: Top-left origin
        //xTopLeft = originalModelPoint.x - patchSize / 2;
        //yTopLeft = originalModelPoint.y - patchSize / 2;
        //xDownRight = xTopLeft + patchSize;
        //yDownRight = yTopLeft + patchSize;
        //for (size_t j = 0; j < _map.keypoints.size(); j++) {
        //    // bbox check
        //    if (_map.keypoints[j].pt.x > xTopLeft &&
        //            _map.keypoints[j].pt.x < xDownRight &&
        //            _map.keypoints[j].pt.y > yTopLeft &&
        //            _map.keypoints[j].pt.y < yDownRight) {
        //        bboxModelKeypoints.push_back(_map.keypoints[j]);
        //    }
        //}

#if	FIND_AND_DRAW_KPTS_IN_ROI
		//draw green rectangle around every map point
		rectangle(imageVideo,
			Point2f(projectedModelPoint.x - patchHalf, projectedModelPoint.y - patchHalf),
			Point2f(projectedModelPoint.x + patchHalf, projectedModelPoint.y + patchHalf),
			CV_RGB(0, 255, 0));
		//draw key points, that lie inside this rectangle
		for( auto kPt : bboxFrameKeypoints)
			circle(imageVideo, kPt.pt, 1, CV_RGB(0, 0, 255), 1, FILLED);
#endif

//#if defined(SAVE_SNAPSHOTS_OUTPUT) && (defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS) || defined(SL_OS_WINDOWS))
//
//        // Draw the bbox in frame
//        Mat imgPoseOptimizationFrame;
//        imageGray.copyTo(imgPoseOptimizationFrame);
//        cvtColor(imgPoseOptimizationFrame, imgPoseOptimizationFrame, CV_GRAY2BGR);
//
//        circle(imgPoseOptimizationFrame, projectedModelPoint, 1, Scalar(0, 0, 255), 1, FILLED);
//        rectangle(imgPoseOptimizationFrame,
//                  Point2f(projectedModelPoint.x - patchHalf, projectedModelPoint.y - patchHalf),
//                  Point2f(projectedModelPoint.x + patchHalf, projectedModelPoint.y + patchHalf),
//                  Scalar(0, 255, 0));
//
//        // Draw the bbox in model
//        Mat imgPoseOptimizationModel;
//        _map.frameGray.copyTo(imgPoseOptimizationModel);
//        cvtColor(imgPoseOptimizationModel, imgPoseOptimizationModel, CV_GRAY2BGR);
//
//        circle(imgPoseOptimizationModel, Point2f(originalModelPoint.x, originalModelPoint.y), 1, Scalar(0, 0, 255), 1, FILLED);
//        rectangle(imgPoseOptimizationModel,
//                  Point2f(originalModelPoint.x - patchHalf, originalModelPoint.y - patchHalf),
//                  Point2f(originalModelPoint.x + patchHalf, originalModelPoint.y + patchHalf),
//                  Scalar(0, 255, 0));
//
//
//        // Perform the rematching
//        Mat bboxFrameDescriptors = getDescriptors(imageGray, bboxFrameKeypoints);
//        Mat bboxModelDescriptors = getDescriptors(imageGray, bboxModelKeypoints);
//        vector<DMatch> newMatches;
//        //_matcher->match(bboxFrameDescriptors, bboxModelDescriptors, newMatches);
//
//        // Draw the bbox matching output
//        Mat imgMatches;
//        drawMatches(imgPoseOptimizationFrame, bboxFrameKeypoints, imgPoseOptimizationModel, bboxModelKeypoints, newMatches, imgMatches);
//        imwrite(SAVE_SNAPSHOTS_OUTPUT + to_string(frameCount) + "-poseoptimization.png", imgMatches);
//
//#endif

		//4. Match the descriptors of the keypoints inside the rectangle around the projected map point
		//with the descritor of the projected map point.
		//(du musst versuchen den einzelnen descriptor des projizierten map point und die descriptoren
		// der keypoints im aktuellen frame aus den cv::Mat's zu extrahieren und einzeln an knnMatch zu übergeben.
		// Vllt. kann man auch diesen parameter "mask" in der Methode knnmatch verwenden... Weiss ich auch nicht...)
		//todo...
		//nur symbolisch, diese descriptoren müssen wir matchen
		//descriptors[i]  => descriptor des map points
		//for( size_t j : frameIndicesInsideRect )
		//	_map.descriptors[frameIndicesInsideRect[j]]
		// => descriptoren der keypoints im rechteck

		//5. Add the found match to the already found matches
		//todo...
		// matches.push_back( ... all new ones );

        bboxModelKeypoints.clear();
        bboxFrameKeypoints.clear();
		frameIndicesInsideRect.clear();
    }

	//6. draw all matches with cv::drawMatches as before in line 217

    //7. Finding PnP solution - again
	//vector<Point3f> modelPoints(matches.size());
	//vector<Point2f> framePoints(matches.size());
	//for (size_t i = 0; i < matches.size(); i++) {
	//	modelPoints[i] = _map.model[matches[i].trainIdx];
	//	framePoints[i] = keypoints[matches[i].queryIdx].pt;
	//}
	//cv::solvePnP(modelPoints, framePoints, _calib->cameraMat(),
	//	_calib->distortion(), rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

    return 0;
}

Mat getRtMatrix( const cv::Mat &rvec, const cv::Mat &tvec)
{
    Mat rvecMat = cv::Mat::zeros(3, 3, CV_64FC1);
    Rodrigues(rvec, rvecMat);

    // Rotation-Translation Matrix Definition
    Mat rtMatrix = cv::Mat::zeros(3, 4, CV_64FC1);
    rtMatrix.at<double>(0,0) = rvecMat.at<double>(0,0);
    rtMatrix.at<double>(0,1) = rvecMat.at<double>(0,1);
    rtMatrix.at<double>(0,2) = rvecMat.at<double>(0,2);
    rtMatrix.at<double>(1,0) = rvecMat.at<double>(1,0);
    rtMatrix.at<double>(1,1) = rvecMat.at<double>(1,1);
    rtMatrix.at<double>(1,2) = rvecMat.at<double>(1,2);
    rtMatrix.at<double>(2,0) = rvecMat.at<double>(2,0);
    rtMatrix.at<double>(2,1) = rvecMat.at<double>(2,1);
    rtMatrix.at<double>(2,2) = rvecMat.at<double>(2,2);
    rtMatrix.at<double>(0,3) = tvec.at<double>(0);
    rtMatrix.at<double>(1,3) = tvec.at<double>(1);
    rtMatrix.at<double>(2,3) = tvec.at<double>(2);

    return rtMatrix;
}


//-----------------------------------------------------------------------------
Point2f SLCVTrackerFeatures::backprojectPoint(Point3f pointToProject, const Mat &rvec, const Mat &tvec) {
    // 3D point vector [x y z 1]'
    cv::Mat point3d_vec = cv::Mat(4, 1, CV_64FC1);
    point3d_vec.at<double>(0) = pointToProject.x;
    point3d_vec.at<double>(1) = pointToProject.y;
    point3d_vec.at<double>(2) = pointToProject.z;
    point3d_vec.at<double>(3) = 1;

    // 2D point vector [u v 1]'
    cv::Mat point2d_vec = cv::Mat(4, 1, CV_64FC1);
    point2d_vec = _calib->cameraMat() * getRtMatrix(rvec, tvec) * point3d_vec;

    // Normalization of [u v]'
    cv::Point2f point2d;
    point2d.x = (float)(point2d_vec.at<double>(0) / point2d_vec.at<double>(2));
    point2d.y = (float)(point2d_vec.at<double>(1) / point2d_vec.at<double>(2));

    return point2d;
}
