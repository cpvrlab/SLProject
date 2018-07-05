//#############################################################################
//  File:      SLCVTrackedFeatures.cpp
//  Author:    Pascal Zingg, Timon Tschanz, Marcus Hudritsch
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <SLApplication.h>
#include <SLSceneView.h>
#include <SLCVFeatureManager.h>
#include <SLCVTrackedFeatures.h>

#if defined(SL_OS_WINDOWS)
#include <direct.h>
#endif

using namespace cv;

//-----------------------------------------------------------------------------
// Globals for benchmarking
int     frames_with_pose = 0;
float   sum_matches = 0;
float   sum_inlier_matches = 0;
float   sum_allmatches_to_inliers = 0.0f;
double  sum_reprojection_error = 0.0;
float   sum_poseopt_difference = 0.0f;
double  translationError = 0;
double  rotationError = 0;
int     frames_since_posefound = 0;

//-----------------------------------------------------------------------------
SLCVTrackedFeatures::SLCVTrackedFeatures(SLNode *node,
                                         SLstring markerFilename) :
                     SLCVTracked(node)
{
    // To match the binary features, we are matching each descriptor in reference with each
    // descriptor in the current frame. The smaller the hamming distance the better the match
    // Hamming distance <-> XOR sum
    _matcher = BFMatcher::create(BFMatcher::BRUTEFORCE_HAMMING, false);

    // Initialize some member variables on startup to prevent uncontrolled behaviour
    _currentFrame.foundPose = false;
    _prevFrame.foundPose = false;
    _currentFrame.reprojectionError = 0.0f;
    _prevFrame.inlierPoints2D = SLCVVPoint2f(nFeatures);
    _forceRelocation = false;
    _frameCount = 0;

    loadMarker(markerFilename);

    // Create directory for debug output if flag is set
    #ifdef SL_DEBUG_OUTPUT_PATH
        #if defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS)
            mkdir(SL_DEBUG_OUTPUT_PATH, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        #elif defined(SL_OS_WINDOWS)
            _mkdir(SL_SAVE_SNAPSHOTS_OUTPUT);
        #else
            #undef SL_SAVE_SNAPSHOTS_OUTPUT
        #endif
    #endif
}

//-----------------------------------------------------------------------------
//! Show statistics if program terminates
SLCVTrackedFeatures::~SLCVTrackedFeatures()
{
    #if SL_DO_FEATURE_BENCHMARKING
    SL_LOG(" \n");
    SL_LOG(" \n");
    SL_LOG("------------------------------------------------------------------\n");
    SL_LOG("SLCVTrackedFeatures statistics \n");
    SL_LOG("------------------------------------------------------------------\n");
    SL_LOG("Avg frame rate                                   : %f FPS\n", SLApplication::scene->frameTimesMS().average());
    SL_LOG("Avg calculation time per frame                   : %f ms\n", SLApplication::scene->trackingTimesMS().average());
    SL_LOG(" \n");
    SL_LOG("Settings for Pose estimation: ------------------------------------\n");
    SL_LOG("Features                                         : %d\n", nFeatures);
    SL_LOG("Minimal ratio for 2 best matches                 : %f\n", minRatio);
    SL_LOG("RANSAC iterations                                : %d\n", iterations);
    SL_LOG("RANSAC mean reprojection error                   : %f\n", reprojection_error);
    SL_LOG("RANSAC confidence                                : %d\n", confidence);
    SL_LOG("Repose frequency                                 : Each %d point\n", reposeFrequency);
    SL_LOG("Initial patch size for Pose optimization         : %d pixels\n", initialPatchSize);
    SL_LOG("Maximal patch size for Pose optimization         : %d pixels\n", maxPatchSize);
    SL_LOG(" \n");
    SL_LOG("Pose information: ------------------------------------------------\n");
    SL_LOG("Avg allmatches to inliers proposition            : %f\n", sum_allmatches_to_inliers / _frameCount);
    SL_LOG("Avg reprojection error (only if POSE)            : %f\n", sum_reprojection_error / frames_with_pose);
    SL_LOG("Pose found                                       : %d of %d frames\n", frames_with_pose, _frameCount);
    SL_LOG("Avg matches                                      : %f\n", sum_matches / frames_with_pose);
    SL_LOG("Avg inlier matches                               : %f\n", sum_inlier_matches / frames_with_pose);
    SL_LOG("Avg more matches with Pose optimization          : %f\n", sum_poseopt_difference / frames_with_pose);

    // Only used for testing with slight movements
    //SL_LOG("Avg Rotation error                               : %f deg\n", rotationError / frames_with_pose);
    //SL_LOG("Avg Translation error                            : %f px\n", translationError / frames_with_pose);
    #endif //SL_DO_FEATURE_BENCHMARKING
}

//-----------------------------------------------------------------------------
//! Loads the marker image form the filesystem
void SLCVTrackedFeatures::loadMarker(string markerFilename)
{
    // Read reference marker
    // (The images source is deallocated by SLScene::unInit)
    SLGLTexture* markerTexture = new SLGLTexture(markerFilename);
    SLCVImage* img = markerTexture->images()[0];

    cvtColor(img->cvMat(), _marker.imageGray, CV_RGB2GRAY);

    cv::rotate(_marker.imageGray, _marker.imageGray, ROTATE_180);
    cv::flip(_marker.imageGray, _marker.imageGray, 1);
}
//-----------------------------------------------------------------------------
/*! Prepares the reference tracker:
1. Detect and describe the keypoints on the reference image
2. Set up 3D points with predefined scaling
3. Perform optional drawing operations on image
*/
void SLCVTrackedFeatures::initFeaturesOnMarker()
{
    assert(!_marker.imageGray.empty() && "Grayscale image is empty!");

    // Clear previous initializations
    _marker.keypoints2D.clear();
    _marker.keypoints3D.clear();
    _marker.descriptors.release();

    // Detect and compute features in marker image
    _featureManager.detectAndDescribe(_marker.imageGray,
                                      _marker.keypoints2D,
                                      _marker.descriptors);
    // Scaling factor for the 3D point.
    // Width of image is A4 size in image, 297mm is the real A4 height
    SLfloat pixelPerMM = (SLfloat)_marker.imageGray.cols / 297.0f;

    // Calculate 3D-Points based on the detected features
    for (unsigned int i = 0; i < _marker.keypoints2D.size(); i++)
    {
        // 2D location in image
        SLCVPoint2f refImageKeypoint = _marker.keypoints2D[i].pt;

        // SLCVPoint scaling
        refImageKeypoint /= pixelPerMM;

        // Here we can use Z=0 because the tracker is planar
        _marker.keypoints3D.push_back(Point3f(refImageKeypoint.x,
                                              refImageKeypoint.y,
                                              0.0f));
    }

    // Draw points and indices which should be reprojected later.
    // Only a few (defined with reposeFrequency)
    // points are used for the reprojection.
    #if defined(SL_DEBUG_OUTPUT_PATH) || SL_DRAW_REPROJECTION_ERROR
    _marker.imageGray.copyTo(_marker.imageDrawing);
    cvtColor(_marker.imageDrawing, _marker.imageDrawing, CV_GRAY2BGR);

    for (size_t i = 0; i < _marker.keypoints3D.size(); i++)
    {   if (i % reposeFrequency)
            continue;

        SLCVPoint2f originalModelPoint = _marker.keypoints2D[i].pt;

        circle(_marker.imageDrawing,
               originalModelPoint,
               1,
               CV_RGB(255, 0, 0),
               1,
               FILLED);

        putText(_marker.imageDrawing,
                to_string(i),
                SLCVPoint2f(originalModelPoint.x - 1,
                            originalModelPoint.y - 1),
                FONT_HERSHEY_SIMPLEX,
                0.25,
                CV_RGB(255, 0, 0),
                1);
    }
    #endif
}

//-----------------------------------------------------------------------------
//! Setter of the feature detector & descriptor type
void SLCVTrackedFeatures::type(SLCVDetectDescribeType ddType)
{
    _featureManager.createDetectorDescriptor(ddType);

    _currentFrame.foundPose = false;
    _prevFrame.foundPose = false;
    _currentFrame.reprojectionError = 0.0f;

    // Set the frame counter to 0 to reinitialize in track
    _frameCount = 0;
}
//-----------------------------------------------------------------------------
/*! The main part of this tracker is to calculate a correct Pose.
@param imageGray Current grayscale frame
@param image Current RGB frame
@param calib Calibration information
@param drawDetection Flag if the detected features should be drawn
@param sv The current scene view
@return So far allways false
*/
SLbool SLCVTrackedFeatures::track(SLCVMat imageGray,
                                  SLCVMat image,
                                  SLCVCalibration *calib,
                                  SLbool drawDetection,
                                  SLSceneView *sv)
{
    assert(!image.empty() && "Image is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(!_marker.imageGray.empty());
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");

    // Initialize reference points if program just started
    if (_frameCount == 0)
    {   _calib = calib;
        initFeaturesOnMarker();
    }

    // Copy image matrix into current frame data
    _currentFrame.image = image;
    _currentFrame.imageGray = imageGray;

    // Determine if relocation or feature tracking should be performed
    bool relocationNeeded = _forceRelocation
                            || !_prevFrame.foundPose
                            || _prevFrame.inlierMatches.size() < 100
                            || frames_since_posefound < 3;

    // If relocation condition meets, calculate the Pose with feature detection, otherwise
    // track the previous determined features
    if (relocationNeeded)
        relocate();
    else 
        tracking();

    // Update the camera according to the new Pose
    updateSceneCamera(sv);

    // Perform OpenCV drawning if flags are set (see SLCVTrackedFeatures.h)
    drawDebugInformation(drawDetection);

    // Prepare next frame and transfer necessary data
    transferFrameData();

    _frameCount++;

    return false;
}

//-----------------------------------------------------------------------------
/*! If relocation should be done, the following steps are necessary:
1. Detect keypoints
2. Describe keypoints (Binary descriptors)
3. Match keypoints in current frame and the reference tracker
4. Try to calculate new Pose with Perspective-n-SLCVPoint algorithm
*/
void SLCVTrackedFeatures::relocate()
{
    _isTracking = false;
    detectKeypointsAndDescriptors();
    _currentFrame.matches = getFeatureMatches();
    _currentFrame.foundPose = calculatePose();

    // Zero time keeping on the tracking branch
    SLScene* s = SLApplication::scene;
    s->optFlowTimesMS().set(0);
}

//-----------------------------------------------------------------------------
/*! To track the already detected keypoints after a sucessful pose estimation,
we track the features with optical flow
*/
void SLCVTrackedFeatures::tracking()
{
    _isTracking = true;
    _currentFrame.foundPose = trackWithOptFlow(_prevFrame.rvec, _prevFrame.tvec);

    // Zero time keeping on the relocation branch
    SLScene* s = SLApplication::scene;
    s->detectTimesMS().set(0);
    s->matchTimesMS().set(0);
}

//-----------------------------------------------------------------------------
/*! Visualizes the following parts of the whole Pose estimation:
- Keypoints
- Inlier matches
- Optical Flow (Small arrows that show how keypoints moved between frames)
- Reprojection with the calculated Pose
*/
void SLCVTrackedFeatures::drawDebugInformation(SLbool drawDetection)
{
    if (drawDetection)
    {
        for (size_t i=0; i < _currentFrame.inlierPoints2D.size(); i++)
            circle(_currentFrame.image,
                   _currentFrame.inlierPoints2D[i],
                   3,
                   Scalar(0, 0, 255));
    }

    #if SL_DRAW_REPROJECTION_POINTS
    SLCVMat imgReprojection = _currentFrame.image;
    #elif defined(SL_SAVE_SNAPSHOTS_OUTPUT)
    SLCVMat imgReprojection;
    _currentFrame.image.copyTo(imgReprojection);
    #endif

    #if SL_DRAW_REPROJECTION_POINTS || defined(SL_DEBUG_OUTPUT_PATH)
    if (_currentFrame.inlierMatches.size() > 0)
    {
        SLCVVPoint2f projectedPoints(_marker.keypoints3D.size());

        cv::projectPoints(_marker.keypoints3D,
                          _currentFrame.rvec,
                          _currentFrame.tvec,
                          _calib->cameraMat(),
                          _calib->distortion(),
                          projectedPoints);

        for (size_t i = 0; i < _marker.keypoints3D.size(); i++)
        {
            if (i % reposeFrequency) continue;

            SLCVPoint2f projectedModelPoint = projectedPoints[i];
            SLCVPoint2f keypointForPose = _currentFrame.keypoints[_currentFrame.inlierMatches.back().queryIdx].pt;

            // draw all projected map features and the original keypoint on video stream
            circle(imgReprojection,
                   projectedModelPoint,
                   2,
                   CV_RGB(255, 0, 0),
                   1,
                   FILLED);

            circle(imgReprojection,
                   keypointForPose,
                   5,
                   CV_RGB(0, 0, 255),
                   1,
                   FILLED);

            //draw the point index and reprojection error
            putText(imgReprojection,
                    to_string(i),
                    SLCVPoint2f(projectedModelPoint.x - 2, projectedModelPoint.y - 5),
                    FONT_HERSHEY_SIMPLEX,
                    0.3,
                    CV_RGB(255, 0, 0),
                    1.0);
        }
    }
    #endif

    #if defined(SL_DEBUG_OUTPUT_PATH)
    // Draw reprojection
    SLCVMat imgOut;
    drawMatches(imgReprojection,
                SLCVVKeyPoint(),
                _marker.imageDrawing,
                SLCVVKeyPoint(),
                SLCVVDMatch(),
                imgOut,
                CV_RGB(255,0,0),
                CV_RGB(255,0,0));

    imwrite(SL_DEBUG_OUTPUT_PATH + to_string(_frameCount) + "_reprojection.png",
            imgOut);

    // Draw keypoints
    if (!_currentFrame.keypoints.empty())
    {   SLCVMat imgKeypoints;
        drawKeypoints(_currentFrame.imageGray,
                      _currentFrame.keypoints,
                      imgKeypoints);

        imwrite(SL_DEBUG_OUTPUT_PATH + to_string(_frameCount) + "_keypoints.png",
                imgKeypoints);
    }

    for (size_t i=0; i < _currentFrame.inlierPoints2D.size(); i++)
        circle(_currentFrame.image,
               _currentFrame.inlierPoints2D[i],
               2,
               Scalar(0, 255, 0));

    // Draw matches
    if (!_currentFrame.inlierMatches.empty())
    {   SLCVMat imgMatches;
        drawMatches(_currentFrame.imageGray,
                    _currentFrame.keypoints,
                    _marker.imageGray,
                    _marker.keypoints2D,
                    _currentFrame.inlierMatches,
                    imgMatches,
                    CV_RGB(255,0,0),
                    CV_RGB(255,0,0));

        imwrite(SL_DEBUG_OUTPUT_PATH + to_string(_frameCount) + "_matching.png",
                imgMatches);
    }

    // Draw optical flow
    if (_isTracking)
    {
        SLCVMat optFlow, rgb;
        _currentFrame.imageGray.copyTo(optFlow);
        cvtColor(optFlow, rgb, CV_GRAY2BGR);
        for (size_t i = 0; i < _currentFrame.inlierPoints2D.size(); i++)
            cv::arrowedLine(rgb,
                            _prevFrame.inlierPoints2D[i],
                            _currentFrame.inlierPoints2D[i],
                            Scalar(0, 255, 0),
                            1,
                            LINE_8, 0, 0.2);

        imwrite(SL_DEBUG_OUTPUT_PATH + to_string(_frameCount) + "-optflow.png", rgb);
    }
    #endif
}

//-----------------------------------------------------------------------------
//! Updates the scenegraph camera with the new pose
void SLCVTrackedFeatures::updateSceneCamera(SLSceneView *sv)
{
    if (_currentFrame.foundPose)
    {
        _objectViewMat = createGLMatrix(_currentFrame.tvec, _currentFrame.rvec);

        // Update Scene Graph camera to display model correctly
        // (positioning cam relative to world coordinates)
        sv->camera()->om(_objectViewMat.inverted());

        frames_with_pose++;
    }

    // Only draw tower if last 2 pose calculations were correct
    if (_prevFrame.foundPose && !_currentFrame.foundPose)
    {   sv->drawBits()->on(SL_DB_HIDDEN);
        frames_since_posefound = 0;
    } else if (_currentFrame.foundPose)
    {   if (frames_since_posefound == 5)
            sv->drawBits()->off(SL_DB_HIDDEN);
        frames_since_posefound++;
    }
}

//-----------------------------------------------------------------------------
/*! Copies the current frame data to the previous frame data struct for the
next frame handling.
TODO: more elegant way to do this whole copy action
*/
void SLCVTrackedFeatures::transferFrameData()
{
    _currentFrame.imageGray.copyTo(_prevFrame.imageGray);
    _currentFrame.image.copyTo(_prevFrame.image);
    _currentFrame.rvec.copyTo(_prevFrame.rvec);
    _currentFrame.tvec.copyTo(_prevFrame.tvec);

    _prevFrame.reprojectionError = _currentFrame.reprojectionError;
    _prevFrame.foundPose         = _currentFrame.foundPose;
    _prevFrame.inlierPoints3D    = _currentFrame.inlierPoints3D;
    _prevFrame.inlierPoints2D    = _currentFrame.inlierPoints2D;

    if (_currentFrame.inlierMatches.size() > 0)
        _prevFrame.inlierMatches = _currentFrame.inlierMatches;

    _currentFrame.keypoints.clear();
    _currentFrame.matches.clear();
    _currentFrame.inlierMatches.clear();
    _currentFrame.inlierPoints2D.clear();
    _currentFrame.inlierPoints3D.clear();
    _currentFrame.reprojectionError = 0;

    _currentFrame.useExtrinsicGuess = _prevFrame.foundPose;

    if (_prevFrame.foundPose)
    {   _currentFrame.rvec = _prevFrame.rvec;
        _currentFrame.tvec = _prevFrame.tvec;
    } else {
        _currentFrame.rvec = SLCVMat::zeros(3, 1, CV_64FC1);
        _currentFrame.tvec = SLCVMat::zeros(3, 1, CV_64FC1);
    }
}

//-----------------------------------------------------------------------------
/*! Get keypoints and descriptors in one step. This is a more efficient way
since we have to build the scaling pyramide only once. If we detect and
describe seperatly, it will lead in two scaling pyramids and is therefore less
meaningful.
*/
void SLCVTrackedFeatures::detectKeypointsAndDescriptors()
{
    SLScene* s = SLApplication::scene;
    SLfloat startMS = s->timeMilliSec();

    _featureManager.detectAndDescribe(_currentFrame.imageGray,
                                      _currentFrame.keypoints,
                                      _currentFrame.descriptors);

    s->detectTimesMS().set(s->timeMilliSec()-startMS);
}

//-----------------------------------------------------------------------------
/*! Get matching features with the defined feature matcher. Since we are using
the k-next-neighbour matcher, we check if the best and second best match are
not too identical with the so called ratio test.
@return Vector of found matches
*/
SLCVVDMatch SLCVTrackedFeatures::getFeatureMatches()
{
    SLScene* s = SLApplication::scene;
    SLfloat startMS = s->timeMilliSec();

    int k = 2;
    SLCVVVDMatch matches;
    _matcher->knnMatch(_currentFrame.descriptors, _marker.descriptors, matches, k);

    // Perform ratio test which determines if k matches from the knn matcher
    // are not too similar. If the ratio of the the distance of the two
    // matches is toward 1, the matches are near identically.
    SLCVVDMatch goodMatches;
    for (size_t i = 0; i < matches.size(); i++)
    {   const DMatch& match1 = matches[i][0];
        const DMatch& match2 = matches[i][1];
        if (match2.distance == 0.0f ||
            (match1.distance / match2.distance) < minRatio)
            goodMatches.push_back(match1);
    }

    s->matchTimesMS().set(s->timeMilliSec() - startMS);
    return goodMatches;
}

//-----------------------------------------------------------------------------
/*! This method does the most important work of the whole pipeline:

RANSAC: We execute first RANSAC to eliminate wrong feature correspondences
(outliers) and only use the correct ones (inliers) for PnP solving
(https://en.wikipedia.org/wiki/Perspective-n-SLCVPoint).\n
\n
Methods of solvePnP:
- P3P: If we have 3 Points given, we have the minimal form of the PnP problem.
  We can treat the points as a triangle definition ABC. We have 3 corner points
  and 3 angles. Because we get many soulutions for the equation, there will be a
  fourth point which removes the ambiguity. Therefore the OpenCV implementation
  requires 4 points to use this method.
- EPNP: This method is used if there are n >= 4 points. The reference points are
  expressed as 4 virtual control points. The coordinates of these points are the
  unknowns for the equtation.
- ITERATIVE: Calculates pose using the DLT (Direct Linear Transform) method.
  If there is a homography will be much easier and no DLT will be used. Otherwise
  we are using the DLT and make a Levenberg-Marquardt optimization. The latter
  helps to decrease the reprojection error which is the sum of the squared
  distances between the image and object points.\n
\n
Overall Steps:
1. Call RANSAC with EPNP: The RANdom Sample Consensus algorithm is called to
remove "wrong" point correspondences which makes the solvePnP more robust.
The so called inliers are used for calculation, wrong correspondences (outliers)
will be ignored. Therefore the method below will first run a solvePnP with the
EPNP method and returns the reprojection error. EPNP works like the following:
    - Choose the 4 control pints: C0 as centroid of reference points, \n
      C1, C2 and C3 from PCA of the reference points
    - Compute barycentric coordinates with the control points
    - Derivate the image reference points with the above
2. Optimize inlier matches
3. Call PnP ITERATIVE: General problem: We have a calibrated cam and sets of
corresponding 2D/3D points. We will calculate the rotation and translation in
respect to world coordinates.
    - If for no extrinsic guess, begin with computation
    - If planarity is detected, find homography, otherwise use DLT method
    - After sucessful determination of a pose, optimize it with \n
      Levenberg-Marquardt (iterative part)

@return True if the pose was found.
 */
bool SLCVTrackedFeatures::calculatePose()
{
    // solvePnP crashes if less than 5 points are given
    if (_currentFrame.matches.size() < 10) return false;

    SLScene* s = SLApplication::scene;
    SLfloat startMS = s->timeMilliSec();

    // Find 2D/3D correspondences
    // At the moment we are using only the two correspondences like this:
    // KeypointsOriginal <-> KeypointsActualscene
    // Train index --> "SLCVPoint" in the model
    // Query index --> "SLCVPoint" in the actual frame

    SLCVVPoint3f modelPoints(_currentFrame.matches.size());
    SLCVVPoint2f framePoints(_currentFrame.matches.size());

    for (size_t i = 0; i < _currentFrame.matches.size(); i++)
    {   modelPoints[i] = _marker.keypoints3D[_currentFrame.matches[i].trainIdx];
        framePoints[i] = _currentFrame.keypoints[_currentFrame.matches[i].queryIdx].pt;
    }

    SLVuchar inliersMask(modelPoints.size());

    //////////////////////
    // 1. RANSAC with EPnP
    //////////////////////

    bool foundPose = cv::solvePnPRansac(modelPoints,
                                        framePoints,
                                        _calib->cameraMat(),
                                        _calib->distortion(),
                                        _currentFrame.rvec, _currentFrame.tvec,
                                        _currentFrame.useExtrinsicGuess,
                                        iterations,
                                        reprojection_error,
                                        confidence,
                                        inliersMask,
                                        SOLVEPNP_EPNP);

    // Get matches with help of inlier indices
    for (size_t i = 0; i < inliersMask.size(); i++)
    {
        size_t idx = inliersMask[i];
        _currentFrame.inlierMatches.push_back(_currentFrame.matches[idx]);
        _currentFrame.inlierPoints2D.push_back(framePoints[idx]);
        _currentFrame.inlierPoints3D.push_back(modelPoints[idx]);
    }

    // Pose optimization
    if (foundPose)
    {
        //SLfloat matchesBefore = (SLfloat)_currentFrame.inlierMatches.size();


        /////////////////////
        // 2. Optimze Matches
        /////////////////////

        optimizeMatches();


        ///////////////////////
        // 3. solvePnP Iterativ
        ///////////////////////

        foundPose = cv::solvePnP(_currentFrame.inlierPoints3D,
                                 _currentFrame.inlierPoints2D,
                                 _calib->cameraMat(),
                                 _calib->distortion(),
                                 _currentFrame.rvec,
                                 _currentFrame.tvec,
                                 true,
                                 SOLVEPNP_ITERATIVE);

        #if SL_DO_FEATURE_BENCHMARKING
        sum_matches                 += _currentFrame.matches.size();
        sum_inlier_matches          += _currentFrame.inlierMatches.size();
        sum_allmatches_to_inliers   += _currentFrame.inlierMatches.size() /
                                       _currentFrame.matches.size();
        sum_poseopt_difference      += _currentFrame.inlierMatches.size() /
                                    matchesBefore;
        #endif
    }

    s->poseTimesMS().set(s->timeMilliSec() - startMS);

    return foundPose;
}

//-----------------------------------------------------------------------------
/*! To get more matches with the calculated pose, we reproject the reference
points to our current frame. Within a predefined patch, we try to rematch not
matched features with the reprojected point. If not possible, we increase the
patch size until we found a match for the point or we reach a threshold.
*/
void SLCVTrackedFeatures::optimizeMatches()
{
    SLfloat reprojectionError = 0;

    // 1. Reproject the model points with the calculated POSE
    SLCVVPoint2f projectedPoints(_marker.keypoints3D.size());
    cv::projectPoints(_marker.keypoints3D,
                      _currentFrame.rvec,
                      _currentFrame.tvec,
                      _calib->cameraMat(),
                      _calib->distortion(),
                      projectedPoints);

    SLCVVKeyPoint bboxFrameKeypoints;
    SLVsize_t frameIndicesInsideRect;

    for (size_t i = 0; i < _marker.keypoints3D.size(); i++)
    {
        //only every reposeFrequency
        if (i % reposeFrequency)
            continue;

        // Check if this point has a match inside matches, continue if so
        SLint alreadyMatched = 0;
        for (size_t j = 0; j < _currentFrame.inlierMatches.size(); j++)
        {   if (_currentFrame.inlierMatches[j].trainIdx == i)
                alreadyMatched++;
        }

        if (alreadyMatched > 0) continue;

        // Get the corresponding projected point of the actual (i) modelpoint
        SLCVPoint2f projectedModelPoint = projectedPoints[i];
        SLCVVDMatch newMatches;

        SLint patchSize = initialPatchSize;

        // Adaptive patch size
        while (newMatches.size() == 0 && patchSize <= maxPatchSize)
        {
            // Increase matches by even number
            patchSize += 2;
            newMatches.clear();
            bboxFrameKeypoints.clear();
            frameIndicesInsideRect.clear();

            // 2. Select only before calculated Keypoints within patch
            // with projected "positioning" keypoint as center
            // OpenCV: Top-left origin
            SLint xTopLeft = (SLint)(projectedModelPoint.x - patchSize / 2);
            SLint yTopLeft = (SLint)(projectedModelPoint.y - patchSize / 2);
            SLint xDownRight = xTopLeft + patchSize;
            SLint yDownRight = yTopLeft + patchSize;

            for (size_t j = 0; j < _currentFrame.keypoints.size(); j++)
            {   // bbox check
                if (_currentFrame.keypoints[j].pt.x > xTopLeft &&
                    _currentFrame.keypoints[j].pt.x < xDownRight &&
                    _currentFrame.keypoints[j].pt.y > yTopLeft &&
                    _currentFrame.keypoints[j].pt.y < yDownRight)
                {   bboxFrameKeypoints.push_back(_currentFrame.keypoints[j]);
                    frameIndicesInsideRect.push_back(j);
                }
            }

            // 3. SLCVMatch the descriptors of the keypoints inside
            // the rectangle around the projected map point
            // with the descritor of the projected map point.

            // This is our descriptor for the model point i
            SLCVMat modelPointDescriptor = _marker.descriptors.row((SLint)i);

            // We extract the descriptors which belong to the keypoints
            // inside the rectangle around the projected map point
            SLCVMat bboxPointsDescriptors;
            for (size_t j : frameIndicesInsideRect)
                bboxPointsDescriptors.push_back(_currentFrame.descriptors.row((SLint)j));

            // 4. Match the frame keypoints inside the rectangle with the projected model point
            _matcher->match(bboxPointsDescriptors, modelPointDescriptor, newMatches);
        }

        if (newMatches.size() > 0)
        {   for (size_t j = 0; j < frameIndicesInsideRect.size(); j++)
            {   newMatches[j].trainIdx = (int)i;
                newMatches[j].queryIdx = (int)frameIndicesInsideRect[j];
            }

            // 5. Only add the best new match to matches vector
            SLCVDMatch bestNewMatch;
            bestNewMatch.distance = 0;

            for (SLCVDMatch newMatch : newMatches)
                if (bestNewMatch.distance < newMatch.distance)
                    bestNewMatch = newMatch;

            // 6. Only add the best new match to matches vector
            _currentFrame.inlierMatches.push_back(bestNewMatch);
        }

        // Get the keypoint which was used for pose estimation
        SLCVPoint2f keypointForPose = _currentFrame.keypoints[_currentFrame.inlierMatches.back().queryIdx].pt;
        reprojectionError += (float)norm(SLCVMat(projectedModelPoint),
                                         SLCVMat(keypointForPose));


        #if SL_DRAW_PATCHES
        //draw green rectangle around every map point
        rectangle(_currentFrame.image,
                  Point2f(projectedModelPoint.x - patchSize / 2, projectedModelPoint.y - patchSize / 2),
                  Point2f(projectedModelPoint.x + patchSize / 2, projectedModelPoint.y + patchSize / 2),
                  CV_RGB(0, 255, 0));

        //draw key points, that lie inside this rectangle
        for (auto kPt : bboxFrameKeypoints)
            circle(_currentFrame.image,
                   kPt.pt,
                   1,
                   CV_RGB(0, 0, 255),
                   1,
                   FILLED);
        #endif
    }

    #if SL_DO_FEATURE_BENCHMARKING
    sum_reprojection_error += reprojectionError / _marker.keypoints3D.size();

    SLCVMat prevRmat, currRmat;
    if (_prevFrame.foundPose)
    {
        Rodrigues(_prevFrame.rvec, prevRmat);
        Rodrigues(_currentFrame.rvec, currRmat);
        double rotationError_rad = acos((trace(prevRmat * currRmat).val[0] - 1.0) / 2.0);
        rotationError += rotationError_rad * 180 / 3.14;
        translationError += cv::norm(_prevFrame.tvec, _currentFrame.tvec);
    }
    #endif

    #if SL_DRAW_REPROJECTION_ERROR
    // Draw the projection error for the current frame
    putText(_currentFrame.image,
            "Reprojection error: " + to_string(reprojectionError / _marker.keypoints3D.size()),
            Point2f(20, 20),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            CV_RGB(255, 0, 0),
            2.0);
    #endif

    // Optimize POSE
    vector<Point3f> modelPoints = vector<Point3f>(_currentFrame.inlierMatches.size());
    vector<Point2f> framePoints = vector<Point2f>(_currentFrame.inlierMatches.size());
    for (size_t i = 0; i < _currentFrame.inlierMatches.size(); i++)
    {   modelPoints[i] = _marker.keypoints3D[_currentFrame.inlierMatches[i].trainIdx];
        framePoints[i] = _currentFrame.keypoints[_currentFrame.inlierMatches[i].queryIdx].pt;
    }

    if (modelPoints.size() == 0) return;
    _currentFrame.inlierPoints3D = modelPoints;
    _currentFrame.inlierPoints2D = framePoints;
}

//-----------------------------------------------------------------------------
/*! Tracks the features with Optical Flow (Lucas Kanade). This will only try to
predict the new location of keypoints. If they were found, we perform a
solvePnP to get the new Pose from feature tracking. The method performs tests
if the Pose is good enough (not too much difference between previous and new
Pose).
@param rvec  Rotation vector (will be used for extrinsic guess)
@param tvec  Translation vector (will be used for extrinsic guess)
@return      True if Pose found, false otherwise
*/
bool SLCVTrackedFeatures::trackWithOptFlow(SLCVMat rvec, SLCVMat tvec)
{
    if (_prevFrame.inlierPoints2D.size() < 4) return false;

    SLScene* s = SLApplication::scene;
    SLfloat startMS = s->timeMilliSec();

    SLVuchar status;
    SLVfloat err;
    SLCVSize winSize(15, 15);

    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                              10,    // terminate after this many iterations, or
                              0.03); // when the search window moves by less than this

    // Find closest possible feature points based on optical flow
    SLCVVPoint2f pred2DPoints(_prevFrame.inlierPoints2D.size());

    cv::calcOpticalFlowPyrLK(
        _prevFrame.imageGray,   // Previous frame
        _currentFrame.imageGray,// Current frame
        _prevFrame.inlierPoints2D,// Previous and current keypoints coordinates.The latter will be
        pred2DPoints,           // expanded if more good coordinates are detected during OptFlow
        status,                 // Output vector for keypoint correspondences (1 = match found)
        err,                    // Error size for each flow
        winSize,                // Search window for each pyramid level
        3,                      // Max levels of pyramid creation
        criteria,               // Configuration from above
        0,                      // Additional flags
        0.001);                 // Minimal Eigen threshold

    // Only use points which are not wrong in any way during the optical flow calculation
    SLCVVPoint2f frame2DPoints;
    SLCVVPoint3f model3DPoints;
    for (size_t i = 0; i < status.size(); i++)
    {   if (status[i])
        {   frame2DPoints.push_back(pred2DPoints[i]);
            //Original code from Zingg/Tschanz got zero size vector
            //model3DPoints.push_back(_currentFrameFrame.inlierPoints3D[i]);
            model3DPoints.push_back(_prevFrame.inlierPoints3D[i]);
        }
    }

    s->optFlowTimesMS().set(s->timeMilliSec() - startMS);

    _currentFrame.inlierPoints2D = frame2DPoints;
    _currentFrame.inlierPoints3D = model3DPoints;

    if (_currentFrame.inlierPoints2D.size() < _prevFrame.inlierPoints2D.size() * 0.75)
        return false;

    /////////////////////
    // Pose Estimation //
    /////////////////////

    startMS = s->timeMilliSec();

    bool foundPose = cv::solvePnP(model3DPoints,
                                  frame2DPoints,
                                  _calib->cameraMat(),
                                  _calib->distortion(),
                                  rvec, tvec,
                                  true);
    bool poseValid = true;

    if (foundPose)
    {   for (int i = 0; i < tvec.cols; i++)
        {   if (abs(tvec.at<double>(i, 0) - tvec.at<double>(i, 0)) > abs(tvec.at<double>(i, 0)) * 0.2)
            {   cout << "translation too large" << endl;
                poseValid = false;
            }
        }
        for (int i = 0; i < rvec.cols; i++)
        {   if (abs(rvec.at<double>(i, 0) - rvec.at<double>(i, 0)) > 0.174533)
            {   cout << "rotation too large" << endl;
                poseValid = false;
            }
        }
    }

    if (foundPose && poseValid)
    {   rvec.copyTo(_currentFrame.rvec);
        tvec.copyTo(_currentFrame.tvec);
    }

    s->poseTimesMS().set(s->timeMilliSec() - startMS);

    return foundPose && poseValid;
}
//-----------------------------------------------------------------------------
