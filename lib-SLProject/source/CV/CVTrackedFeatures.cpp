//#############################################################################
//  File:      CVTrackedFeatures.cpp
//  Author:    Pascal Zingg, Timon Tschanz, Marcus Hudritsch
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVImage.h>
#include <CVFeatureManager.h>
#include <CVTrackedFeatures.h>
#include <Utils.h>

#if defined(SL_OS_WINDOWS)
#    include <direct.h>
#endif

using namespace cv;

//-----------------------------------------------------------------------------
// Globals for benchmarking
int    frames_with_pose          = 0;
float  sum_matches               = 0;
float  sum_inlier_matches        = 0;
float  sum_allmatches_to_inliers = 0.0f;
double sum_reprojection_error    = 0.0;
float  sum_poseopt_difference    = 0.0f;
double translationError          = 0;
double rotationError             = 0;
int    frames_since_posefound    = 0;

//-----------------------------------------------------------------------------
CVTrackedFeatures::CVTrackedFeatures(string markerFilename)
{
    // To match the binary features, we are matching each descriptor in reference with each
    // descriptor in the current frame. The smaller the hamming distance the better the match
    // Hamming distance <-> XOR sum
    _matcher = BFMatcher::create(BFMatcher::BRUTEFORCE_HAMMING, false);

    // Initialize some member variables on startup to prevent uncontrolled behaviour
    _currentFrame.foundPose         = false;
    _prevFrame.foundPose            = false;
    _currentFrame.reprojectionError = 0.0f;
    _prevFrame.inlierPoints2D       = CVVPoint2f(nFeatures);
    _forceRelocation                = false;
    _frameCount                     = 0;

    loadMarker(std::move(markerFilename));

// Create directory for debug output if flag is set
#ifdef DEBUG_OUTPUT_PATH
#    if defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS)
    mkdir(DEBUG_OUTPUT_PATH, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#    elif defined(SL_OS_WINDOWS)
    _mkdir(DEBUG_OUTPUT_PATH);
#    else
#        undef SAVE_SNAPSHOTS_OUTPUT
#    endif
#endif
}
//-----------------------------------------------------------------------------
//! Show statistics if program terminates
CVTrackedFeatures::~CVTrackedFeatures()
{
#if DO_FEATURE_BENCHMARKING
    Utils::log("");
    Utils::log("");
    Utils::log("------------------------------------------------------------------");
    Utils::log("CVTrackedFeatures statistics");
    Utils::log("------------------------------------------------------------------");
    Utils::log("Avg calculation time per frame                   : %f ms", _trackingTimesMS().average());
    Utils::log("");
    Utils::log("Settings for Pose estimation: ------------------------------------");
    Utils::log("Features                                         : %d", nFeatures);
    Utils::log("Minimal ratio for 2 best matches                 : %f", minRatio);
    Utils::log("RANSAC iterations                                : %d", iterations);
    Utils::log("RANSAC mean reprojection error                   : %f", reprojection_error);
    Utils::log("RANSAC confidence                                : %d", confidence);
    Utils::log("Repose frequency                                 : Each %d point", reposeFrequency);
    Utils::log("Initial patch size for Pose optimization         : %d pixels", initialPatchSize);
    Utils::log("Maximal patch size for Pose optimization         : %d pixels", maxPatchSize);
    Utils::log("");
    Utils::log("Pose information: ------------------------------------------------");
    Utils::log("Avg allmatches to inliers proposition            : %f", sum_allmatches_to_inliers / _frameCount);
    Utils::log("Avg reprojection error (only if POSE)            : %f", sum_reprojection_error / frames_with_pose);
    Utils::log("Pose found                                       : %d of %d frames", frames_with_pose, _frameCount);
    Utils::log("Avg matches                                      : %f", sum_matches / frames_with_pose);
    Utils::log("Avg inlier matches                               : %f", sum_inlier_matches / frames_with_pose);
    Utils::log("Avg more matches with Pose optimization          : %f", sum_poseopt_difference / frames_with_pose);

// Only used for testing with slight movements
//Utils::log("Avg Rotation error                               : %f deg", rotationError / frames_with_pose);
//Utils::log("Avg Translation error                            : %f px", translationError / frames_with_pose);
#endif //DO_FEATURE_BENCHMARKING
}
//-----------------------------------------------------------------------------
//! Loads the marker image form the filesystem
void CVTrackedFeatures::loadMarker(string markerFilename)
{
    // Load the file directly
    if (!Utils::fileExists(markerFilename))
    {
        markerFilename = CVImage::defaultPath + markerFilename;
        if (!Utils::fileExists(markerFilename))
        {
            string msg = "CVTrackedFeatures::loadMarker: File not found: " + markerFilename;
            Utils::exitMsg("SLProject", msg.c_str(), __LINE__, __FILE__);
        }
    }

    CVImage img(markerFilename);
    cvtColor(img.cvMat(), _marker.imageGray, cv::COLOR_RGB2GRAY);
}
//-----------------------------------------------------------------------------
/*! Prepares the reference tracker:
1. Detect and describe the keypoints on the reference image
2. Set up 3D points with predefined scaling
3. Perform optional drawing operations on image
*/
void CVTrackedFeatures::initFeaturesOnMarker()
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
    float pixelPerMM = (float)_marker.imageGray.cols / 297.0f;

    // Calculate 3D-Points based on the detected features
    for (auto& keypoint : _marker.keypoints2D)
    {
        // 2D location in image
        CVPoint2f refImageKeypoint = keypoint.pt;

        // CVPoint scaling
        refImageKeypoint /= pixelPerMM;

        // Here we can use Z=0 because the tracker is planar
        _marker.keypoints3D.push_back(Point3f(refImageKeypoint.x,
                                              refImageKeypoint.y,
                                              0.0f));
    }

// Draw points and indices which should be reprojected later.
// Only a few (defined with reposeFrequency)
// points are used for the reprojection.
#if defined(DEBUG_OUTPUT_PATH) || DRAW_REPROJECTION_POINTS
    _marker.imageGray.copyTo(_marker.imageDrawing);
    cvtColor(_marker.imageDrawing, _marker.imageDrawing, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < _marker.keypoints3D.size(); i++)
    {
        if (i % reposeFrequency)
            continue;

        CVPoint2f originalModelPoint = _marker.keypoints2D[i].pt;

        circle(_marker.imageDrawing,
               originalModelPoint,
               1,
               CV_RGB(255, 0, 0),
               1,
               FILLED);

        putText(_marker.imageDrawing,
                to_string(i),
                CVPoint2f(originalModelPoint.x - 1,
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
void CVTrackedFeatures::type(CVDetectDescribeType ddType)
{
    _featureManager.createDetectorDescriptor(ddType);

    _currentFrame.foundPose         = false;
    _prevFrame.foundPose            = false;
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
bool CVTrackedFeatures::track(CVMat          imageGray,
                              CVMat          image,
                              CVCalibration* calib)
{
    assert(!image.empty() && "Image is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(!_marker.imageGray.empty());

    // Initialize reference points if program just started
    if (_frameCount == 0)
    {
        _calib = calib;
        initFeaturesOnMarker();
    }

    // Copy image matrix into current frame data
    _currentFrame.image     = image;
    _currentFrame.imageGray = imageGray;

    // Determine if relocation or feature tracking should be performed
    bool relocationNeeded = _forceRelocation ||
                            !_prevFrame.foundPose ||
                            _prevFrame.inlierMatches.size() < 100 ||
                            frames_since_posefound < 3;

    // If relocation condition meets, calculate the Pose with feature detection, otherwise
    // track the previous determined features
    if (relocationNeeded)
        relocate();
    else
        tracking();

    if (_currentFrame.foundPose)
    {
        _objectViewMat = createGLMatrix(_currentFrame.tvec, _currentFrame.rvec);
        frames_with_pose++;
    }

    // Perform OpenCV drawning if flags are set (see CVTrackedFeatures.h)
    drawDebugInformation(_drawDetection);

    // Prepare next frame and transfer necessary data
    transferFrameData();

    _frameCount++;

    return _currentFrame.foundPose;
}
//-----------------------------------------------------------------------------
/*! If relocation should be done, the following steps are necessary:
1. Detect keypoints
2. Describe keypoints (Binary descriptors)
3. Match keypoints in current frame and the reference tracker
4. Try to calculate new Pose with Perspective-n-Point algorithm
*/
void CVTrackedFeatures::relocate()
{
    _isTracking = false;
    detectKeypointsAndDescriptors();
    _currentFrame.matches   = getFeatureMatches();
    _currentFrame.foundPose = calculatePose();

    // Zero time keeping on the tracking branch
    CVTracked::optFlowTimesMS.set(0);
}

//-----------------------------------------------------------------------------
/*! To track the already detected keypoints after a sucessful pose estimation,
we track the features with optical flow
*/
void CVTrackedFeatures::tracking()
{
    _isTracking             = true;
    _currentFrame.foundPose = trackWithOptFlow(_prevFrame.rvec, _prevFrame.tvec);

    // Zero time keeping on the relocation branch
    CVTracked::detectTimesMS.set(0);
    CVTracked::matchTimesMS.set(0);
}

//-----------------------------------------------------------------------------
/*! Visualizes the following parts of the whole Pose estimation:
- Keypoints
- Inlier matches
- Optical Flow (Small arrows that show how keypoints moved between frames)
- Reprojection with the calculated Pose
*/
void CVTrackedFeatures::drawDebugInformation(bool drawDetection)
{
    if (drawDetection)
    {
        for (auto& inlierPoint : _currentFrame.inlierPoints2D)
            circle(_currentFrame.image,
                   inlierPoint,
                   3,
                   Scalar(0, 0, 255));
    }

#if DRAW_REPROJECTION_POINTS
    CVMat imgReprojection = _currentFrame.image;
#elif defined(SAVE_SNAPSHOTS_OUTPUT)
    CVMat imgReprojection;
    _currentFrame.image.copyTo(imgReprojection);
#endif

#if DRAW_REPROJECTION_POINTS || defined(DEBUG_OUTPUT_PATH)
    if (!_currentFrame.inlierMatches.empty())
    {
        CVVPoint2f projectedPoints(_marker.keypoints3D.size());

        cv::projectPoints(_marker.keypoints3D,
                          _currentFrame.rvec,
                          _currentFrame.tvec,
                          _calib->cameraMat(),
                          _calib->distortion(),
                          projectedPoints);

        for (size_t i = 0; i < _marker.keypoints3D.size(); i++)
        {
            if (i % reposeFrequency) continue;

            CVPoint2f projectedModelPoint = projectedPoints[i];
            CVPoint2f keypointForPose     = _currentFrame.keypoints[_currentFrame.inlierMatches.back().queryIdx].pt;

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
                    CVPoint2f(projectedModelPoint.x - 2, projectedModelPoint.y - 5),
                    FONT_HERSHEY_SIMPLEX,
                    0.3,
                    CV_RGB(255, 0, 0),
                    1.0);
        }
    }
#endif

#if defined(DEBUG_OUTPUT_PATH)
    // Draw reprojection
    CVMat imgOut;
    drawMatches(imgReprojection,
                CVVKeyPoint(),
                _marker.imageDrawing,
                CVVKeyPoint(),
                CVVDMatch(),
                imgOut,
                CV_RGB(255, 0, 0),
                CV_RGB(255, 0, 0));

    imwrite(DEBUG_OUTPUT_PATH + to_string(_frameCount) + "_reprojection.png",
            imgOut);

    // Draw keypoints
    if (!_currentFrame.keypoints.empty())
    {
        CVMat imgKeypoints;
        drawKeypoints(_currentFrame.imageGray,
                      _currentFrame.keypoints,
                      imgKeypoints);

        imwrite(DEBUG_OUTPUT_PATH + to_string(_frameCount) + "_keypoints.png",
                imgKeypoints);
    }

    for (size_t i = 0; i < _currentFrame.inlierPoints2D.size(); i++)
        circle(_currentFrame.image,
               _currentFrame.inlierPoints2D[i],
               2,
               Scalar(0, 255, 0));

    // Draw matches
    if (!_currentFrame.inlierMatches.empty())
    {
        CVMat imgMatches;
        drawMatches(_currentFrame.imageGray,
                    _currentFrame.keypoints,
                    _marker.imageGray,
                    _marker.keypoints2D,
                    _currentFrame.inlierMatches,
                    imgMatches,
                    CV_RGB(255, 0, 0),
                    CV_RGB(255, 0, 0));

        imwrite(DEBUG_OUTPUT_PATH + to_string(_frameCount) + "_matching.png",
                imgMatches);
    }

    // Draw optical flow
    if (_isTracking)
    {
        CVMat optFlow, rgb;
        _currentFrame.imageGray.copyTo(optFlow);
        cvtColor(optFlow, rgb, CV_GRAY2BGR);
        for (size_t i = 0; i < _currentFrame.inlierPoints2D.size(); i++)
            cv::arrowedLine(rgb,
                            _prevFrame.inlierPoints2D[i],
                            _currentFrame.inlierPoints2D[i],
                            Scalar(0, 255, 0),
                            1,
                            LINE_8,
                            0,
                            0.2);

        imwrite(DEBUG_OUTPUT_PATH + to_string(_frameCount) + "-optflow.png", rgb);
    }
#endif
}
//-----------------------------------------------------------------------------
/*! Copies the current frame data to the previous frame data struct for the
next frame handling.
TODO: more elegant way to do this whole copy action
*/
void CVTrackedFeatures::transferFrameData()
{
    _currentFrame.imageGray.copyTo(_prevFrame.imageGray);
    _currentFrame.image.copyTo(_prevFrame.image);
    _currentFrame.rvec.copyTo(_prevFrame.rvec);
    _currentFrame.tvec.copyTo(_prevFrame.tvec);

    _prevFrame.reprojectionError = _currentFrame.reprojectionError;
    _prevFrame.foundPose         = _currentFrame.foundPose;
    _prevFrame.inlierPoints3D    = _currentFrame.inlierPoints3D;
    _prevFrame.inlierPoints2D    = _currentFrame.inlierPoints2D;

    if (!_currentFrame.inlierMatches.empty())
        _prevFrame.inlierMatches = _currentFrame.inlierMatches;

    _currentFrame.keypoints.clear();
    _currentFrame.matches.clear();
    _currentFrame.inlierMatches.clear();
    _currentFrame.inlierPoints2D.clear();
    _currentFrame.inlierPoints3D.clear();
    _currentFrame.reprojectionError = 0;

    _currentFrame.useExtrinsicGuess = _prevFrame.foundPose;

    if (_prevFrame.foundPose)
    {
        _currentFrame.rvec = _prevFrame.rvec;
        _currentFrame.tvec = _prevFrame.tvec;
    }
    else
    {
        _currentFrame.rvec = CVMat::zeros(3, 1, CV_64FC1);
        _currentFrame.tvec = CVMat::zeros(3, 1, CV_64FC1);
    }
}
//-----------------------------------------------------------------------------
/*! Get keypoints and descriptors in one step. This is a more efficient way
since we have to build the scaling pyramide only once. If we detect and
describe seperatly, it will lead in two scaling pyramids and is therefore less
meaningful.
*/
void CVTrackedFeatures::detectKeypointsAndDescriptors()
{
    float startMS = _timer.elapsedTimeInMilliSec();

    _featureManager.detectAndDescribe(_currentFrame.imageGray,
                                      _currentFrame.keypoints,
                                      _currentFrame.descriptors);

    CVTracked::detectTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);
}
//-----------------------------------------------------------------------------
/*! Get matching features with the defined feature matcher. Since we are using
the k-next-neighbour matcher, we check if the best and second best match are
not too identical with the so called ratio test.
@return Vector of found matches
*/
CVVDMatch CVTrackedFeatures::getFeatureMatches()
{
    float startMS = _timer.elapsedTimeInMilliSec();

    int        k = 2;
    CVVVDMatch matches;
    _matcher->knnMatch(_currentFrame.descriptors, _marker.descriptors, matches, k);

    // Perform ratio test which determines if k matches from the knn matcher
    // are not too similar. If the ratio of the the distance of the two
    // matches is toward 1, the matches are near identically.
    CVVDMatch goodMatches;
    for (auto& match : matches)
    {
        const DMatch& match1 = match[0];
        const DMatch& match2 = match[1];
        if (match2.distance == 0.0f ||
            (match1.distance / match2.distance) < minRatio)
            goodMatches.push_back(match1);
    }

    CVTracked::matchTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);
    return goodMatches;
}
//-----------------------------------------------------------------------------
/*! This method does the most important work of the whole pipeline:

RANSAC: We execute first RANSAC to eliminate wrong feature correspondences
(outliers) and only use the correct ones (inliers) for PnP solving
(https://en.wikipedia.org/wiki/Perspective-n-Point).\n
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
bool CVTrackedFeatures::calculatePose()
{
    // solvePnP crashes if less than 5 points are given
    if (_currentFrame.matches.size() < 10) return false;

    float startMS = _timer.elapsedTimeInMilliSec();

    // Find 2D/3D correspondences
    // At the moment we are using only the two correspondences like this:
    // KeypointsOriginal <-> KeypointsActualscene
    // Train index --> "CVPoint" in the model
    // Query index --> "CVPoint" in the actual frame

    if (_currentFrame.matches.size() < 10)
        return false;

    CVVPoint3f modelPoints(_currentFrame.matches.size());
    CVVPoint2f framePoints(_currentFrame.matches.size());

    for (size_t i = 0; i < _currentFrame.matches.size(); i++)
    {
        modelPoints[i] = _marker.keypoints3D[(uint)_currentFrame.matches[i].trainIdx];
        framePoints[i] = _currentFrame.keypoints[(uint)_currentFrame.matches[i].queryIdx].pt;
    }

    vector<uchar> inliersMask(modelPoints.size());

    //////////////////////
    // 1. RANSAC with EPnP
    //////////////////////

    bool foundPose = cv::solvePnPRansac(modelPoints,
                                        framePoints,
                                        _calib->cameraMat(),
                                        _calib->distortion(),
                                        _currentFrame.rvec,
                                        _currentFrame.tvec,
                                        _currentFrame.useExtrinsicGuess,
                                        iterations,
                                        reprojection_error,
                                        confidence,
                                        inliersMask,
                                        SOLVEPNP_EPNP);

    // Get matches with help of inlier indices
    for (size_t idx : inliersMask)
    {
        _currentFrame.inlierMatches.push_back(_currentFrame.matches[idx]);
        _currentFrame.inlierPoints2D.push_back(framePoints[idx]);
        _currentFrame.inlierPoints3D.push_back(modelPoints[idx]);
    }

    // Pose optimization
    if (foundPose)
    {
        //float matchesBefore = (float)_currentFrame.inlierMatches.size();

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

#if DO_FEATURE_BENCHMARKING
        sum_matches += _currentFrame.matches.size();
        sum_inlier_matches += _currentFrame.inlierMatches.size();
        sum_allmatches_to_inliers += _currentFrame.inlierMatches.size() /
                                     _currentFrame.matches.size();
        sum_poseopt_difference += _currentFrame.inlierMatches.size() /
                                  matchesBefore;
#endif
    }

    CVTracked::poseTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

    return foundPose;
}
//-----------------------------------------------------------------------------
/*! To get more matches with the calculated pose, we reproject the reference
points to our current frame. Within a predefined patch, we try to rematch not
matched features with the reprojected point. If not possible, we increase the
patch size until we found a match for the point or we reach a threshold.
*/
void CVTrackedFeatures::optimizeMatches()
{
    float reprojectionError = 0;

    // 1. Reproject the model points with the calculated POSE
    CVVPoint2f projectedPoints(_marker.keypoints3D.size());
    cv::projectPoints(_marker.keypoints3D,
                      _currentFrame.rvec,
                      _currentFrame.tvec,
                      _calib->cameraMat(),
                      _calib->distortion(),
                      projectedPoints);

    CVVKeyPoint    bboxFrameKeypoints;
    vector<size_t> frameIndicesInsideRect;

    for (size_t i = 0; i < _marker.keypoints3D.size(); i++)
    {
        //only every reposeFrequency
        if (i % reposeFrequency)
            continue;

        // Check if this point has a match inside matches, continue if so
        int alreadyMatched = 0;
        //todo: this is bad, because for every marker keypoint we have to iterate all inlierMatches!
        //better: iterate inlierMatches once at the beginning and mark all marker keypoints as inliers or not!
        for (size_t j = 0; j < _currentFrame.inlierMatches.size(); j++)
        {
            if (_currentFrame.inlierMatches[(uint)j].trainIdx == (int)i)
                alreadyMatched++;
        }

        if (alreadyMatched > 0) continue;

        // Get the corresponding projected point of the actual (i) modelpoint
        CVPoint2f projectedModelPoint = projectedPoints[i];
        CVVDMatch newMatches;

        int patchSize = initialPatchSize;

        // Adaptive patch size
        while (newMatches.empty() && patchSize <= maxPatchSize)
        {
            // Increase matches by even number
            patchSize += 2;
            newMatches.clear();
            bboxFrameKeypoints.clear();
            frameIndicesInsideRect.clear();

            // 2. Select only before calculated Keypoints within patch
            // with projected "positioning" keypoint as center
            // OpenCV: Top-left origin
            int xTopLeft   = (int)(projectedModelPoint.x - (float)patchSize / 2.0f);
            int yTopLeft   = (int)(projectedModelPoint.y - (float)patchSize / 2.0f);
            int xDownRight = xTopLeft + patchSize;
            int yDownRight = yTopLeft + patchSize;

            for (size_t j = 0; j < _currentFrame.keypoints.size(); j++)
            { // bbox check
                if (_currentFrame.keypoints[j].pt.x > xTopLeft &&
                    _currentFrame.keypoints[j].pt.x < xDownRight &&
                    _currentFrame.keypoints[j].pt.y > yTopLeft &&
                    _currentFrame.keypoints[j].pt.y < yDownRight)
                {
                    bboxFrameKeypoints.push_back(_currentFrame.keypoints[j]);
                    frameIndicesInsideRect.push_back(j);
                }
            }

            // 3. Match the descriptors of the keypoints inside
            // the rectangle around the projected map point
            // with the descritor of the projected map point.

            // This is our descriptor for the model point i
            CVMat modelPointDescriptor = _marker.descriptors.row((int)i);

            // We extract the descriptors which belong to the keypoints
            // inside the rectangle around the projected map point
            CVMat bboxPointsDescriptors;
            for (size_t j : frameIndicesInsideRect)
                bboxPointsDescriptors.push_back(_currentFrame.descriptors.row((int)j));

            // 4. Match the frame keypoints inside the rectangle with the projected model point
            _matcher->match(bboxPointsDescriptors, modelPointDescriptor, newMatches);
        }

        if (!newMatches.empty())
        {
            for (size_t j = 0; j < frameIndicesInsideRect.size(); j++)
            {
                newMatches[j].trainIdx = (int)i;
                newMatches[j].queryIdx = (int)frameIndicesInsideRect[j];
            }

            // 5. Only add the best new match to matches vector
            CVDMatch bestNewMatch;
            bestNewMatch.distance = 0;

            for (CVDMatch newMatch : newMatches)
                if (bestNewMatch.distance < newMatch.distance)
                    bestNewMatch = newMatch;

            // 6. Only add the best new match to matches vector
            _currentFrame.inlierMatches.push_back(bestNewMatch);
        }

        // Get the keypoint which was used for pose estimation
        CVPoint2f keypointForPose = _currentFrame.keypoints[(uint)_currentFrame.inlierMatches.back().queryIdx].pt;
        reprojectionError += (float)norm(CVMat(projectedModelPoint),
                                         CVMat(keypointForPose));

#if DRAW_PATCHES
        //draw green rectangle around every map point
        rectangle(_currentFrame.image,
                  Point2f(projectedModelPoint.x - (float)patchSize / 2.0f,
                          projectedModelPoint.y - (float)patchSize / 2.0f),
                  Point2f(projectedModelPoint.x + (float)patchSize / 2.0f,
                          projectedModelPoint.y + (float)patchSize / 2.0f),
                  CV_RGB(0, 255, 0));

        //draw key points, that lie inside this rectangle
        for (const auto& kPt : bboxFrameKeypoints)
            circle(_currentFrame.image,
                   kPt.pt,
                   1,
                   CV_RGB(0, 0, 255),
                   1,
                   FILLED);
#endif
    }

#if DO_FEATURE_BENCHMARKING
    sum_reprojection_error += reprojectionError / _marker.keypoints3D.size();

    CVMat prevRmat, currRmat;
    if (_prevFrame.foundPose)
    {
        Rodrigues(_prevFrame.rvec, prevRmat);
        Rodrigues(_currentFrame.rvec, currRmat);
        double rotationError_rad = acos((trace(prevRmat * currRmat).val[0] - 1.0) / 2.0);
        rotationError += rotationError_rad * 180 / 3.14;
        translationError += cv::norm(_prevFrame.tvec, _currentFrame.tvec);
    }
#endif

#if DRAW_REPROJECTION_POINTS
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
    {
        modelPoints[i] = _marker.keypoints3D[(uint)_currentFrame.inlierMatches[i].trainIdx];
        framePoints[i] = _currentFrame.keypoints[(uint)_currentFrame.inlierMatches[i].queryIdx].pt;
    }

    if (modelPoints.empty()) return;
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
bool CVTrackedFeatures::trackWithOptFlow(CVMat rvec, CVMat tvec)
{
    if (_prevFrame.inlierPoints2D.size() < 4) return false;

    float startMS = _timer.elapsedTimeInMilliSec();

    vector<uchar> status;
    vector<float> err;
    CVSize        winSize(15, 15);

    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                              10,    // terminate after this many iterations, or
                              0.03); // when the search window moves by less than this

    // Find closest possible feature points based on optical flow
    CVVPoint2f pred2DPoints(_prevFrame.inlierPoints2D.size());

    //todo: do not relate optical flow to previous frame! better to original marker image, otherwise we will drift
    cv::calcOpticalFlowPyrLK(
      _prevFrame.imageGray,      // Previous frame
      _currentFrame.imageGray,   // Current frame
      _prevFrame.inlierPoints2D, // Previous and current keypoints coordinates.The latter will be
      pred2DPoints,              // expanded if more good coordinates are detected during OptFlow
      status,                    // Output vector for keypoint correspondences (1 = match found)
      err,                       // Error size for each flow
      winSize,                   // Search window for each pyramid level
      3,                         // Max levels of pyramid creation
      criteria,                  // Configuration from above
      0,                         // Additional flags
      0.001);                    // Minimal Eigen threshold

    // Only use points which are not wrong in any way during the optical flow calculation
    CVVPoint2f frame2DPoints;
    CVVPoint3f model3DPoints;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            frame2DPoints.push_back(pred2DPoints[i]);
            //Original code from Zingg/Tschanz got zero size vector
            //model3DPoints.push_back(_currentFrameFrame.inlierPoints3D[i]);
            model3DPoints.push_back(_prevFrame.inlierPoints3D[i]);
        }
    }

    CVTracked::optFlowTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

    _currentFrame.inlierPoints2D = frame2DPoints;
    _currentFrame.inlierPoints3D = model3DPoints;

    if (_currentFrame.inlierPoints2D.size() < _prevFrame.inlierPoints2D.size() * 0.75)
        return false;

    /////////////////////
    // Pose Estimation //
    /////////////////////

    startMS = _timer.elapsedTimeInMilliSec();

    bool foundPose = cv::solvePnP(model3DPoints,
                                  frame2DPoints,
                                  _calib->cameraMat(),
                                  _calib->distortion(),
                                  rvec,
                                  tvec,
                                  true);
    bool poseValid = true;

    if (foundPose)
    {
        for (int i = 0; i < tvec.cols; i++)
        {
            if (abs(tvec.at<double>(i, 0) - tvec.at<double>(i, 0)) > abs(tvec.at<double>(i, 0)) * 0.2)
            {
                cout << "translation too large" << endl;
                poseValid = false;
            }
        }
        for (int i = 0; i < rvec.cols; i++)
        {
            if (abs(rvec.at<double>(i, 0) - rvec.at<double>(i, 0)) > 0.174533)
            {
                cout << "rotation too large" << endl;
                poseValid = false;
            }
        }
    }

    if (foundPose && poseValid)
    {
        rvec.copyTo(_currentFrame.rvec);
        tvec.copyTo(_currentFrame.tvec);
    }

    CVTracked::poseTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

    return foundPose && poseValid;
}
//-----------------------------------------------------------------------------
