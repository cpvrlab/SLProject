//#############################################################################
//  File:      CVTrackedFeatures.h
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Pascal Zingg, Timon Tschanz, Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVTRACKEDFEATURES_H
#define CVTRACKEDFEATURES_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/
#include <CVTypedefs.h>
#include <CVFeatureManager.h>
#include <CVRaulMurOrb.h>
#include <CVTracked.h>

#define SL_SPLIT_DETECT_COMPUTE 0
#define SL_DO_FEATURE_BENCHMARKING 0

// Settings for drawing things into current camera frame
#define SL_DRAW_REPROJECTION_POINTS 1
#define SL_DRAW_REPROJECTION_ERROR 1
#define SL_DRAW_PATCHES 1

//#define SL_SAVE_DEBUG_OUTPUT
#ifdef SL_SAVE_DEBUG_OUTPUT
#    if defined(SL_OS_LINUX) || defined(SL_OS_MACOS) || defined(SL_OS_MACIOS)
#        define SL_DEBUG_OUTPUT_PATH "/tmp/cv_tracking/"
#    elif defined(SL_OS_WINDOWS)
#        define SL_DEBUG_OUTPUT_PATH "cv_tracking/"
#    endif
#endif

// Feature detection and extraction
const int   nFeatures = 2000;
const float minRatio  = 0.7f;

// RANSAC parameters
const int    iterations         = 500;
const float  reprojection_error = 2.0f;
const double confidence         = 0.95;

// Repose patch size
const int reposeFrequency  = 10;
const int initialPatchSize = 2;
const int maxPatchSize     = 60;

//-----------------------------------------------------------------------------
//! CVTrackedFeatures is the main part of the AR Christoffelturm scene
/*! The implementation tries to find a valid pose based on feature points in
realtime. The feature matching algorithm checks the points of the current camera
frame with against a reference. There are two important parts of this procedure:
The relocalisation, which will be called if we have to find the pose with no hint
where the camera could be. The other one is called feature tracking: If a pose
was found, the implementation tries to track them and update the pose respectively.
*/
class CVTrackedFeatures : public CVTracked
{
public:
    explicit CVTrackedFeatures(string markerFilename);
    ~CVTrackedFeatures();
    bool track(CVMat          imageGray,
               CVMat          image,
               CVCalibration* calib) final;
    // Getters
    bool                 forceRelocation() { return _forceRelocation; }
    CVDetectDescribeType type() { return _featureManager.type(); }

    // Setters
    void forceRelocation(bool fR) { _forceRelocation = fR; }
    void type(CVDetectDescribeType ddType);

private:
    void      loadMarker(string markerFilename);
    void      initFeaturesOnMarker();
    void      relocate();
    void      tracking();
    void      drawDebugInformation(bool drawDetection);
    void      transferFrameData();
    void      detectKeypointsAndDescriptors();
    CVVDMatch getFeatureMatches();
    bool      calculatePose();
    void      optimizeMatches();
    bool      trackWithOptFlow(CVMat rvec, CVMat tvec);

    cv::Ptr<cv::DescriptorMatcher> _matcher;    //!< Descriptor matching algorithm
    CVCalibration*                 _calib;      //!< Current calibration in use
    int                            _frameCount; //!< NO. of frames since process start
    bool                           _isTracking; //!< True if tracking

    //! Data of a 2D marker image
    struct SLFeatureMarker2D
    {
        CVMat       imageGray;    //!< Grayscale image of the marker
        CVMat       imageDrawing; //!< Color debug image
        CVVKeyPoint keypoints2D;  //!< 2D keypoints in pixels
        CVVPoint3f  keypoints3D;  //!< 3D feature points in mm
        CVMat       descriptors;  //!< Descriptors of the 2D keypoints
    };

    //! Feature date for a video frame
    struct SLFrameData
    {
        CVMat       image;             //!< Reference to color video frame
        CVMat       imageGray;         //!< Reference to grayscale video frame
        CVVPoint2f  inlierPoints2D;    //!< Inlier 2D points after RANSAC
        CVVPoint3f  inlierPoints3D;    //!< Inlier 3D points after RANSAC on the marker
        CVVKeyPoint keypoints;         //!< 2D keypoints detected in video frame
        CVMat       descriptors;       //!< Descriptors of keypoints
        CVVDMatch   matches;           //!< matches between video decriptors and marker descriptors
        CVVDMatch   inlierMatches;     //!< matches that lead to correct transform
        CVMat       rvec;              //!< Rotation of the camera pose
        CVMat       tvec;              //!< Translation of the camera pose
        bool        foundPose;         //!< True if pose was found
        float       reprojectionError; //!< Reprojection error of the pose
        bool        useExtrinsicGuess; //!< flag if extrinsic gues should be used
    };

    SLFeatureMarker2D _marker;          //!< 2D marker data
    SLFrameData       _currentFrame;    //!< The current video frame data
    SLFrameData       _prevFrame;       //!< The previous video frame data
    bool              _forceRelocation; //!< Force relocation every frame (no opt. flow tracking)
    CVFeatureManager  _featureManager;  //!< Feature detector-descriptor wrapper instance
};
//-----------------------------------------------------------------------------
#endif // CVTrackedFeatures_H
