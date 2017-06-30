//#############################################################################
//  File:      SLCVTrackerFeatures.h
//  Author:    Pascal Zingg, Timon Tschanz
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTRACKERFEATURES_H
#define SLCVTRACKERFEATURES_H

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/
#include <SLCV.h>
#include <SLCVTracker.h>
#include <SLNode.h>
#include <SLCVRaulMurOrb.h>

using namespace cv;

#define DEBUG_OUTPUT 0
#define FORCE_REPOSE 0
#define DISTINGUISH_FEATURE_DETECT_COMPUTE 0
#define BENCHMARKING 1

// Settings for drawing things into current camera frame
#define DRAW_INLIERMATCHES 0
#define DRAW_REPROJECTION_POINTS 0
#define DRAW_REPROJECTION_ERROR 0
#define DRAW_PATCHES 0


// Set stones Tracker as default reference image
#ifndef SL_TRACKER_IMAGE_NAME
    #define SL_TRACKER_IMAGE_NAME "stones"
#endif

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
const int maxPatchSize = 60;

//-----------------------------------------------------------------------------
//!???
/*!???
*/
class SLCVTrackerFeatures : public SLCVTracker
{
public:
        SLCVTrackerFeatures     (SLNode* node);
        ~SLCVTrackerFeatures    ();
        SLbool  track           (SLCVMat imageGray,
                                SLCVMat image,
                                SLCVCalibration* calib,
                                SLSceneView* sv);

private:
        void                initializeReference (string trackerName);
        void                relocate            ();
        void                tracking            ();
        void                drawDebugInformation();
        void                updateSceneCamera   (SLSceneView* sv);
        void                transferFrameData   ();
        SLCVVKeyPoint       detectKeypoints     ();
        SLCVMat             computeDescriptors  ();
        void                getKeypointsAndDescriptors();
        SLCVVDMatch         getFeatureMatches   ();
        bool                calculatePose       ();
        void                optimizeMatches     ();
        bool                trackWithOptFlow    (SLCVMat rvec, SLCVMat tvec);

        Ptr<DescriptorMatcher>  _matcher;       //!< Descriptor matching algorithm
        SLCVCalibration*        _calib;         //!< Current calibration in use
        SLint                   _frameCount=0;  //!< NO. of frames since process start
        bool                    _isTracking;    //!< True if tracking

        //! Data of a 2D marker image
        struct FeatureMarker2D
        {   SLCVVPoint3f    keypoints3D;        //!< 3D feature points in mm
            SLCVMat         imageGray;          //!< Grayscale image of the marker
            SLCVMat         imageDrawing;       //!< Color debug image
            SLCVVKeyPoint   keypoints2D;        //!< 2D keypoints in pixels
            SLCVMat         descriptors;        //!< Descriptors of the 2D keypoints
            SLCVVKeyPoint   bboxKeypoints2D;    //!< bounding boxes of 2D keypoints
        };

        //! Feature date for a video frame
        struct FrameData
        {   SLCVMat         image;              //!< Reference to color video frame
            SLCVMat         imageGray;          //!< Reference to grayscale video frame
            SLCVVPoint2f    inlierPoints2D;     //!< Inlier 2D points after RANSAC
            SLCVVPoint3f    inlierPoints3D;     //!< Inlier 3D points after RANSAC on the marker
            SLCVVKeyPoint   keypoints;          //!< 2D keypoints detected in video frame
            SLCVMat         descriptors;        //!< Descriptors of keypoints
            SLCVVDMatch     matches;            //!< matches between video decriptors and marker descriptors
            SLCVVDMatch     inlierMatches;      //!< matches that lead to correct transform
            SLCVMat         rvec;               //!< Rotation of the camera pose
            SLCVMat         tvec;               //!< Translation of the camera pose
            SLbool          foundPose;          //!< True if pose was found
            SLfloat         reprojectionError;  //!< Reprojection error of the pose
            SLbool          useExtrinsicGuess;  //!< flag if extrinsic gues should be used
        };

        FeatureMarker2D     _marker;            //!< 2D marker data
        FrameData           _current;           //!< The current video frame date
        FrameData           _prev;              //!< The previous video frame date
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackerFeatures_H
