//#############################################################################
//  File:      SLCVTrackerFeatures.h
//  Author:    Pascal Zingg, Timon Tschanz
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTrackerFeatures_H
#define SLCVTrackerFeatures_H

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
#define OPTIMIZE_POSE 1
#define DISTINGUISH_FEATURE_DETECT_COMPUTE 0
#define TRACKING_MEASUREMENT 1

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

// Settings for drawing things into current camera frame
#define DRAW_KEYPOINTS 1
#define DRAW_REPROJECTION 1
#define DRAW_REPOSE_INFO 1

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


class SLCVTrackerFeatures : public SLCVTracker
{
public:
        SLCVTrackerFeatures         (SLNode* node);
        ~SLCVTrackerFeatures        ();
        SLbool  track               (SLCVMat imageGray,
                                     SLCVMat image,
                                     SLCVCalibration* calib,
                                     SLSceneView* sv);

private:
        static SLVMat4f         objectViewMats; //!< object view SLCVMatrices
        SLCVRaulMurOrb*         _detector;
        Ptr<ORB>                _descriptor;
        Ptr<DescriptorMatcher>  _matcher;

        SLMat4f                 _pose;
        SLCVCalibration         *_calib;
        int                     frameCount = 0, reposePatchSize;

        // TODO: Merge structs? Class representation?
        struct current {
            SLCVMat             image;
            SLCVMat             imageGray;

            vector<Point2f>     inlierPoints2D;
            vector<Point3f>     inlierPoints3D;

            SLCVVKeyPoint       keypoints;
            SLCVMat             descriptors;
            vector<DMatch>      matches;
            vector<DMatch>      inlierMatches;

            SLCVMat             rvec;
            SLCVMat             tvec;

            bool                foundPose;
            float               reprojectionError;
            bool                useExtrinsicGuess;
        } _current;

        struct prev {
            SLCVMat             image;
            SLCVMat             imageGray;

            vector<Point2f>     inlierPoints2D;
            vector<Point3f>     inlierPoints3D;

            SLCVMat             rvec;
            SLCVMat             tvec;

            bool                foundPose;
            float               reprojectionError;
        } _prev;

        struct map {
            vector<Point3f>     model;
            SLCVMat             frameGray;
            SLCVMat             imgDrawing;
            SLCVVKeyPoint       keypoints;
            SLCVMat             descriptors;
            SLCVVKeyPoint       bboxModelKeypoints;
        } _map;

        void initModel();
        void relocate();
        void tracking();
        void saveImageOutput();
        void updateSceneCam(SLSceneView* sv);
        void transferFrameData();

        SLCVVKeyPoint getKeypoints(const SLCVMat &imageGray);

        SLCVMat getDescriptors(const SLCVMat &imageGray, SLCVVKeyPoint &keypoints);

        void getKeypointsAndDescriptors(const SLCVMat &imageGray, SLCVVKeyPoint &keypoints, SLCVMat &descriptors);

        vector<DMatch> getFeatureMatches(const SLCVMat &descriptors);

        bool calculatePose(const SLCVMat &imageVideo, vector<KeyPoint> &keypoints, vector<DMatch> &matches,
            vector<DMatch> &inliers, Mat &rvec, SLCVMat &tvec, bool extrinsicGuess, const SLCVMat& descriptors);

        bool solvePnP(vector<Point3f> &modelPoints, vector<Point2f> &framePoints, bool guessExtrinsic,
            SLCVMat &rvec, SLCVMat &tvec, vector<unsigned char> &inliersMask);

        bool optimizePose(const SLCVMat &imageVideo, vector<KeyPoint> &keypoints, const SLCVMat &descriptors,
            vector<DMatch> &matches, SLCVMat &rvec, SLCVMat &tvec, float reprojectionError=0);

        bool trackWithOptFlow(Mat &previousFrame, vector<Point2f> &previousPoints, Mat &actualFrame,
                              Mat rvec, Mat tvec, SLCVMat &frame);
};
//-----------------------------------------------------------------------------
#endif // SLCVTrackerFeatures_H
