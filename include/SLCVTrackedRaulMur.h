//#############################################################################
//  File:      SLCVTrackedRaulMur.h
//  Author:    Michael Göttlicher
//  Date:      Dez 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVTRACKERRAULMUR_H
#define SLCVTRACKERRAULMUR_H

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/
#include <SLCV.h>
#include <SLCVTracked.h>
#include <SLNode.h>
#include <SLCVFrame.h>
#include <SLCVKeyFrameDB.h>

using namespace cv;

//-----------------------------------------------------------------------------
//! SLCVTrackedRaulMur is the main part of the AR Christoffelturm scene
/*! 
*/
class SLCVTrackedRaulMur : public SLCVTracked
{
public:
    // Tracking states
    enum eTrackingState {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    SLCVTrackedRaulMur(SLNode *node, ORBVocabulary* vocabulary,
        SLCVKeyFrameDB* keyFrameDB);
    ~SLCVTrackedRaulMur();
    SLbool track(SLCVMat imageGray,
        SLCVMat image,
        SLCVCalibration* calib,
        SLbool drawDetection,
        SLSceneView* sv);

protected:
    bool Relocalization();
    bool TrackWithMotionModel();
    bool TrackLocalMap();
    void SearchLocalPoints();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    //Motion Model
    cv::Mat mVelocity;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO = false;

private:
    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    SLCVKeyFrameDB* mpKeyFrameDatabase;

    // Current Frame
    SLCVFrame mCurrentFrame;

    //extractor instance
    ORB_SLAM2::ORBextractor* _extractor = NULL;

    //Last Frame, KeyFrame and Relocalisation Info
    unsigned int mnLastRelocFrameId = 0;

    //Last Frame, KeyFrame and Relocalisation Info
    SLCVFrame mLastFrame;
    unsigned int mnLastKeyFrameId;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<SLCVKeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    //Local Map 
    //(maybe always the last inserted keyframe?)
    SLCVKeyFrame* mpReferenceKF = NULL;
    std::vector<SLCVMapPoint*> mvpLocalMapPoints;
    std::vector<SLCVKeyFrame*> mvpLocalKeyFrames;

    //New KeyFrame rules (according to fps)
    // Max/Min Frames to insert keyframes and to check relocalisation
    int mMinFrames = 0;
    int mMaxFrames = 30; //= fps

    //Current matches in frame
    int mnMatchesInliers = 0;
};
//-----------------------------------------------------------------------------
#endif //SLCVTRACKERRAULMUR_H