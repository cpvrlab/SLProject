//#############################################################################
//  File:      SLCVTrackedRaulMur.h
//  Author:    Michael Goettlicher
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
#include <SLCVMap.h>

class SLCVMapNode;

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
        SLCVKeyFrameDB* keyFrameDB, SLCVMap* map, SLCVMapNode* mapNode = NULL );
    ~SLCVTrackedRaulMur();
    SLbool track(SLCVMat imageGray,
        SLCVMat image,
        SLCVCalibration* calib,
        SLbool drawDetection,
        SLSceneView* sv);

    //add map points to scene and keypoints to video image
    void decorateSceneAndVideo(cv::Mat& image);

    //setters
    void showMatchesPC(bool s) { _showMatchesPC = s; }
    void showLocalMapPC(bool s) { _showLocalMapPC = s; }
    void showKeyPoints(bool s) { _showKeyPoints = s; }
    void showKeyPointsMatched(bool s) { _showKeyPointsMatched = s; }
    //getters
    SLCVMap* getMap() { return _map; }
    SLCVKeyFrameDB* getKfDB() { return mpKeyFrameDatabase; }
    int getNMapMatches() { return mnMatchesInliers; }
    bool showMatchesPC() { return _showMatchesPC; }
    bool showLocalMapPC() { return _showLocalMapPC; }
    bool showKeyPoints() { return _showKeyPoints; }
    bool showKeyPointsMatched() { return _showKeyPointsMatched; }
    double poseDifference() { return _poseDifference; }
    double meanReprojectionError() { return _meanReprojectionError; }
    int mapPointsCount() {
        if (_map)
            return _map->MapPointsInMap();
        else
            return 0;
    }
    string getPrintableState() {
        switch (mState)
        {
        case SYSTEM_NOT_READY:
            return "SYSTEM_NOT_READY";
        case NO_IMAGES_YET:
            return "NO_IMAGES_YET";
        case NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case OK:
            if (!mbVO) {
                if (!mVelocity.empty())
                    return "OK_MM"; //motion model tracking
                else
                    return "OK_RF"; //reference frame tracking
            }
            else {
                return "OK_VO";
            }
            return "OK";
        case LOST:
            return "LOST";

        return "";
        }
    }

    enum TransformType {
        ROT_X=0, ROT_Y, ROT_Z, TRANS_X, TRANS_Y, TRANS_Z, SCALE
    };

    //// Build rotation matrix
    //Mat buildRotMat(float &valDeg, int type);
    //Mat buildTransMat(float &val, int type);
    //void rotate(float value, int type);
    //void translate(float value, int type);
    //void scale(float value);
    void applyTransformation(double value, TransformType type);
    void saveState();

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

    //map containing map points
    SLCVMap* _map = NULL;

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

    //flags, if we have to update the scene object of the map point matches
    bool _showMatchesPC = true;
    bool _showLocalMapPC = false;
    bool _showKeyPoints = false;
    bool _showKeyPointsMatched = true;
    //SLMaterial* _pcMat1 = NULL;
    //SLMaterial* _pcMat2 = NULL;

    //mean reprojection error
    double _meanReprojectionError = -1.0;
    //L2 norm of the difference between the last and the current camera pose
    double _poseDifference = -1.0;

    //scene nodes to point clouds:
    //SLNode* _mapPC=NULL;
    //SLNode* _mapMatchesPC = NULL;
    //SLNode* _mapLocalPC = NULL;
    //SLNode* _keyFrames = NULL;
    SLCVMapNode* _mapNode = NULL;

    //cv::Mat _image;
    SLCVCalibration*        _calib = NULL;         //!< Current calibration in use
    SLint                   _frameCount=0;    //!< NO. of frames since process start
};
//-----------------------------------------------------------------------------
#endif //SLCVTRACKERRAULMUR_H