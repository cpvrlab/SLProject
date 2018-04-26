//#############################################################################
//  File:      SLCVMapTracking.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVMAPTRACKING_H
#define SLCVMAPTRACKING_H

#include <SLTrackingInfosInterface.h>
#include <SLCVFrame.h>

class SLCVKeyFrameDB;
class SLCVMap;
class SLCVMapNode;

//-----------------------------------------------------------------------------
//! Map Tracking
/*All map trackings have in common, that their tracking tracks the Elemtents of 
based a SLCVMap, that is SLCVMapPoints and SLCVKeyFrames.
A tracking implements different tracking and relocalization strategies.
The SLCVMapTracking class also impements the state machine and the states.
Also, this class contains functions to update the scene (SLCVMapNode).
*/
class SLCVMapTracking : public SLTrackingInfosInterface
{
public:
    SLCVMapTracking(SLCVKeyFrameDB* keyFrameDB, SLCVMap* map, SLCVMapNode* mapNode);
    SLCVMapTracking(SLCVMapNode* mapNode);

    /****** State machine parameter *********************************/
    // Tracking states
    enum eTrackingState {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3,
        IDLE = 4
    };

    eTrackingState mState = NOT_INITIALIZED;;
    eTrackingState mLastProcessedState = NOT_INITIALIZED;

    virtual void Reset() = 0;

    //!request state idle
    void requestStateIdle()
    {
        std::lock_guard<std::mutex> guard(_mutexStates);
        _idleRequested = true;
    }
    //!If system is in idle, it resumes with INITIAIZED or NOT_INITIALIZED state depending on if system is initialized.
    void requestResume()
    {
        std::lock_guard<std::mutex> guard(_mutexStates);
        _resumeRequested = true;
    }
    //!request reset. state switches to idle afterwards.
    void requestReset()
    {
        std::lock_guard<std::mutex> guard(_mutexStates);
        _resetRequested = true;
    }
    //!check current state
    bool hasStateIdle()
    {
        std::lock_guard<std::mutex> guard(_mutexStates);
        return mState == IDLE;
    }
    bool isMapInitialized()
    {
        std::lock_guard<std::mutex> guard(_mutexStates);
        return _mapInitialized;
    }
    void setMapInitialized(bool state)
    {
        std::lock_guard<std::mutex> guard(_mutexStates);
        _mapInitialized = state;
    }

    bool _mapInitialized = false;
    //!System switches to IDLE state as soon as possible.
    bool _resumeRequested = false;
    //!System switches to IDLE state as soon as possible.
    bool _idleRequested = false;
    //!From every possible state, the system changes to state RESETTING as soon as possible.
    //! The system switches to IDLE, as soon as it is resetted.
    bool _resetRequested = false;

    std::mutex _mutexStates;
    /****** State machine parameter - end *****************************/

    int getNMapMatches() override;
    int getNumKeyFrames() override;

    float poseDifference() override;
    float meanReprojectionError() override;
    int mapPointsCount() override;

    //getters
    SLCVMap* getMap() { return _map; }
    SLCVKeyFrameDB* getKfDB() { return mpKeyFrameDatabase; }
    SLCVMapNode* getMapNode() { return _mapNode; }
protected:
    //!calculation of mean reprojection error of all matches
    void calculateMeanReprojectionError();
    //!calculate pose difference
    void calculatePoseDifference();
    //!draw found features to video stream image
    void decorateVideoWithKeyPoints(cv::Mat& image);
    //!show rectangle for key points in video that where matched to map points
    void decorateVideoWithKeyPointMatches(cv::Mat& image);
    //!decorate scene with matched map points, local map points and matched map points
    void decorateScene();
    //add map points to scene and keypoints to video image
    void decorateSceneAndVideo(cv::Mat& image);

    SLCVMapNode* _mapNode;
    //! flags, if map has changed (e.g. after key frame insertion or culling)
    bool _mapHasChanged = false;

    //mean reprojection error
    double _meanReprojectionError = -1.0;
    //L2 norm of the difference between the last and the current camera pose
    double _poseDifference = -1.0;

    // KeyFrame database for place recognition (relocalization and loop detection).
    SLCVKeyFrameDB* mpKeyFrameDatabase;
    //map containing map points
    SLCVMap* _map = NULL;

    // Current Frame
    SLCVFrame mCurrentFrame;
    //Last Frame, KeyFrame and Relocalisation Info
    SLCVFrame mLastFrame;

    //Local Map 
    //(maybe always the last inserted keyframe?)
    SLCVKeyFrame* mpReferenceKF = NULL;
    std::vector<SLCVMapPoint*> mvpLocalMapPoints;
    std::vector<SLCVKeyFrame*> mvpLocalKeyFrames;

    //Current matches in frame
    int mnMatchesInliers = 0;

    std::mutex _meanProjErrorLock;
    std::mutex _poseDiffLock;
    std::mutex _mapLock;
    std::mutex _nMapMatchesLock;
};

#endif //SLCVMAPTRACKING_H
