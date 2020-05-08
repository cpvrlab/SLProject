#ifndef WAISLAM_H
#define WAISLAM_H

#include <WAIHelper.h>
#include <WAIKeyFrameDB.h>
#include <WAIMap.h>
#include <WAIModeOrbSlam2.h>
#include <OrbSlam/LocalMapping.h>
#include <OrbSlam/LoopClosing.h>
#include <OrbSlam/Initializer.h>
#include <OrbSlam/ORBmatcher.h>
#include <OrbSlam/Optimizer.h>
#include <OrbSlam/PnPsolver.h>
#include <LocalMap.h>
#include <opencv2/core.hpp>
#include <WAISlamTools.h>
#include <memory>
#include <fbow.h>

/* 
 * This class should not be instanciated. It contains only pure virtual methods
 * and some variables with getter that are useful for slam in a subclass.
 */

class WAISlam : public WAISlamTools
{
public:
    struct Params
    {
        //run local mapper and loopclosing serial to tracking
        bool serial = false;
        //retain the images in the keyframes, so we can store them later
        bool retainImg = false;
        //in onlyTracking mode we do not use local mapping and loop closing
        bool onlyTracking = false;
        //If true, keyframes loaded from a map will not be culled and the pose will not be changed. Local bundle adjustment is applied only on newly added kfs.
        //Also, the loop closing will be disabled so that there will be no optimization of the essential graph and no global bundle adjustment.
        bool fixOldKfs = false;
        //use lucas canade optical flow tracking
        bool trackOptFlow = false;

        //keyframe culling strategy params:
        // A keyframe is considered redundant if _cullRedundantPerc of the MapPoints it sees, are seen
        // in at least other 3 keyframes (in the same or finer scale)
        float cullRedundantPerc = 0.95f; //originally it was 0.9
    };

    WAISlam(const cv::Mat&          intrinsic,
            const cv::Mat&          distortion,
            fbow::Vocabulary*       voc,
            KPextractor*            iniExtractor,
            KPextractor*            extractor,
            std::unique_ptr<WAIMap> globalMap,
            bool                    trackingOnly      = false,
            bool                    serial            = false,
            bool                    retainImg         = false,
            float                   cullRedundantPerc = 0.95f);

    virtual ~WAISlam();

    virtual void reset();

    void createFrame(WAIFrame& frame, cv::Mat& imageGray);

    virtual void updatePose(WAIFrame& frame);
    virtual bool update(cv::Mat& imageGray);
    virtual void resume();

    virtual bool isTracking();
    virtual bool hasStateIdle();
    virtual void requestStateIdle();

    virtual bool retainImage();

    std::vector<WAIMapPoint*> getMatchedMapPoints(WAIFrame* frame);
    int                       getMatchedCorrespondances(WAIFrame* frame, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>& matching);

    virtual bool                      isInitialized() { return _initialized; }
    virtual WAIMap*                   getMap() { return _globalMap.get(); }
    virtual WAIFrame                  getLastFrame();
    virtual WAIFrame*                 getLastFramePtr();
    virtual std::vector<WAIMapPoint*> getLocalMapPoints() { return _localMap.mapPoints; }
    virtual int                       getNumKeyFrames() { return (int)_globalMap->KeyFramesInMap(); }

    virtual std::vector<WAIMapPoint*> getMapPoints()
    {
        if (_globalMap != nullptr)
            return _globalMap->GetAllMapPoints();
        return std::vector<WAIMapPoint*>();
    }

    virtual std::vector<WAIKeyFrame*> getKeyFrames()
    {
        if (_globalMap != nullptr)
            return _globalMap->GetAllKeyFrames();
        return std::vector<WAIKeyFrame*>();
    }

    virtual std::string getPrintableState()
    {
        switch (_state)
        {
            case WAI::TrackingState_Idle:
                return std::string("TrackingState_Idle\n");
                break;
            case WAI::TrackingState_Initializing:
                return std::string("TrackingState_Initializing");
                break;
            case WAI::TrackingState_None:
                return std::string("TrackingState_None");
                break;
            case WAI::TrackingState_TrackingLost:
                return std::string("TrackingState_TrackingLost");
                break;
            case WAI::TrackingState_TrackingOK:
                return std::string("TrackingState_TrackingOK");
                break;
        }
        return std::string("");
    }

    virtual int     getKeyPointCount() { return _lastFrame.N; }
    virtual int     getKeyFrameCount() { return (int)_globalMap->KeyFramesInMap(); }
    virtual int     getMapPointCount() { return (int)_globalMap->MapPointsInMap(); }
    virtual cv::Mat getPose();
    virtual void    setMap(std::unique_ptr<WAIMap> globalMap);

    virtual WAI::TrackingState getTrackingState() { return _state; }

    virtual void drawInfo(cv::Mat& imageRGB,
                          bool     showInitLine,
                          bool     showKeyPoints,
                          bool     showKeyPointsMatched);

    KPextractor* getKPextractor()
    {
        return _extractor;
    };

    int getMapPointMatchesCount() const;

    std::string getLoopCloseStatus();

    int getLoopCloseCount();

    int getKeyFramesInLoopCloseQueueCount();

protected:
    void updateState(WAI::TrackingState state);

    bool        _requestFinish;
    bool        _isFinish;
    bool        _isStop;
    std::mutex  _stateMutex;
    bool        finishRequested();
    void        requestFinish();
    bool        isStop();
    bool        isFinished();
    void        flushQueue();
    int         getNextFrame(WAIFrame& frame);
    static void updatePoseThread(WAISlam* ptr);

    WAI::TrackingState   _state = WAI::TrackingState_Idle;
    std::mutex           _cameraExtrinsicMutex;
    std::mutex           _mutexStates;
    std::mutex           _lastFrameMutex;
    bool                 _retainImg           = false;
    unsigned long        _lastRelocFrameId    = 0;
    unsigned long        _lastKeyFrameFrameId = 0;
    bool                 _serial              = false;
    bool                 _trackingOnly        = false;
    KPextractor*         _extractor           = nullptr;
    KPextractor*         _iniExtractor        = nullptr;
    int                  _infoMatchedInliners = 0;
    std::thread*         _poseUpdateThread;
    std::queue<WAIFrame> _framesQueue;
    std::mutex           _frameQueueMutex;
};

#endif
