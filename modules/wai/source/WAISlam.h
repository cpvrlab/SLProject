#ifndef WAISLAM_H
#define WAISLAM_H

#include <WAIHelper.h>
#include <WAIKeyFrameDB.h>
#include <WAIMap.h>
#include <orb_slam/LocalMapping.h>
#include <orb_slam/LoopClosing.h>
#include <orb_slam/Initializer.h>
#include <orb_slam/ORBmatcher.h>
#include <orb_slam/Optimizer.h>
#include <orb_slam/PnPsolver.h>
#include <LocalMap.h>
#include <opencv2/core.hpp>
#include <WAISlamTools.h>
#include <WAIMap.h>
#include <WAIMapPoint.h>
#include <WAIKeyFrame.h>
#include <memory>

//-----------------------------------------------------------------------------
/*
 * This class should not be instanciated. It contains only pure virtual methods
 * and some variables with getter that are useful for slam in a subclass.
 */
class WAISlam : public WAISlamTools
{
public:
    struct Params
    {
        // ensure all new keyframe have enough in common with loaded map
        bool ensureKFIntegration = false;
        // wait for localmapping
        bool serial = false;
        // retain the images in the keyframes, so we can store them later
        bool retainImg = false;
        // in onlyTracking mode we do not use local mapping and loop closing
        bool onlyTracking = false;
        // If true, keyframes loaded from a map will not be culled and the pose will not be changed. Local bundle adjustment is applied only on newly added kfs.
        // Also, the loop closing will be disabled so that there will be no optimization of the essential graph and no global bundle adjustment.
        bool fixOldKfs = false;
        // use lucas canade optical flow tracking
        bool trackOptFlow = false;

        // keyframe culling strategy params:
        //  A keyframe is considered redundant if _cullRedundantPerc of the MapPoints it sees, are seen
        //  in at least other 3 keyframes (in the same or finer scale)
        float cullRedundantPerc = 0.95f; // originally it was 0.9

        // Min common words as a factor of max common words within candidates
        //  for relocalization and loop closing
        float minCommonWordFactor = 0.8f;

        // Min acceleration score filter in detectRelocalizationCandidates
        bool minAccScoreFilter = false;
    };

    WAISlam(const cv::Mat&          intrinsic,
            const cv::Mat&          distortion,
            WAIOrbVocabulary*       voc,
            KPextractor*            iniExtractor,
            KPextractor*            relocExtractor,
            KPextractor*            extractor,
            std::unique_ptr<WAIMap> globalMap,
            WAISlam::Params         params);

    virtual ~WAISlam();

    virtual void reset();

    void changeIntrinsic(cv::Mat intrinsic, cv::Mat distortion);
    void createFrame(WAIFrame& frame, cv::Mat& imageGray);

    virtual void updatePose(WAIFrame& frame);
    virtual bool update(cv::Mat& imageGray);
    virtual void updatePoseKFIntegration(WAIFrame& frame);
    virtual void resume();

    virtual bool isTracking();
    virtual bool hasStateIdle();
    virtual void requestStateIdle();

    virtual bool retainImage();

    void transformCoords(cv::Mat transform);

    std::vector<WAIMapPoint*> getMatchedMapPoints(WAIFrame* frame);
    int                       getMatchedCorrespondances(WAIFrame*                            frame,
                                                        std::pair<std::vector<cv::Point2f>,
                                            std::vector<cv::Point3f>>& matching);

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
            case WAITrackingState::Idle:
                return std::string("TrackingState_Idle\n");
                break;
            case WAITrackingState::Initializing:
                return std::string("Initializing");
                break;
            case WAITrackingState::None:
                return std::string("None");
                break;
            case WAITrackingState::TrackingLost:
                return std::string("TrackingLost");
                break;
            case WAITrackingState::TrackingOK:
                return std::string("TrackingOK");
                break;
            case WAITrackingState::TrackingStart:
                return std::string("TrackingStart");
                break;
            default:
                return std::string("");
        }
    }

    virtual int getKeyPointCount() { return _lastFrame.N; }
    virtual int getKeyFrameCount() { return (int)_globalMap->KeyFramesInMap(); }
    virtual int getMapPointCount() { return (int)_globalMap->MapPointsInMap(); }
    // get camera extrinsic
    virtual cv::Mat getPose();
    // set camera extrinsic guess
    virtual void setCamExrinsicGuess(cv::Mat extrinsicGuess);
    virtual void setMap(std::unique_ptr<WAIMap> globalMap);

    virtual WAITrackingState getTrackingState() { return _state; }

    virtual void drawInfo(cv::Mat& imageBGR,
                          float    scale,
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
    void updateState(WAITrackingState state);

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

    WAITrackingState _state = WAITrackingState::Idle;
    std::mutex       _cameraExtrinsicMutex;
    std::mutex       _cameraExtrinsicGuessMutex;
    std::mutex       _mutexStates;
    std::mutex       _lastFrameMutex;

    WAISlam::Params _params;

    unsigned int         _relocFrameCounter   = 0;
    unsigned long        _lastRelocFrameId    = 0;
    unsigned long        _lastKeyFrameFrameId = 0;
    KPextractor*         _extractor           = nullptr;
    KPextractor*         _relocExtractor      = nullptr;
    KPextractor*         _iniExtractor        = nullptr;
    int                  _infoMatchedInliners = 0;
    std::thread*         _poseUpdateThread;
    std::queue<WAIFrame> _framesQueue;
    std::mutex           _frameQueueMutex;
};
//-----------------------------------------------------------------------------
#endif
