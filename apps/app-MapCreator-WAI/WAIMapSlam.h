#ifndef WAISLAMSMAP_H
#define WAISLAMSMAP_H

#include <WAIHelper.h>
#include <WAIKeyFrameDB.h>
#include <WAIMap.h>
#include <OrbSlam/LocalMapping.h>
#include <OrbSlam/LoopClosing.h>
#include <OrbSlam/Initializer.h>
#include <OrbSlam/ORBmatcher.h>
#include <OrbSlam/Optimizer.h>
#include <OrbSlam/PnPsolver.h>
#include <LocalMap.h>
#include <opencv2/core.hpp>
#include <WAISlamTools.h>
#include <WAIMap.h>
#include <WAIMapPoint.h>
#include <WAIKeyFrame.h>
#include <memory>

/* 
 * This class should not be instanciated. It contains only pure virtual methods
 * and some variables with getter that are useful for slam in a subclass.
 */
class WAIMapSlam : public WAISlamTools
{
public:
    struct Params
    {
        bool  serial            = false;
        bool  retainImg         = false;
        bool  fixOldKfs         = false;
        float cullRedundantPerc = 0.99f;
    };

    WAIMapSlam(const cv::Mat&          intrinsic,
               const cv::Mat&          distortion,
               WAIOrbVocabulary*       voc,
               KPextractor*            extractor,
               std::unique_ptr<WAIMap> globalMap,
               WAIMapSlam::Params      p);

    virtual ~WAIMapSlam();

    virtual void reset();

    void changeIntrinsic(cv::Mat intrinsic, cv::Mat distortion);
    void createFrame(WAIFrame& frame, cv::Mat& imageGray);

    virtual void updatePose(WAIFrame& frame);
    virtual void updatePose2(WAIFrame& frame);
    virtual bool update(cv::Mat& imageGray);
    virtual bool update2(cv::Mat& imageGray);
    virtual void resume(){};

    virtual bool isTracking();
    virtual bool hasStateIdle() { return _state == WAI::TrackingState_Idle; };
    virtual bool retainImage() {return false;};
    virtual void requestStateIdle();

    bool needNewKeyFrame2(LocalMapping* localMapper, WAIFrame& frame, int nInliers, const unsigned long lastKeyFrameFrameId);

    void transformCoords(cv::Mat transform);

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

    virtual int     getKeyPointCount() { return _lastFrame.N; }
    virtual int     getKeyFrameCount() { return (int)_globalMap->KeyFramesInMap(); }
    virtual int     getMapPointCount() { return (int)_globalMap->MapPointsInMap(); }
    virtual cv::Mat getPose();
    virtual void    setMap(std::unique_ptr<WAIMap> globalMap);

    virtual WAI::TrackingState getTrackingState() { return _state; }

    virtual void drawInfo(cv::Mat& imageRGB,
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

    Params _params;

    WAI::TrackingState   _state = WAI::TrackingState_Idle;
    std::mutex           _cameraExtrinsicMutex;
    std::mutex           _mutexStates;
    std::mutex           _lastFrameMutex;
    bool                 _retainImg           = false;
    unsigned long        _lastRelocFrameId    = 0;
    unsigned long        _lastKeyFrameFrameId = 0;
    KPextractor*         _extractor           = nullptr;
    int                  _infoMatchedInliners = 0;
    std::thread*         _poseUpdateThread;
    std::queue<WAIFrame> _framesQueue;
    std::mutex           _frameQueueMutex;
};

#endif
