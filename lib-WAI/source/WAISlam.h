#ifndef WAISLAM_H
#define WAISLAM_H
#include <WAIHelper.h>
#include <WAIKeyFrameDB.h>
#include <WAIMap.h>
#include <WAIOrbVocabulary.h>
#include <OrbSlam/LocalMapping.h>
#include <OrbSlam/LoopClosing.h>
#include <OrbSlam/Initializer.h>
#include <OrbSlam/ORBmatcher.h>
#include <OrbSlam/Optimizer.h>
#include <OrbSlam/PnPsolver.h>
#include <LocalMap.h>
#include <opencv2/core.hpp>

enum TrackingState
{
    TrackingState_None,
    TrackingState_Idle,
    TrackingState_Initializing,
    TrackingState_TrackingOK,
    TrackingState_TrackingLost
};

struct InitializerData
{
    Initializer*             initializer;
    WAIFrame                 initialFrame;
    std::vector<cv::Point2f> prevMatched; //all keypoints in initialFrame
    std::vector<cv::Point3f> iniPoint3D;
    std::vector<int>         iniMatches; //has length of keypoints of initial frame and contains matched keypoint index in current frame
};

/* 
 * This class should not be instanciated. It contains only pure virtual methods
 * and some variables with getter that are useful for slam in a subclass.
 */

class WAISlamTools
{
public:
    static void drawKeyPointInfo(WAIFrame& frame, cv::Mat& image);
    static void drawKeyPointMatches(WAIFrame& frame, cv::Mat& image);
    static void drawInitInfo(InitializerData& iniData, WAIFrame& frame, cv::Mat& imageRGB);

    static bool initialize(InitializerData& iniData,
                           WAIFrame&        frame,
                           ORBVocabulary*   voc,
                           LocalMap&        localMap,
                           int              mapPointsNeeded,
                           unsigned long&   lastKeyFrameFrameId);

    static bool genInitialMap(WAIMap*       globalMap,
                              LocalMapping* localMapper,
                              LoopClosing*  loopCloser,
                              LocalMap&     localMap,
                              bool          serial);

    static bool oldInitialize(WAIFrame&        frame,
                              InitializerData& iniData,
                              WAIMap*          map,
                              LocalMap&        localMap,
                              LocalMapping*    localMapper,
                              LoopClosing*     loopCloser,
                              ORBVocabulary*   voc,
                              int              mapPointsNeeded,
                              unsigned long&   lastKeyFrameFrameId);

    static bool relocalization(WAIFrame& currentFrame,
                               WAIMap*   waiMap,
                               LocalMap& localMap,
                               int&      inliers);

    static bool tracking(WAIMap*   map,
                         LocalMap& localMap,
                         WAIFrame& frame,
                         WAIFrame& lastFrame,
                         int       lastRelocFrameId,
                         cv::Mat&  velocity,
                         int&      inliers);

    static bool trackLocalMap(LocalMap& localMap,
                              WAIFrame& frame,
                              int       lastRelocFrameId,
                              int&      inliers);

    static void mapping(WAIMap*             map,
                        LocalMap&           localMap,
                        LocalMapping*       localMapper,
                        WAIFrame&           frame,
                        int                 inliers,
                        const unsigned long lastRelocFrameId,
                        unsigned long&      lastKeyFrameFrameId);

    static void serialMapping(WAIMap*             map,
                              LocalMap&           localMap,
                              LocalMapping*       localMapper,
                              LoopClosing*        loopCloser,
                              WAIFrame&           frame,
                              int                 inliers,
                              const unsigned long lastRelocFrameId,
                              unsigned long&      lastKeyFrameFrameId);

    static void motionModel(WAIFrame& frame,
                            WAIFrame& lastFrame,
                            cv::Mat&  velocity,
                            cv::Mat&  pose);

    static bool trackReferenceKeyFrame(LocalMap& map, WAIFrame& lastFrame, WAIFrame& frame);

    static bool trackWithMotionModel(cv::Mat velocity, WAIFrame& previousFrame, WAIFrame& frame);

    static void updateLocalMap(WAIFrame& frame, LocalMap& localMap);

    static int trackLocalMapPoints(LocalMap& localMap, int lastRelocFrameId, WAIFrame& frame);

    static bool needNewKeyFrame(WAIMap*             globalMap,
                                LocalMap&           localMap,
                                LocalMapping*       localMapper,
                                WAIFrame&           frame,
                                int                 nInliners,
                                const unsigned long lastRelocFrameId,
                                const unsigned long lastKeyFrameFrameId);

    static void createNewKeyFrame(LocalMapping*  localMapper,
                                  LocalMap&      localMap,
                                  WAIMap*        globalMap,
                                  WAIFrame&      frame,
                                  unsigned long& lastKeyFrameFrameId);

    static WAIFrame createMarkerFrame(std::string    markerFile,
                                      KPextractor*   markerExtractor,
                                      const cv::Mat& markerCameraIntrinsic,
                                      ORBVocabulary* voc);
    static bool     findMarkerHomography(WAIFrame&    markerFrame,
                                         WAIKeyFrame* kfCand,
                                         cv::Mat&     homography,
                                         int          minMatches);
    static bool     doMarkerMapPreprocessing(std::string    markerFile,
                                             cv::Mat&       nodeTransform,
                                             float          markerWidthInM,
                                             KPextractor*   markerExtractor,
                                             WAIMap*        map,
                                             const cv::Mat& markerCameraIntrinsic,
                                             ORBVocabulary* voc);

protected:
    WAISlamTools(){};

    TrackingState   _state = TrackingState_Idle;
    cv::Mat         _distortion;
    cv::Mat         _cameraIntrinsic;
    cv::Mat         _cameraExtrinsic;
    InitializerData _iniData;
    WAIFrame        _lastFrame;
    LocalMap        _localMap;
    WAIMap*         _globalMap;
    ORBVocabulary*  _voc;
    cv::Mat         _velocity;
    bool            _initialized;

    LocalMapping*             _localMapping;
    LoopClosing*              _loopClosing;
    std::thread*              _processNewKeyFrameThread = nullptr;
    std::vector<std::thread*> _mappingThreads;
    std::thread*              _loopClosingThread = nullptr;
};

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

    WAISlam(cv::Mat        intrinsic,
            cv::Mat        distortion,
            ORBVocabulary* voc,
            KPextractor*   iniExtractor,
            KPextractor*   extractor,
            WAIMap*        globalMap,
            bool           trackingOnly      = false,
            bool           serial            = false,
            bool           retainImg         = false,
            float          cullRedundantPerc = 0.95f);

    ~WAISlam();

    virtual void reset();

    void createFrame(WAIFrame &frame, cv::Mat& imageGray);

    virtual void update(WAIFrame& frame);
    virtual bool update(cv::Mat& imageGray);
    virtual void resume();

    virtual bool isTracking();
    virtual bool hasStateIdle();
    virtual void requestStateIdle();
    virtual bool retainImage();

    static std::vector<WAIMapPoint*>                                 getMatchedMapPoints(WAIFrame* frame);
    static std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> getMatchedCorrespondances(WAIFrame* frame);

    virtual bool                      isInitialized() { return _initialized; }
    virtual WAIMap*                   getMap() { return _globalMap; }
    virtual WAIFrame*                 getLastFrame() { return &_lastFrame; }
    virtual std::vector<WAIMapPoint*> getLocalMapPoints() { return _localMap.mapPoints; }
    virtual int                       getNumKeyFrames() { return _globalMap->KeyFramesInMap(); }

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
            case TrackingState_Idle:
                return std::string("TrackingState_Idle\n");
                break;
            case TrackingState_Initializing:
                return std::string("TrackingState_Initializing");
                break;
            case TrackingState_None:
                return std::string("TrackingState_None");
                break;
            case TrackingState_TrackingLost:
                return std::string("TrackingState_TrackingLost");
                break;
            case TrackingState_TrackingOK:
                return std::string("TrackingState_TrackingOK");
                break;
        }
    }

    virtual int     getKeyPointCount() { return _lastFrame.N; }
    virtual int     getKeyFrameCount() { return _globalMap->KeyFramesInMap(); }
    virtual int     getMapPointCount() { return _globalMap->MapPointsInMap(); }
    virtual cv::Mat getPose();
    virtual void    setMap(WAIMap* globalMap);

    virtual TrackingState getTrackingState() { return _state; }

    virtual void drawInfo(cv::Mat& imageRGB,
                          bool     showInitLine,
                          bool     showKeyPoints,
                          bool     showKeyPointsMatched);

    KPextractor* getKPextractor()
    {
        return _extractor;
    };

    int getMapPointMatchesCount();

    std::string getLoopCloseStatus();

    int getLoopCloseCount();

    int getKeyFramesInLoopCloseQueueCount();

protected:

    void updateState(TrackingState state);

    std::mutex    _cameraExtrinsicMutex;
    std::mutex    _mutexStates;
    std::mutex    _lastFrameMutex;
    bool          _retainImg           = false;
    unsigned long _lastRelocFrameId    = 0;
    unsigned long _lastKeyFrameFrameId = 0;
    bool          _serial              = false;
    bool          _trackingOnly        = false;
    KPextractor*  _extractor           = nullptr;
    KPextractor*  _iniExtractor        = nullptr;
    int           _infoMatchedInliners = 0;
};

#endif
