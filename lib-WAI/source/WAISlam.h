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
#include <opencv2/core.hpp>

enum TrackingState
{
    TrackingState_None,
    TrackingState_Idle,
    TrackingState_Initializing,
    TrackingState_TrackingOK,
    TrackingState_TrackingLost
};

struct LocalMap
{
    WAIKeyFrame*              refKF;
    WAIKeyFrame*              lastKF;
    std::vector<WAIKeyFrame*> keyFrames;
    std::vector<WAIMapPoint*> mapPoints;
};

struct InitializerData
{
    Initializer*             initializer;
    WAIFrame                 initialFrame;
    WAIFrame                 secondFrame;
    std::vector<cv::Point2f> prevMatched;
    std::vector<cv::Point3f> iniPoint3D;
    std::vector<int>         iniMatches;
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
                           WAIKeyFrameDB*   keyFrameDatabase);

    static bool relocalization(WAIFrame&      currentFrame,
                               WAIMap*        waiMap,
                               WAIKeyFrameDB* keyFrameDatabase,
                               LocalMap&      localMap,
                               int&           inliers);

    static bool tracking(WAIMap*        map,
                         WAIKeyFrameDB* keyFrameDatabase,
                         LocalMap&      localMap,
                         WAIFrame&      frame,
                         WAIFrame&      lastFrame,
                         int            lastRelocFrameId,
                         cv::Mat&       velocity,
                         int&           inliers);

    static bool trackLocalMap(LocalMap& localMap,
                              WAIFrame& frame,
                              int       lastRelocFrameId,
                              int&      inliers);

    static void mapping(WAIMap*        map,
                        WAIKeyFrameDB* keyFrameDatabase,
                        LocalMap&      localMap,
                        LocalMapping*  localMapper,
                        WAIFrame&      frame,
                        int            inliers);

    static void serialMapping(WAIMap*        map,
                              WAIKeyFrameDB* keyFrameDatabase,
                              LocalMap&      localMap,
                              LocalMapping*  localMapper,
                              LoopClosing*   loopCloser,
                              WAIFrame&      frame,
                              int            inliers);

    static void motionModel(WAIFrame& frame,
                            WAIFrame& lastFrame,
                            cv::Mat&  velocity,
                            cv::Mat&  pose);

    static bool trackReferenceKeyFrame(LocalMap& map, WAIFrame& lastFrame, WAIFrame& frame);

    static bool trackWithMotionModel(cv::Mat velocity, WAIFrame& previousFrame, WAIFrame& frame);

    static void updateLocalMap(WAIFrame& frame, LocalMap& localMap);

    static int trackLocalMapPoints(LocalMap& localMap, int lastRelocFrameId, WAIFrame& frame);

    static bool needNewKeyFrame(WAIMap*       globalMap,
                                LocalMap&     localMap,
                                LocalMapping* localMapper,
                                WAIFrame&     frame,
                                int           nInliners);

    static bool genInitialMap(WAIMap*       globalMap,
                              LocalMapping* localMapper,
                              LoopClosing*  loopCloser,
                              LocalMap&     localMap,
                              bool          serial);

    static void createNewKeyFrame(LocalMapping*  localMapper,
                                  LocalMap&      localMap,
                                  WAIMap*        globalMap,
                                  WAIKeyFrameDB* keyFrameDatabase,
                                  WAIFrame&      frame);

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
    WAIKeyFrameDB*  _keyFrameDatabase;
    ORBVocabulary*  _voc;
    cv::Mat         _velocity;
    bool            _initialized;

    LocalMapping* _localMapping;
    LoopClosing*  _loopClosing;
    std::thread*  _localMappingThread = nullptr;
    std::thread*  _loopClosingThread  = nullptr;
};

class WAISlam : public WAISlamTools
{
public:
    WAISlam(cv::Mat      intrinsic,
            cv::Mat      distortion,
            std::string  orbVocFile,
            KPextractor* extractor);

    WAISlam(cv::Mat      intrinsic,
            cv::Mat      distortion,
            std::string  orbVocFile,
            KPextractor* extractor,
            bool         trackingOnly,
            bool         serial    = false,
            bool         retainImg = false);

    virtual void reset();
    virtual bool update(cv::Mat& imageGray);
    virtual void resume();

    virtual bool hasStateIdle();
    virtual void requestStateIdle();
    virtual bool retainImage();

    static std::vector<WAIMapPoint*>                                 getMatchedMapPoints(WAIFrame* frame);
    static std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> getMatchedCorrespondances(WAIFrame* frame);

    virtual bool                      isInitialized() { return _initialized; }
    virtual WAIMap*                   getMap() { return _globalMap; }
    virtual WAIKeyFrameDB*            getKfDB() { return _keyFrameDatabase; }
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

    virtual int           getKeyPointCount() { return _lastFrame.N; }
    virtual int           getKeyFrameCount() { return _globalMap->KeyFramesInMap(); }
    virtual int           getMapPointCount() { return _globalMap->MapPointsInMap(); }
    virtual cv::Mat       getPose() { return _cameraExtrinsic; }
    virtual TrackingState getTrackingState() { return _state; }
    virtual void          setState(TrackingState state)
    {
        if (state == TrackingState_Initializing)
        {
            _initialized = false;
            reset();
        }
        _state = state;
    }

    virtual void drawInfo(cv::Mat& imageRGB,
                          bool     showInitLine,
                          bool     showKeyPoints,
                          bool     showKeyPointsMatched);

    KPextractor* getKPextractor()
    {
        return _extractor;
    };

protected:
    std::mutex   _mutexStates;
    bool         _retainImg;
    int          _lastRelocId;
    bool         _serial;
    bool         _trackingOnly;
    KPextractor* _extractor;
};

class WAISlamMarker : public WAISlam
{
public:
    WAISlamMarker(cv::Mat      intrinsic,
                  cv::Mat      distortion,
                  std::string  orbVocFile,
                  KPextractor* extractor,
                  KPextractor* markerExtractor,
                  std::string  markerFile,
                  bool         serial    = false,
                  bool         retainImg = false);

    void     reset();
    WAIFrame createMarkerFrame(std::string markerFile, KPextractor* markerExtractor);

    bool doMarkerMapPreprocessing(std::string markerFile,
                                  cv::Mat&    nodeTransform,
                                  float       markerWidthInM);

    bool findMarkerHomography(WAIFrame&    markerFrame,
                              WAIKeyFrame* kfCand,
                              cv::Mat&     homography,
                              int          minMatches);

    bool update(cv::Mat& imageGray);
    void resume();

    bool hasStateIdle();
    void requestStateIdle();
    bool retainImage();

    std::vector<WAIMapPoint*> getMarkerCornerMapPoints();

private:
    WAIFrame     _markerFrame;
    KPextractor* _markerExtractor = nullptr;

    WAIMapPoint* _mpUL = nullptr;
    WAIMapPoint* _mpUR = nullptr;
    WAIMapPoint* _mpLL = nullptr;
    WAIMapPoint* _mpLR = nullptr;
};
