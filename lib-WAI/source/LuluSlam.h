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

struct SLAMLatestState
{
    WAIFrame     lastFrame;
    cv::Mat      lastFramePose;
    WAIKeyFrame* lastKeyFrame;
    unsigned int lastRelocFrameId;
    unsigned int lastFrameId;
    cv::Mat      velocity;
};

struct localMap
{
    WAIKeyFrame*              refKF;
    std::vector<WAIKeyFrame*> keyFrames;
    std::vector<WAIMapPoint*> mapPoints;
};

struct initializerData
{
    Initializer*             initializer;
    WAIFrame                 initialFrame;
    std::vector<cv::Point2f> prevMatched;
    std::vector<cv::Point3f> iniPoint3D;
    std::vector<int>         iniMatches;
};

class LuluSLAM
{
public:
    initializerData iniData;
    localMap        lmap;
    WAIMap*         globalMap;
    SLAMLatestState last;
    TrackingState   state;
    LocalMapping*   localMapping;
    LoopClosing*    loopClosing;
    list<cv::Mat>   relativeFramePoses;
    WAIKeyFrameDB*  keyFrameDatabase;
    KPextractor*    extractor;
    ORBVocabulary*  voc;
    std::thread*    localMappingThread = nullptr;
    std::thread*    loopClosingThread  = nullptr;

    LuluSLAM(std::string orbVocFile, KPextractor* extractorp);

    bool update(cv::Mat camera, cv::Mat dist, cv::Mat imageGray, cv::Mat imageRGB);

    static void drawInitInfo(initializerData& iniData, WAIFrame& newFrame, cv::Mat& imageRGB);

    static bool initialize(initializerData  iniData,
                           cv::Mat&         camera,
                           cv::Mat&         distortion,
                           ORBVocabulary*   voc,
                           WAIMap*          map,
                           WAIKeyFrameDB*   keyFrameDatabase,
                           localMap&        lmap,
                           LocalMapping*    lmapper,
                           SLAMLatestState& last,
                           WAIFrame&        frame,
                           list<cv::Mat>&   relativeFramePoses);

    static bool relocalization(WAIFrame&        currentFrame,
                               SLAMLatestState& last,
                               WAIMap*          waiMap,
                               WAIKeyFrameDB*   keyFrameDatabase);

    static bool trackingAndMapping(cv::Mat&         camera,
                                   cv::Mat&         distortion,
                                   WAIMap*          map,
                                   WAIKeyFrameDB*   keyFrameDatabase,
                                   SLAMLatestState& last,
                                   localMap&        localMap,
                                   LocalMapping*    localMapper,
                                   WAIFrame&        frame,
                                   list<cv::Mat>&   relativeFramePoses);

    static bool track(WAIFrame& frame, SLAMLatestState& last, localMap& localMap, list<cv::Mat>& relativeFramePoses);

    static bool trackReferenceKeyFrame(SLAMLatestState& last, localMap& map, WAIFrame& frame);

    static bool trackWithMotionModel(SLAMLatestState& last, WAIFrame& frame, list<cv::Mat>& relativeFramePoses);

    static void updateLocalMap(WAIFrame& frame,
                               localMap& lmap);

    static int matchLocalMapPoints(localMap&        lmap,
                                   SLAMLatestState& last,
                                   WAIFrame&        frame);

    static bool needNewKeyFrame(WAIMap*          map,
                                localMap&        lmap,
                                LocalMapping*    lmapper,
                                SLAMLatestState& last,
                                WAIFrame&        frame,
                                int              nInliners);

    static bool createInitialMapMonocular(initializerData& iniData,
                                          SLAMLatestState& last,
                                          ORBVocabulary*   voc,
                                          WAIMap*          map,
                                          LocalMapping*    lmapper,
                                          localMap&        lmap,
                                          int              mapPointsNeeded,
                                          WAIKeyFrameDB*   keyFrameDatabase,
                                          WAIFrame&        frame,
                                          list<cv::Mat>&   relativeFramePoses);

    static void createNewKeyFrame(LocalMapping*    localMapper,
                                  localMap&        lmap,
                                  WAIMap*          map,
                                  WAIKeyFrameDB*   keyFrameDatabase,
                                  SLAMLatestState& last,
                                  WAIFrame&        frame);
    ;
};
