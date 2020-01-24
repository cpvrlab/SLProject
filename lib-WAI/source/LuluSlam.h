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
    initializerData mIniData;
    localMap        mLmap;
    WAIMap*         mGlobalMap;
    SLAMLatestState mLast;
    TrackingState   mState;
    LocalMapping*   mLocalMapping;
    LoopClosing*    mLoopClosing;
    list<cv::Mat>   mRelativeFramePoses;
    WAIKeyFrameDB*  mKeyFrameDatabase;
    KPextractor*    mExtractor;
    ORBVocabulary*  mVoc;
    cv::Mat         mDistortion;
    cv::Mat         mCameraIntrinsic;
    cv::Mat         mCameraExtrinsic;
    std::thread*    mLocalMappingThread = nullptr;
    std::thread*    mLoopClosingThread  = nullptr;
    std::mutex      mMutexStates;
    bool            mInitialized;
    WAIFrame        mLastFrame;

    LuluSLAM(cv::Mat intrinsic, cv::Mat distortion, std::string orbVocFile, KPextractor* extractorp);

    WAIFrame genFrame(cv::Mat imageGray);
    bool update(cv::Mat &imageGray, cv::Mat &imageRGB);

    cv::Mat getExtrinsic();

    static void drawInitInfo(initializerData& iniData, WAIFrame& newFrame, cv::Mat& imageRGB);

    static bool initialize(initializerData &iniData,
                          cv::Mat &camera,
                          cv::Mat &distortion,
                          ORBVocabulary *voc,
                          WAIMap *map,
                          WAIKeyFrameDB* keyFrameDatabase,
                          localMap &lmap,
                          LocalMapping *lmapper,
                          LoopClosing *loopClosing,
                          SLAMLatestState& last,
                          WAIFrame &frame,
                          list<cv::Mat> &relativeFramePoses);

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
                                   list<cv::Mat>&   relativeFramePoses,
                                   cv::Mat&         pose);

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

    static bool createInitialMapMonocular(initializerData &iniData,
                                         SLAMLatestState &last,
                                         ORBVocabulary *voc,
                                         WAIMap *map,
                                         LocalMapping *lmapper,
                                         LoopClosing *loopCloser,
                                         localMap &lmap,
                                         int mapPointsNeeded,
                                         WAIKeyFrameDB*    keyFrameDatabase,
                                         WAIFrame &frame,
                                         list<cv::Mat>&    relativeFramePoses);

    static void createNewKeyFrame(LocalMapping*    localMapper,
                                  localMap&        lmap,
                                  WAIMap*          map,
                                  WAIKeyFrameDB*   keyFrameDatabase,
                                  SLAMLatestState& last,
                                  WAIFrame&        frame);


    bool hasStateIdle();

    void requestStateIdle();

    WAIMap* getMap();

    WAIKeyFrameDB* getKfDB();

    bool retainImage();

    void resume();
    void reset();
    void setInitialized(bool b);
    bool isInitialized();

    KPextractor * getKPextractor();
    WAIFrame* getLastFrame();
    std::vector<WAIMapPoint*> getMapPoints();
    std::vector<WAIKeyFrame*> getKeyFrames();
    std::vector<WAIMapPoint*> getLocalMapPoints();
    std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> getMatchedCorrespondances(WAIFrame& frame);

    static void drawKeyPointInfo(WAIFrame &frame, cv::Mat& image);

    static void drawKeyPointMatches(WAIFrame &frame, cv::Mat& image);

    static std::vector<WAIMapPoint*> getMatchedMapPoints(WAIFrame* frame);

    std::string getPrintableState();

    int getKeyPointCount();

    int getKeyFrameCount();

    int getMapPointCount();

    cv::Mat getPose();
};
