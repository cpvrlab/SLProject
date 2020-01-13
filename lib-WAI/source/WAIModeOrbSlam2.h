#ifndef WAI_MODE_ORB_SLAM_2
#define WAI_MODE_ORB_SLAM_2

#include <thread>

#include <opencv2/core.hpp>

#include <WAIHelper.h>
#include <WAIKeyFrameDB.h>
#include <WAIMap.h>
#include <WAIOrbVocabulary.h>

#include <OrbSlam/SURFextractor.h>
#include <OrbSlam/LocalMapping.h>
#include <OrbSlam/LoopClosing.h>
#include <OrbSlam/Initializer.h>
#include <OrbSlam/ORBmatcher.h>
#include <OrbSlam/Optimizer.h>
#include <OrbSlam/PnPsolver.h>

#define OPTFLOW_GRID_COLS 7
#define OPTFLOW_GRID_ROWS 4

namespace WAI
{

enum TrackingState
{
    TrackingState_None,
    TrackingState_Idle,
    TrackingState_Initializing,
    TrackingState_TrackingOK,
    TrackingState_TrackingLost
};

class WAI_API ModeOrbSlam2
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

    ModeOrbSlam2(cv::Mat       cameraMat,
                 cv::Mat       distortionMat,
                 const Params& params,
                 std::string   orbVocFile,
                 bool          applyMinAccScoreFilter = false,
                 std::string   markerFile = "");
    ~ModeOrbSlam2();
    bool getPose(cv::Mat* pose);
    bool update(cv::Mat& imageGray, cv::Mat& imageRGB);

    static bool relocalization(WAIFrame&      currentFrame,
                               WAIKeyFrameDB* keyFrameDB,
                               unsigned int*  lastRelocFrameId,
                               WAIMap&        waiMap,
                               bool           applyMinAccScoreFilter = true,
                               bool           relocWithAllKFs = false);

    void reset();
    bool isInitialized();

    void disableMapping();
    void enableMapping();

    WAIMap*        getMap() { return _map; }
    WAIKeyFrameDB* getKfDB() { return mpKeyFrameDatabase; }

    // New KeyFrame rules (according to fps)
    // Max/Min Frames to insert keyframes and to check relocalisation
    int mMinFrames = 0;
    int mMaxFrames = 30; //= fps (max number of frames between keyframes)

    // Debug functions
    std::string   getPrintableState();
    TrackingState getTrackingState() { return _state; }
    std::string   getPrintableType();
    int           getKeyPointCount();
    int           getMapPointCount();
    int           getMapPointMatchesCount();
    int           getKeyFrameCount();
    int           getNMapMatches();
    int           getNumKeyFrames();
    float         poseDifference();
    float         getMeanReprojectionError();
    void          findMatches(std::vector<cv::Point2f>& vP2D, std::vector<cv::Point3f>& vP3Dw);

    std::string getLoopCloseStatus();
    int         getLoopCloseCount();
    int         getKeyFramesInLoopCloseQueueCount();

    std::vector<WAIMapPoint*>                                 getMapPoints();
    std::vector<WAIMapPoint*>                                 getMatchedMapPoints();
    std::vector<WAIMapPoint*>                                 getLocalMapPoints();
    std::vector<WAIMapPoint*>                                 getMarkerCornerMapPoints();
    std::vector<WAIKeyFrame*>                                 getKeyFrames();
    std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> getMatchedCorrespondances();
    std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> getCorrespondances();

    KPextractor* getKPextractor()
    {
        return mpExtractor;
    }

    bool getTrackOptFlow();
    void setTrackOptFlow(bool flag);

    // state machine
    void pause();
    void resume();
    void requestStateIdle();
    bool hasStateIdle();
    bool retainImage() { return _params.retainImg; }
    void setInitialized(bool initialized) { _initialized = initialized; }

    void setExtractor(KPextractor* extractor, KPextractor* iniExtractor, KPextractor* markerExtractor = nullptr);
    void setVocabulary(std::string orbVocFile);

    WAIFrame getCurrentFrame();

    bool doMarkerMapPreprocessing(std::string markerFile, cv::Mat& nodeTransform, float markerWidthInM);
    void decorateVideoWithKeyPoints(cv::Mat& image);
    void decorateVideoWithKeyPointMatches(cv::Mat& image);

    // marker correction stuff
    WAIFrame createMarkerFrame(std::string  markerFile,
                               KPextractor* markerExtractor);

private:
    enum TrackingType
    {
        TrackingType_None,
        TrackingType_ORBSLAM,
        TrackingType_MotionModel,
        TrackingType_OptFlow
    };

    void initialize(cv::Mat& imageGray, cv::Mat& imageRGB);
    bool createInitialMapMonocular(int mapPointsNeeded);
    void track3DPts(cv::Mat& imageGray, cv::Mat& imageRGB);

    //bool        relocalization();
    bool trackReferenceKeyFrame();
    bool trackWithMotionModel();
    bool trackLocalMap();
    bool trackWithOptFlow();

    bool needNewKeyFrame();
    void createNewKeyFrame();

    bool posInGrid(const cv::KeyPoint& kp, int& posX, int& posY, int minX, int minY);
    void checkReplacedInLastFrame();
    void updateLocalMap();
    void updateLocalKeyFrames();
    void updateLocalPoints();
    void searchLocalPoints();
    void updateLastFrame();
    //void globalBundleAdjustment();

    WAIKeyFrame* currentKeyFrame();

    cv::Mat _pose;

    bool   _applyMinAccScoreFilter;
    bool   _poseSet = false;
    bool   _initialized;
    Params _params;

    cv::Mat _cameraMat;
    cv::Mat _distortionMat;

    TrackingState  _state             = TrackingState_None;
    TrackingType   _trackingType      = TrackingType_None;
    WAIKeyFrameDB* mpKeyFrameDatabase = nullptr;
    WAIMap*        _map               = nullptr;

    ORB_SLAM2::ORBVocabulary* mpVocabulary   = nullptr;
    ORB_SLAM2::KPextractor*   mpExtractor    = nullptr;
    ORB_SLAM2::KPextractor*   mpIniExtractor = nullptr;

    ORB_SLAM2::LocalMapping* mpLocalMapper = nullptr;
    ORB_SLAM2::LoopClosing*  mpLoopCloser  = nullptr;
    ORB_SLAM2::Initializer*  mpInitializer = nullptr;

    std::thread* mptLocalMapping = nullptr;
    std::thread* mptLoopClosing  = nullptr;

    std::mutex _meanProjErrorLock;
    std::mutex _poseDiffLock;
    std::mutex _mapLock;
    std::mutex _nMapMatchesLock;
    std::mutex _optFlowLock;

    WAIFrame                 mCurrentFrame;
    WAIFrame                 mInitialFrame;
    WAIFrame                 mLastFrame;
    std::vector<int>         mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    bool                     _bOK           = false;
    bool                     _mapHasChanged = false;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO = false;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat>      mlRelativeFramePoses;
    list<WAIKeyFrame*> mlpReferences;
    list<double>       mlFrameTimes;
    list<bool>         mlbLost;

    //C urrent matches in frame
    int mnMatchesInliers = 0;

    //Last Frame, KeyFrame and Relocalisation Info
    WAIKeyFrame* mpLastKeyFrame     = nullptr;
    unsigned int mnLastRelocFrameId = 0;
    unsigned int mnLastKeyFrameId;

    // Local Map
    // (maybe always the last inserted keyframe?)
    WAIKeyFrame*              mpReferenceKF = nullptr;
    std::vector<WAIMapPoint*> mvpLocalMapPoints;
    std::vector<WAIKeyFrame*> mvpLocalKeyFrames;

    // Motion Model
    cv::Mat mVelocity;

    // optical flow
    bool                 _optFlowOK = false;
    cv::Mat              _optFlowTcw;
    vector<WAIMapPoint*> _optFlowMapPtsLastFrame;
    vector<cv::KeyPoint> _optFlowKeyPtsLastFrame;
    float                _optFlowGridElementWidthInv;
    float                _optFlowGridElementHeightInv;

    // state machine
    void stateTransition();
    void resetRequests();
    void requestResume();

    bool       _idleRequested   = false;
    bool       _resumeRequested = false;
    std::mutex _mutexStates;

    // debug visualization
    void decorate(cv::Mat& image);
    void calculateMeanReprojectionError();
    void calculatePoseDifference();

    double _meanReprojectionError = -1.0;
    double _poseDifference        = -1.0;
    bool   _showMapPC             = true;
    bool   _showMatchesPC         = true;
    bool   _showLocalMapPC        = false;
    bool   _showKeyFrames         = true;
    bool   _showCovisibilityGraph = false;
    bool   _showSpanningTree      = true;
    bool   _showLoopEdges         = true;
    bool   _renderKfBackground    = false;
    bool   _allowKfsAsActiveCam   = false;

    // marker correction stuff
    bool findMarkerHomography(WAIFrame&    markerFrame,
                              WAIKeyFrame* kfCand,
                              cv::Mat&     homography,
                              int          minMatches);

    bool _createMarkerMap;

    std::string             _markerFile;
    WAIFrame                _markerFrame;
    ORB_SLAM2::KPextractor* _markerExtractor;

    WAIMapPoint* _mpUL;
    WAIMapPoint* _mpUR;
    WAIMapPoint* _mpLL;
    WAIMapPoint* _mpLR;
};
}

#endif
