#ifndef WAI_MODE_ORB_SLAM_2
#define WAI_MODE_ORB_SLAM_2

#include <thread>

#include <opencv2/core.hpp>

#include <WAIHelper.h>
#include <WAIMode.h>
#include <WAISensorCamera.h>
#include <WAIKeyFrameDB.h>
#include <WAIMap.h>
#include <WAIOrbVocabulary.h>

#include <SURFextractor.h>
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

class WAI_API ModeOrbSlam2 : public Mode
{
    public:
    ModeOrbSlam2(SensorCamera* camera,
                 bool          serial,
                 bool          retainImg,
                 bool          onlyTracking,
                 bool          trackOptFlow,
                 std::string   orbVocFile);
    ~ModeOrbSlam2();
    bool getPose(cv::Mat* pose);
    void notifyUpdate();

    void reset();
    bool isInitialized();

    void disableMapping();

    void enableMapping();

    WAIMap*        getMap() { return _map; }
    WAIKeyFrameDB* getKfDB() { return mpKeyFrameDatabase; }

    // New KeyFrame rules (according to fps)
    // Max/Min Frames to insert keyframes and to check relocalisation
    int mMinFrames = 0;
    int mMaxFrames = 30; //= fps

    // Debug functions
    std::string getPrintableState();
    std::string getPrintableType();
    uint32_t    getMapPointCount();
    uint32_t    getMapPointMatchesCount();
    uint32_t    getKeyFrameCount();
    int         getNMapMatches();
    int         getNumKeyFrames();
    float       poseDifference();
    float       getMeanReprojectionError();
    void        findMatches(std::vector<cv::Point2f> &vP2D, std::vector<cv::Point3f> &vP3Dw);

    std::string getLoopCloseStatus();
    uint32_t    getLoopCloseCount();
    uint32_t    getKeyFramesInLoopCloseQueueCount();

    std::vector<WAIMapPoint*>                                 getMapPoints();
    std::vector<WAIMapPoint*>                                 getMatchedMapPoints();
    std::vector<WAIMapPoint*>                                 getLocalMapPoints();
    std::vector<WAIKeyFrame*>                                 getKeyFrames();
    std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> getMatchedCorrespondances();
    std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> getCorrespondances();

    void showKeyPointsMatched(const bool flag)
    {
        _showKeyPointsMatched = flag;
    }

    void showKeyPoints(const bool flag)
    {
        _showKeyPoints = flag;
    }

    bool getTrackOptFlow();
    void setTrackOptFlow(bool flag);

    // state machine
    void pause();
    void resume();
    void requestStateIdle();
    bool hasStateIdle();
    void setInitialized(bool initialized) { _initialized = initialized; }

    void loadMapData(std::vector<WAIKeyFrame*> keyFrames, std::vector<WAIMapPoint*> mapPoints, int numLoopClosings);

    private:
    enum TrackingState
    {
        TrackingState_None,
        TrackingState_Idle,
        TrackingState_Initializing,
        TrackingState_TrackingOK,
        TrackingState_TrackingLost
    };

    enum TrackingType
    {
        TrackingType_None,
        TrackingType_ORBSLAM,
        TrackingType_MotionModel,
        TrackingType_OptFlow
    };

    void initialize();
    bool createInitialMapMonocular();
    void track3DPts();

    bool relocalization();
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
    void globalBundleAdjustment();

    WAIKeyFrame* currentKeyFrame();

    cv::Mat _pose;

    bool _poseSet = false;
    bool _serial;
    bool _retainImg;
    bool _initialized;
    bool _onlyTracking;
    bool _trackOptFlow;

    SensorCamera*  _camera            = nullptr;
    TrackingState  _state             = TrackingState_None;
    TrackingType   _trackingType      = TrackingType_None;
    WAIKeyFrameDB* mpKeyFrameDatabase = nullptr;
    WAIMap*        _map               = nullptr;

    ORB_SLAM2::ORBVocabulary* mpVocabulary      = nullptr;
    ORB_SLAM2::KPextractor*   _extractor        = nullptr;
    ORB_SLAM2::KPextractor*   mpIniORBextractor = nullptr;
    ORB_SLAM2::LocalMapping*  mpLocalMapper     = nullptr;
    ORB_SLAM2::LoopClosing*   mpLoopCloser      = nullptr;
    ORB_SLAM2::Initializer*   mpInitializer     = nullptr;

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
    bool                     _bOK;
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
    void decorate();
    void calculateMeanReprojectionError();
    void calculatePoseDifference();
    void decorateVideoWithKeyPoints(cv::Mat& image);
    void decorateVideoWithKeyPointMatches(cv::Mat& image);

    double _meanReprojectionError = -1.0;
    double _poseDifference        = -1.0;
    bool   _showKeyPoints         = false;
    bool   _showKeyPointsMatched  = true;
    bool   _showMapPC             = true;
    bool   _showMatchesPC         = true;
    bool   _showLocalMapPC        = false;
    bool   _showKeyFrames         = true;
    bool   _showCovisibilityGraph = false;
    bool   _showSpanningTree      = true;
    bool   _showLoopEdges         = true;
    bool   _renderKfBackground    = false;
    bool   _allowKfsAsActiveCam   = false;
};
}

#endif
