#ifndef WAI_MODE_ORB_SLAM_2_DATA_ORIENTED
#define WAI_MODE_ORB_SLAM_2_DATA_ORIENTED

#include <WAIPlatform.h>
#include <OrbSlamDataOriented/WAIOrbPattern.h>
#include <WAISensorCamera.h>
#include <OrbSlamDataOriented/WAIOrbExtraction.h>

#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

#define FRAME_GRID_ROWS 36 //48
#define FRAME_GRID_COLS 64

enum OrbSlamStatus
{
    OrbSlamStatus_None,
    OrbSlamStatus_Initializing,
    OrbSlamStatus_Tracking
};

struct MapPointTrackingInfos
{
    bool32 inView;
    r32    projX;
    r32    projY;
    r32    scaleLevel;
    r32    viewCos;
};

struct KeyFrame;

struct MapPoint
{
    i32 index;

    cv::Mat                  position;
    cv::Mat                  normalVector;
    cv::Mat                  descriptor;
    std::map<KeyFrame*, i32> observations; // key is pointer to a keyframe, value is index into that keyframes keypoint vector

    bool32 bad;

    r32 maxDistance;
    r32 minDistance;

    KeyFrame* referenceKeyFrame;
    i32       firstObservationFrameId;
    i32       foundInKeyFrameCounter;
    i32       visibleInKeyFrameCounter;

    i32 trackReferenceForFrame;
    i32 lastFrameSeen;
    i32 localBundleAdjustmentKeyFrameIndex; // used only during local BA

    MapPointTrackingInfos trackingInfos; // TODO(jan): this should not be in here
};

struct KeyFrameRelocalizationData
{
    i32 queryId;
    i32 words;
    r32 score;
};

struct KeyFrame
{
    i32    index;
    i32    frameId;
    i32    numberOfKeyPoints;
    bool32 bad;

    cv::Mat cameraMat;

    std::vector<cv::KeyPoint> keyPoints; // only used for visualization
    std::vector<cv::KeyPoint> undistortedKeyPoints;
    std::vector<MapPoint*>    mapPointMatches; // same size as keyPoints, initialized with -1

    std::vector<size_t> keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    cv::Mat             descriptors;
    cv::Mat             cTw;
    cv::Mat             wTc;
    cv::Mat             worldOrigin;

    KeyFrame*              parent;
    std::vector<KeyFrame*> children; // children in covisibility graph
    i32                    trackReferenceForFrame;
    i32                    localBundleAdjustmentKeyFrameIndex; // used only during local BA
    i32                    localBundleAdjustmentFixedKeyFrameIndex;

    std::vector<KeyFrame*>   orderedConnectedKeyFrames;
    std::vector<i32>         orderedWeights;
    std::map<KeyFrame*, i32> connectedKeyFrameWeights;

    KeyFrame* referenceKeyFrame;

    DBoW2::BowVector     bowVector;
    DBoW2::FeatureVector featureVector;

    KeyFrameRelocalizationData relocalizationData;
};

struct Frame
{
    i32     id;
    cv::Mat cameraMat;
    i32     numberOfKeyPoints;

    r32 scaleFactor;
    r32 logScaleFactor;

    // Pose matrices
    cv::Mat cTw;
    cv::Mat wTc;
    cv::Mat worldOrigin;

    // Keypoints, descriptors and mapPoints
    cv::Mat                   descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    std::vector<cv::KeyPoint> undistortedKeyPoints;
    std::vector<MapPoint*>    mapPointMatches;
    std::vector<bool32>       mapPointIsOutlier;

    std::vector<size_t> keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    KeyFrame* referenceKeyFrame;

    DBoW2::BowVector     bowVector;
    DBoW2::FeatureVector featureVector;
};

struct GridConstraints
{
    r32 minX;
    r32 minY;
    r32 maxX;
    r32 maxY;
    r32 invGridElementWidth;
    r32 invGridElementHeight;
};

struct LocalMappingState
{
    std::list<KeyFrame*> newKeyFrames; // TODO(jan): replace with vector?
    std::list<MapPoint*> newMapPoints;
};

struct OrbSlamState
{
    OrbSlamStatus status;
    bool32        trackingWasOk;

    i32 maxFramesBetweenKeyFrames;
    i32 minFramesBetweenKeyFrames;

    // local map
    std::vector<KeyFrame*> localKeyFrames;
    std::vector<MapPoint*> localMapPoints;

    // initialization stuff
    Frame                    initialFrame;
    std::vector<cv::Point2f> previouslyMatchedKeyPoints;
    std::vector<i32>         initializationMatches;
    bool32                   initialFrameSet;
    OrbExtractionParameters  initializationOrbExtractionParameters;

    // camera stuff
    r32 fx, fy, cx, cy;
    r32 invfx, invfy;

    GridConstraints gridConstraints;

    OrbExtractionParameters orbExtractionParameters;

    std::set<KeyFrame*>               keyFrames;
    std::vector<std::list<KeyFrame*>> invertedKeyFrameFile;
    std::set<MapPoint*>               mapPoints;
    i32                               nextFrameId;
    i32                               nextKeyFrameId;
    i32                               nextMapPointId;

    KeyFrame* referenceKeyFrame;

    Frame lastFrame;

    i32 lastKeyFrameId;
    i32 lastRelocalizationFrameId;

    i32 frameCounter;

    ORBVocabulary* orbVocabulary;

    LocalMappingState localMapping;
};

static inline cv::Mat getKeyFrameRotation(const KeyFrame* keyFrame)
{
    cv::Mat result = keyFrame->cTw.rowRange(0, 3).colRange(0, 3).clone();

    return result;
}

static inline cv::Mat getKeyFrameTranslation(const KeyFrame* keyFrame)
{
    cv::Mat result = keyFrame->cTw.rowRange(0, 3).col(3).clone();

    return result;
}

static inline cv::Mat getKeyFrameCameraCenter(const KeyFrame* keyFrame)
{
    cv::Mat result = keyFrame->worldOrigin.clone();

    return result;
}

static inline void addKeyFrameToInvertedFile(KeyFrame*                          keyFrame,
                                             std::vector<std::list<KeyFrame*>>& invertedKeyFrameFile)
{
    // TODO(jan): mutex

    for (DBoW2::BowVector::const_iterator vit = keyFrame->bowVector.begin(), vend = keyFrame->bowVector.end(); vit != vend; vit++)
    {
        invertedKeyFrameFile[vit->first].push_back(keyFrame);
    }
}

// TODO(jan): move this somewhere smarter
static void initializeMapPoint(MapPoint*      mapPoint,
                               KeyFrame*      referenceKeyFrame,
                               const cv::Mat& worldPosition,
                               i32&           nextMapPointId)
{
    mapPoint->index                    = nextMapPointId++;
    mapPoint->referenceKeyFrame        = referenceKeyFrame;
    mapPoint->firstObservationFrameId  = referenceKeyFrame->frameId;
    mapPoint->visibleInKeyFrameCounter = 1;
    mapPoint->foundInKeyFrameCounter   = 1;

    worldPosition.copyTo(mapPoint->position);
    mapPoint->normalVector = cv::Mat::zeros(3, 1, CV_32F);
}

namespace WAI
{

class WAI_API ModeOrbSlam2DataOriented : public Mode
{
    public:
    ModeOrbSlam2DataOriented(SensorCamera* camera, std::string vocabularyPath);
    void notifyUpdate();
    bool getPose(cv::Mat* pose);

    std::vector<MapPoint*> getMapPoints();
    std::vector<MapPoint*> getLocalMapPoints();
    std::vector<MapPoint*> getMatchedMapPoints();
    std::vector<KeyFrame*> getKeyFrames();
    i32                    getMapPointCount() { return _state.mapPoints.size(); }

    void showKeyPointsMatched(const bool flag)
    {
        _showKeyPointsMatched = flag;
    }

    void showKeyPoints(const bool flag)
    {
        _showKeyPoints = flag;
    }

    private:
    SensorCamera* _camera;
    OrbSlamState  _state;
    cv::Mat       _pose;

    bool _showKeyPoints        = false;
    bool _showKeyPointsMatched = true;
};
}

#endif