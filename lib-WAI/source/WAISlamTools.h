#ifndef WAISLAMTOOLS_H
#define WAISLAMTOOLS_H
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
    static void drawKeyPointInfo(WAIFrame& frame, cv::Mat& image, float scale);
    static void drawKeyPointMatches(WAIFrame& frame, cv::Mat& image, float scale);
    static void drawInitInfo(InitializerData& iniData, WAIFrame& frame, cv::Mat& imageRGB, float scale);

    static bool initialize(InitializerData&  iniData,
                           WAIFrame&         frame,
                           WAIOrbVocabulary* voc,
                           LocalMap&         localMap,
                           int               mapPointsNeeded,
                           unsigned long&    lastKeyFrameFrameId);

    static bool genInitialMap(WAIMap*       globalMap,
                              LocalMapping* localMapper,
                              LoopClosing*  loopCloser,
                              LocalMap&     localMap,
                              bool          serial);

    static bool oldInitialize(WAIFrame&         frame,
                              InitializerData&  iniData,
                              WAIMap*           map,
                              LocalMap&         localMap,
                              LocalMapping*     localMapper,
                              LoopClosing*      loopCloser,
                              WAIOrbVocabulary* voc,
                              int               mapPointsNeeded,
                              unsigned long&    lastKeyFrameFrameId);

    static int findFrameFixedMapMatches(WAIFrame&                 frame,
                                        WAIMap*                   waiMap,
                                        std::vector<cv::Point2f>& points2d,
                                        std::vector<cv::Point3f>& points3d);

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

    static WAIFrame createMarkerFrame(std::string       markerFile,
                                      KPextractor*      markerExtractor,
                                      const cv::Mat&    markerCameraIntrinsic,
                                      WAIOrbVocabulary* voc);

    static bool findMarkerHomography(WAIFrame&    markerFrame,
                                     WAIKeyFrame* kfCand,
                                     cv::Mat&     homography,
                                     int          minMatches);

    static bool doMarkerMapPreprocessing(std::string       markerFile,
                                         cv::Mat&          nodeTransform,
                                         float             markerWidthInM,
                                         KPextractor*      markerExtractor,
                                         WAIMap*           map,
                                         const cv::Mat&    markerCameraIntrinsic,
                                         WAIOrbVocabulary* voc);

protected:
    WAISlamTools(){};

    cv::Mat         _distortion;
    cv::Mat         _cameraIntrinsic;
    cv::Mat         _cameraExtrinsic;
    InitializerData _iniData;
    WAIFrame        _lastFrame;

    std::unique_ptr<WAIMap> _globalMap;
    LocalMap                _localMap;
    WAIOrbVocabulary*       _voc;
    cv::Mat                 _velocity;
    bool                    _initialized;

    LocalMapping*             _localMapping;
    LoopClosing*              _loopClosing;
    std::thread*              _processNewKeyFrameThread = nullptr;
    std::vector<std::thread*> _mappingThreads;
    std::thread*              _loopClosingThread = nullptr;
};

#endif
