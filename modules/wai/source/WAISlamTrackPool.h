#ifndef WAI_SLAM_TRACKPOOL_H
#define WAI_SLAM_TRACKPOOL_H

#include <WAISlamTools.h>

class WAISlamTrackPool : public WAISlamTools
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

    WAISlamTrackPool(const cv::Mat&          intrinsic,
                     const cv::Mat&          distortion,
                     WAIOrbVocabulary*       voc,
                     KPextractor*            iniExtractor,
                     KPextractor*            relocExtractor,
                     KPextractor*            extractor,
                     std::unique_ptr<WAIMap> globalMap,
                     Params                  params);

    virtual ~WAISlamTrackPool();

    bool update(cv::Mat& imageGray);

    virtual WAITrackingState getTrackingState() { return _state; }
    void                     changeIntrinsic(cv::Mat intrinsic, cv::Mat distortion);
    cv::Mat                  getPose();

    void drawInfo(cv::Mat& imageBGR,
                  float    scale,
                  bool     showInitLine,
                  bool     showKeyPoints,
                  bool     showKeyPointsMatched);

    virtual std::vector<WAIMapPoint*> getMapPoints()
    {
        if (_globalMap != nullptr)
            return _globalMap->GetAllMapPoints();
        return std::vector<WAIMapPoint*>();
    }
    std::vector<WAIMapPoint*> getMatchedMapPoints();

private:
    void createFrame(WAIFrame& frame, cv::Mat& imageGray);
    void updatePose(WAIFrame& frame);

    Params _params;

    unsigned int  _relocFrameCounter   = 0;
    unsigned long _lastRelocFrameId    = 0;
    unsigned long _lastKeyFrameFrameId = 0;
    KPextractor*  _extractor           = nullptr;
    KPextractor*  _relocExtractor      = nullptr;
    KPextractor*  _iniExtractor        = nullptr;
    int           _infoMatchedInliners = 0;

    WAITrackingState _state;

    std::mutex _cameraExtrinsicMutex;
};
//-----------------------------------------------------------------------------
#endif
