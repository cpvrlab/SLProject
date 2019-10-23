//#############################################################################
//  File:      WAIMapPoint.h
//  Author:    Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef WAIMAPPOINT_H
#define WAIMAPPOINT_H

#include <WAIHelper.h>
#include <WAIMap.h>
#include <map>
#include <mutex>
#include <vector>

#include <opencv2/core/core.hpp>

#include <WAIMath.h>

class WAIKeyFrame;
class WAIFrame;
class WAIMap;
//-----------------------------------------------------------------------------
//!
/*! 
*/
class WAI_API WAIMapPoint
{
    public:
    //!constructor used during map loading
    WAIMapPoint(int id, const cv::Mat& Pos, WAIMap* pMap);
    WAIMapPoint(const cv::Mat& Pos, WAIKeyFrame* pRefKF, WAIMap* pMap);

    //ghm1: getters for scene object position initialization
    WAI::V3 worldPosVec();
    WAI::V3 normalVec();

    void worldPosVec(WAI::V3);

    void    SetWorldPos(const cv::Mat& Pos);
    cv::Mat GetWorldPos();

    cv::Mat      GetNormal();
    WAIKeyFrame* GetReferenceKeyFrame();

    std::map<WAIKeyFrame*, size_t> GetObservations();
    int                            Observations();

    void AddObservation(WAIKeyFrame* pKF, size_t idx);
    void EraseObservation(WAIKeyFrame* pKF);

    int  GetIndexInKeyFrame(WAIKeyFrame* pKF);
    bool IsInKeyFrame(WAIKeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void         Replace(WAIMapPoint* pMP);
    WAIMapPoint* GetReplaced();

    void  IncreaseVisible(int n = 1);
    void  IncreaseFound(int n = 1);
    float GetFoundRatio();

    void    ComputeDistinctiveDescriptors();
    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int   PredictScale(const float& currentDist, WAIKeyFrame* pF);
    int   PredictScale(const float& currentDist, WAIFrame* pF);

    //ghm1: used for map IO
    WAIKeyFrame* refKf() const { return mpRefKF; }
    void         refKf(WAIKeyFrame* refKf) { mpRefKF = refKf; }

    enum RefKfSource
    {
        RefKfSource_None             = 0,
        RefKfSource_Constructor      = 1,
        RefKfSource_EraseObservation = 2
    };

    RefKfSource refKfSource = RefKfSource_None;

    size_t getSizeOfCvMat(const cv::Mat& mat);
    size_t getSizeOf();

    public:
    long unsigned int mnId = -1;
    //ghm1: this keeps track of the highest used id, to never use the same id again
    static long unsigned int nNextId;
    long int                 mnFirstKFid;
    int                      nObs = 0;

    // Variables used by the tracking
    //ghm1: projection point
    float mTrackProjX = 0.0f;
    float mTrackProjY = 0.0f;

    //ghm1: flags, if the map point is in frustum of the current frame
    bool              mbTrackInView            = false;
    int               mnTrackScaleLevel        = 0;
    float             mTrackViewCos            = 0.0f;
    long unsigned int mnTrackReferenceForFrame = 0;
    long unsigned int mnLastFrameSeen          = 0;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat           mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;

    protected:
    //open cv coordinate representation: z-axis points to principlal point,
    // x-axis to the right and y-axis down
    cv::Mat mWorldPos;

    // Keyframes observing the point and associated index in keyframe
    std::map<WAIKeyFrame*, size_t> mObservations;

    // Mean viewing direction
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Reference KeyFrame
    WAIKeyFrame* mpRefKF = NULL;

    // Tracking counters
    int mnVisible = 0;
    int mnFound   = 0;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool         mbBad = false;
    WAIMapPoint* mpReplaced;

    // Scale invariance distances
    float mfMinDistance = 0.f;
    float mfMaxDistance = 0.f;

    WAIMap* mpMap = NULL;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

#endif // !WAIMAPPOINT_H
