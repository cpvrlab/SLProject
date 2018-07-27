//#############################################################################
//  File:      SLCVMapPoint.h
//  Author:    Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVMAPPOINT_H
#define SLCVMAPPOINT_H

#include <SLCV.h>
#include <SLVec3.h>
#include <map>
#include <vector>
#include <opencv2/core/core.hpp>

class SLCVKeyFrame;
class SLCVFrame;
class SLCVMap;
//-----------------------------------------------------------------------------
//! 
/*! 
*/
class SLCVMapPoint
{
public:
    //!constructor used during map loading
    SLCVMapPoint(int id, const cv::Mat &Pos, SLCVMap* pMap);
    SLCVMapPoint(const cv::Mat &Pos, SLCVKeyFrame *pRefKF, SLCVMap* pMap);

    //ghm1: getters for scene object position initialization
    SLVec3f worldPosVec();
    SLVec3f normalVec();

    void worldPosVec(SLVec3f);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    SLCVKeyFrame* GetReferenceKeyFrame();

    std::map<SLCVKeyFrame*, size_t> GetObservations();
    int Observations();

    void AddObservation(SLCVKeyFrame* pKF, size_t idx);
    void EraseObservation(SLCVKeyFrame* pKF);

    int GetIndexInKeyFrame(SLCVKeyFrame* pKF);
    bool IsInKeyFrame(SLCVKeyFrame *pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(SLCVMapPoint* pMP);
    SLCVMapPoint* GetReplaced();

    void IncreaseVisible(int n = 1);
    void IncreaseFound(int n = 1);
    float GetFoundRatio();

    void ComputeDistinctiveDescriptors();
    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, SLCVKeyFrame* pF);
    int PredictScale(const float &currentDist, SLCVFrame* pF);

    //ghm1: used for map IO
    SLCVKeyFrame* refKf() const { return mpRefKF; }
    void refKf(SLCVKeyFrame* refKf) { mpRefKF = refKf; }

    size_t getSizeOfCvMat(const cv::Mat& mat);
    size_t getSizeOf();
public:
    long unsigned int mnId = -1;
    //ghm1: this keeps track of the highest used id, to never use the same id again
    static long unsigned int nNextId;
    long int mnFirstKFid;
    int nObs = 0;

    // Variables used by the tracking
    //ghm1: projection point
    float mTrackProjX = 0.0f;
    float mTrackProjY = 0.0f;

    //ghm1: flags, if the map point is in frustum of the current frame
    bool mbTrackInView = false;
    int mnTrackScaleLevel = 0;
    float mTrackViewCos = 0.0f;
    long unsigned int mnTrackReferenceForFrame = 0;
    long unsigned int mnLastFrameSeen = 0;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;

protected:
    //open cv coordinate representation: z-axis points to principlal point,
    // x-axis to the right and y-axis down
    cv::Mat mWorldPos;

    // Keyframes observing the point and associated index in keyframe
    std::map<SLCVKeyFrame*, size_t> mObservations;

    // Mean viewing direction
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Reference KeyFrame
    SLCVKeyFrame* mpRefKF=NULL;

    // Tracking counters
    int mnVisible = 0;
    int mnFound = 0;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad = false;
    SLCVMapPoint* mpReplaced;

    // Scale invariance distances
    float mfMinDistance = 0.f;
    float mfMaxDistance = 0.f;

    SLCVMap* mpMap = NULL;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

#endif // !SLCVMAPPOINT_H
