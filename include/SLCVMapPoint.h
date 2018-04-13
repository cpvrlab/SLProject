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
    SLCVMapPoint() 
        : mnFirstKFid(-1)
    {}
    SLCVMapPoint(const cv::Mat &Pos, SLCVKeyFrame *pRefKF, SLCVMap* pMap);

    int id() const { return _id; }
    int Observations() { return _nObs; }
    void id(int id) { _id = id; }
    void worldPos(const SLCVMat& pos) { 
        pos.copyTo(_worldPos); 
    }
    SLCVMat worldPos() const { 
        return _worldPos.clone(); 
    }

    SLVec3f worldPosVec();
    SLVec3f normalVec();

    void refKf(SLCVKeyFrame* refKf) { mpRefKF = refKf; }
    void level(int level) { _level = level; }
    bool isBad() const { return mbBad; } //we have no bad systematic
    void SetBadFlag();
    cv::Mat GetNormal() { return mNormalVector.clone(); }
    SLCVKeyFrame* refKf() const { return mpRefKF; }

    void AddObservation(SLCVKeyFrame* pKF, size_t idx);
    void EraseObservation(SLCVKeyFrame* pKF);
    std::map<SLCVKeyFrame*, size_t> GetObservations() const { return mObservations; }
    bool IsInKeyFrame(SLCVKeyFrame *pKF);

    int GetIndexInKeyFrame(SLCVKeyFrame* pKF);

    float GetMaxDistanceInvariance() { return 1.2f*mfMaxDistance; }
    float GetMinDistanceInvariance() { return 0.8f*mfMinDistance; }
    void UpdateNormalAndDepth();
    int PredictScale(const float &currentDist, SLCVKeyFrame* pF);
    int PredictScale(const float &currentDist, SLCVFrame* pF);
    cv::Mat GetDescriptor() { return mDescriptor.clone(); }
    void ComputeDistinctiveDescriptors();

    void IncreaseFound(int n=1) { mnFound += n; }
    void IncreaseVisible(int n=1) { mnVisible += n; }
    float GetFoundRatio();

    // Variables used by the tracking
    //ghm1: projection point
    float mTrackProjX = 0.0f;
    float mTrackProjY = 0.0f;
    //float mTrackProjXR = 0.0f;
    //ghm1: flags, if the map point is in frustum of the current frame
    bool mbTrackInView = false;
    int mnTrackScaleLevel = 0;
    float mTrackViewCos = 0.0f;
    long unsigned int mnLastFrameSeen = 0;
    long unsigned int mnTrackReferenceForFrame = 0;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Keyframes observing the point and associated index in keyframe
    std::map<SLCVKeyFrame*, size_t> mObservations;
    // Reference KeyFrame
    SLCVKeyFrame* mpRefKF;

    // Variables used by bundle adjustment
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;

    long int mnFirstKFid;

    void Replace(SLCVMapPoint* pMP);

private:
    int _id=-1;
    static long unsigned int nNextId;

    //open cv coordinate representation: z-axis points to principlal point,
    // x-axis to the right and y-axis down
    SLCVMat _worldPos;

    // Mean viewing direction
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    int _nObs=0;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad=false;
    SLCVMapPoint* mpReplaced;

    // Scale invariance distances
    float mfMinDistance = 0.f;
    float mfMaxDistance = 0.f;

    //keypoint octave (level)
    int _level = -1;

    // Tracking counters
    int mnVisible = 0;
    int mnFound = 0;

    SLCVMap* mpMap = NULL;
};

typedef std::vector<SLCVMapPoint> SLCVVMapPoint;

#endif // !SLCVMAPPOINT_H
