//#############################################################################
//  File:      SLCVMapPoint.h
//  Author:    Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVMAPPOINT_H
#define SLCVMAPPOINT_H

#include <vector>
#include <map>
#include <SLCV.h>

class SLCVKeyFrame;
class SLCVFrame;
//-----------------------------------------------------------------------------
//! 
/*! 
*/
class SLCVMapPoint
{
public:
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
    bool isBad() const { return false; } //we have no bad systematic
    cv::Mat GetNormal() { return mNormalVector.clone(); }
    SLCVKeyFrame* refKf() const { return mpRefKF; }

    void AddObservation(SLCVKeyFrame* pKF, size_t idx);
    std::map<SLCVKeyFrame*, size_t> GetObservations() const { return mObservations; }

    int GetIndexInKeyFrame(SLCVKeyFrame* pKF);

    float GetMaxDistanceInvariance() { return 1.2f*mfMaxDistance; }
    float GetMinDistanceInvariance() { return 0.8f*mfMinDistance; }
    void UpdateNormalAndDepth();
    int PredictScale(const float &currentDist, SLCVFrame* pF);
    cv::Mat GetDescriptor() { return mDescriptor.clone(); }
    void ComputeDistinctiveDescriptors();

    void IncreaseFound(int n=1) { mnFound += n; }
    void IncreaseVisible(int n=1) { mnVisible += n; }

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

    // Keyframes observing the point and associated index in keyframe
    std::map<SLCVKeyFrame*, size_t> mObservations;
    // Reference KeyFrame
    SLCVKeyFrame* mpRefKF;

private:
    int _id=-1;
    //open cv coordinate representation: z-axis points to principlal point,
    // x-axis to the right and y-axis down
    SLCVMat _worldPos;

    // Mean viewing direction
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    int _nObs=0;

    // Scale invariance distances
    float mfMinDistance = 0.f;
    float mfMaxDistance = 0.f;

    //keypoint octave (level)
    int _level = -1;

    // Tracking counters
    int mnVisible = 0;
    int mnFound = 0;
};

typedef std::vector<SLCVMapPoint> SLCVVMapPoint;

#endif // !SLCVMAPPOINT_H
