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
//-----------------------------------------------------------------------------
//! 
/*! 
*/
class SLCVMapPoint
{
public:
    void id(int id) { _id = id; }
    void worldPos(const SLCVMat& pos) { _worldPos = pos; }
    SLVec3f worldPos();

    bool isBad() { return false; } //we have no bad systematic

    void AddObservation(SLCVKeyFrame* pKF, size_t idx);
    std::map<SLCVKeyFrame*, size_t> SLCVMapPoint::GetObservations() { return mObservations; }

private:
    int _id=-1;
    //open cv coordinate representation: z-axis points to principlal point,
    // x-axis to the right and y-axis down
    SLCVMat _worldPos;

    // Keyframes observing the point and associated index in keyframe
    std::map<SLCVKeyFrame*, size_t> mObservations;

    // Mean viewing direction
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Reference KeyFrame
    SLCVKeyFrame* mpRefKF;

    int _nObs=0;
};

typedef std::vector<SLCVMapPoint> SLCVVMapPoint;

#endif // !SLCVMAPPOINT_H
