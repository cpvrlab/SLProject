//#############################################################################
//  File:      SLCVMapPoint.cpp
//  Author:    Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "stdafx.h"
#include "SLCVMapPoint.h"
#include <SLCVKeyFrame.h>

//-----------------------------------------------------------------------------
SLVec3f SLCVMapPoint::worldPos()
{ 
    SLVec3f vec;
    vec.x = _worldPos.at<float>(0,0);
    vec.y = _worldPos.at<float>(1,0);
    vec.z = _worldPos.at<float>(2,0);
    return vec;
}
//-----------------------------------------------------------------------------
void SLCVMapPoint::AddObservation(SLCVKeyFrame* pKF, size_t idx)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
    _nObs++;
}