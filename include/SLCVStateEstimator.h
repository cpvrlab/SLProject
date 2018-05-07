//#############################################################################
//  File:      SLCVStateEstimator.h
//  Author:    Jan Dellsperger
//  Date:      Apr 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVSTATEESTIMATOR_H
#define SLCVSTATEESTIMATOR_H

#define STATE_ESTIMATOR_MAX_STATE_COUNT 2

#include <stdafx.h>

class SLCVStateEstimator
{
public:
    struct StateAndTime
    {
        SLMat4f state;
        SLTimePoint time;
    };

    struct DeltaToPrevious
    {
        SLVec3f translation;
        SLVec3f rotation;
    };
  
    SLMat4f getPose();
    void updatePose(const SLMat4f& slMat, const SLTimePoint& time);
    SLVec3f dP();
    SLVec3f dR();
    SLint64 dT();
    SLint64 dTc();

private:
    StateAndTime _state;
    StateAndTime _previousState;
    DeltaToPrevious _deltas[STATE_ESTIMATOR_MAX_STATE_COUNT];
    SLVec3f _summedTranslationDelta;
    SLVec3f _summedRotationDelta;
    bool _stateUpdated = false;
    int _deltaIndex = -1;
    int _deltaCount = 0;
    SLint64 _dT;
    SLint64 _dTc;
    std::mutex _poseLock;
};

#endif
