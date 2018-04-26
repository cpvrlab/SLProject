//#############################################################################
//  File:      SLCVStateEstimator.cpp
//  Author:    Jan Dellsperger
//  Date:      Apr 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCVStateEstimator.h>

SLMat4f SLCVStateEstimator::getPose()
{
    SLMat4f result;
  
    StateAndTime state, previousState;
    bool stateUpdated;
  
    {
        std::lock_guard<std::mutex> guard(_poseLock);
        state = _state;
        previousState = _previousState;
        stateUpdated = _stateUpdated;
    }

    if (stateUpdated)
    {
        SLint64 dT = duration_cast<microseconds>(state.time-previousState.time).count();
        if (dT > 0)
        {
            _deltaIndex = (_deltaIndex + 1) % STATE_ESTIMATOR_MAX_STATE_COUNT;
            DeltaToPrevious* delta = &_deltas[_deltaIndex];
            SLVec3f r;
            if (_deltaCount >= STATE_ESTIMATOR_MAX_STATE_COUNT)
            {
                _summedTranslationDelta -= delta->translation;
                _summedRotationDelta -= delta->rotation;
            }
            else
            {
                _deltaCount++;
            }

            SLVec3f r1, r2;
            previousState.state.toEulerAnglesZYX(r1.z, r1.y, r1.x);
            state.state.toEulerAnglesZYX(r2.z, r2.y, r2.x);
  
            delta->translation = (state.state.translation() - previousState.state.translation()) / dT;
            delta->rotation = (r2 - r1) / dT;
  
            _summedTranslationDelta += delta->translation;
            _summedRotationDelta += delta->rotation;
            _lastStateTime = state.time;

            {
                std::lock_guard<std::mutex> guard(_poseLock);
                _stateUpdated = false;
            }
        }
    }

    if (_deltaCount > 0)
    {
        SLVec3f dP = _summedTranslationDelta;
        SLVec3f dR = _summedRotationDelta;
        SLMat4f lastState = state.state;
  
        SLint64 dTc = duration_cast<microseconds>(SLClock::now()-_lastStateTime).count();
  
        SLVec3f rLast;
        lastState.toEulerAnglesZYX(rLast.z, rLast.y, rLast.x);

        SLVec3f p = lastState.translation() + dP*dTc;
        SLVec3f r = rLast + dR*dTc;
  
        result.translation(p, false);
        result.fromEulerAnglesXYZ(r.x, r.y, r.z); 
    }
  
    return result;
}

void SLCVStateEstimator::updatePose(const SLMat4f& slMat, const SLTimePoint& time)
{
    std::lock_guard<std::mutex> guard(_poseLock);
    _previousState = _state;
    _state.state = slMat;
    _state.time = time;
    _stateUpdated = true;
}
