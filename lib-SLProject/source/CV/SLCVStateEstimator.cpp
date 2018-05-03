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
#include <SLCVCapture.h>

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

#if 1
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
                _summedRotationDelta = _summedRotationDelta - delta->rotation;
            }
            else
            {
                _deltaCount++;
            }

            SLVec4f rV1, rV2;
            SLVec3f t1, t2;
            SLVec3f s1, s2;
            state.state.decompose(t2, rV2, s2);
            previousState.state.decompose(t1, rV1, s1);

            //SLQuat4f r1 = SLQuat4f(rV1.x, rV1.y, rV1.z, rV1.w);
            //SLQuat4f r2 = SLQuat4f(rV2.x, rV2.y, rV2.z, rV2.w);

            //SLQuat4f dR = r2 - r1;
            SLVec3f dR = (s2 - s1) / dT;
            //dR.scale(1.0f / dT);
            
            delta->translation = (t2 - t1) / dT;
            delta->rotation = dR;
  
            _summedTranslationDelta += delta->translation;
            _summedRotationDelta = _summedRotationDelta + delta->rotation;
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
        //SLQuat4f dR = _summedRotationDelta;
        SLVec3f dR = _summedRotationDelta;
        SLMat4f lastState = state.state;

        SLCVCapture::FrameAndTime lastFrameAndTime;
        SLCVCapture::lastFrameAsync(&lastFrameAndTime);
        SLint64 dTc = duration_cast<microseconds>(lastFrameAndTime.time-_lastStateTime).count();

        SLVec3f tLast;
        SLVec4f rVLast;
        SLVec3f sLast;
        lastState.decompose(tLast, rVLast, sLast);

        //SLQuat4f rLast = SLQuat4f(rVLast.x, rVLast.y, rVLast.z, rVLast.w);
        
        SLVec3f p = lastState.translation() + dP*dTc;
        //SLQuat4f r = rLast + dR*dTc;
        SLVec3f rLast;
        lastState.toEulerAnglesZYX(rLast.z, rLast.y, rLast.x);
        SLVec3f r = rLast + dR*dTc;

        //SLVec3f rV;
        //r.toEulerAnglesXYZ(rV.x, rV.y, rV.z);
  
        result.translation(p, false);
        //result.fromEulerAnglesXYZ(rV.x, rV.y, rV.z);
        result.fromEulerAnglesXYZ(r.x, r.y, r.z);
    }
#else
    result = previousState.state;
#endif
  
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

SLVec3f SLCVStateEstimator::dT()
{
    SLVec3f result = _summedTranslationDelta;
    return result;
}

SLVec3f SLCVStateEstimator::dR()
{
    SLVec3f result;
    //_summedRotationDelta.toEulerAnglesXYZ(result.x, result.x, result.z);
    result = _summedRotationDelta;
    return result;
}
