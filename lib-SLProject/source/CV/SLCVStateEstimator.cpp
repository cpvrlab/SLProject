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

SLCVStateEstimator::SLCVStateEstimator()
{
    _x = cv::Mat::zeros(6, 1, CV_32F);
    _uncertainty = cv::Mat::zeros(6, 6, CV_32F);
    _nextState = cv::Mat::zeros(6, 6, CV_32F);
    _measurement = cv::Mat::zeros(6, 6, CV_32F);
    _measurementUncertainty = cv::Mat::zeros(6, 6, CV_32F);
    _identity = cv::Mat::eye(6, 6, CV_32F);

    // initial uncertainty
    for (int i = 0; i < 6; i++)
    {
        _uncertainty.at<SLfloat>(i, i) = 1000.0f;
        _measurementUncertainty.at<SLfloat>(i, i) = 1.0f;
    }

    // next state function
    _nextState.at<SLfloat>(0, 0) = 1.0f;
    _nextState.at<SLfloat>(0, 3) = 1.0f; // dT
    _nextState.at<SLfloat>(1, 1) = 1.0f;
    _nextState.at<SLfloat>(1, 4) = 1.0f; // dT
    _nextState.at<SLfloat>(2, 2) = 1.0f;
    _nextState.at<SLfloat>(2, 5) = 1.0f; // dT
    _nextState.at<SLfloat>(3, 3) = 1.0f;
    _nextState.at<SLfloat>(4, 4) = 1.0f;
    _nextState.at<SLfloat>(5, 5) = 1.0f;

    // measurement function. positions are observed, velocities derived
    _measurement.at<SLfloat>(0, 0) = 1.0f;
    _measurement.at<SLfloat>(1, 1) = 1.0f;
    _measurement.at<SLfloat>(2, 2) = 1.0f;
}

void SLCVStateEstimator::predict()
{
    _x = _nextState * _x;
    _uncertainty = _nextState * _uncertainty * _nextState.t();
}

void SLCVStateEstimator::update()
{
    SLVec3f zVec = _state.state.translation();
    cv::Mat z = cv::Mat::zeros(3, 1, CV_32F);
    z.at<SLfloat>(0, 0) = zVec.x;
    z.at<SLfloat>(1, 0) = zVec.y;
    z.at<SLfloat>(2, 0) = zVec.z;
    cv::Mat y = z - (_measurement * _x);
    cv::Mat s = _measurement * _uncertainty * _measurement.t() + _measurementUncertainty;
    cv::Mat k = _uncertainty * _measurement.t() * s.inv();
    _x = _x + (k * y);
    _uncertainty = (_identity - (k * _measurement)) * _uncertainty;
}

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
        update();
        _dT = duration_cast<milliseconds>(state.time-previousState.time).count();
        
        _deltaIndex = (_deltaIndex + 1) % STATE_ESTIMATOR_MAX_STATE_COUNT;
        DeltaToPrevious* delta = &_deltas[_deltaIndex];
        if (_deltaCount >= STATE_ESTIMATOR_MAX_STATE_COUNT)
        {
            _summedTranslationDelta -= delta->translation;
            _summedRotationDelta = _summedRotationDelta - delta->rotation;
        }
        else
        {
            _deltaCount++;
        }

        if (_dT > 0)
        {
            SLVec3f t1 = previousState.state.translation();
            SLVec3f t2 = state.state.translation();
            SLVec3f r1, r2;
            previousState.state.toEulerAnglesZYX(r1.z, r1.y, r1.x);
            state.state.toEulerAnglesZYX(r2.z, r2.y, r2.x);

            delta->translation = (t2 - t1) / _dT;
            delta->rotation = (r2 - r1) / _dT;
  
            _summedTranslationDelta += delta->translation;
            _summedRotationDelta += delta->rotation;
        }
        else
        {
            delta->translation = SLVec3f(0.0f, 0.0f, 0.0f);
            delta->rotation = SLVec3f(0.0f, 0.0f, 0.0f);
        }

        {
            std::lock_guard<std::mutex> guard(_poseLock);
            _stateUpdated = false;
        }
    }

    if (_deltaCount > 0)
    {
        SLVec3f dP = _summedTranslationDelta;
        SLVec3f dR = _summedRotationDelta;
        SLMat4f lastState = state.state;

        SLCVCapture::FrameAndTime lastFrameAndTime;
        SLCVCapture::lastFrameAsync(&lastFrameAndTime);
        _dTc = duration_cast<milliseconds>(lastFrameAndTime.time-state.time).count();
        //_dTc = duration_cast<milliseconds>(SLClock::now()-state.time).count();

        if (_dTc > 0)
        {
            SLVec3f p = lastState.translation() + dP*_dTc;
            SLVec3f rLast;
            lastState.toEulerAnglesZYX(rLast.z, rLast.y, rLast.x);
            SLVec3f r = rLast + dR*_dTc;
  
            //result.translation(p, false);

            predict();
            result.translation(_x.at<SLfloat>(0, 0), _x.at<SLfloat>(1, 1), _x.at<SLfloat>(2, 2));

            result.fromEulerAnglesXYZ(r.x, r.y, r.z);
        }
        else
        {
            result = state.state;
        }
    }
#else
    result = state.state;
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

SLVec3f SLCVStateEstimator::dP()
{
    DeltaToPrevious* delta = &_deltas[_deltaIndex];
    SLVec3f result = delta->translation;
    return result;
}

SLVec3f SLCVStateEstimator::dR()
{
    DeltaToPrevious* delta = &_deltas[_deltaIndex];
    SLVec3f result = delta->rotation;
    return result;
}

SLint64 SLCVStateEstimator::dT()
{
    SLint64 result = _dT;
    return result;
}

SLint64 SLCVStateEstimator::dTc()
{
    SLint64 result = _dTc;
    return result;
}
