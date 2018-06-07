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

SLCVStateEstimator::SLCVStateEstimator(PredictionModel predictionModel) :
_predictionModel(predictionModel)
{
    switch (predictionModel)
    {
        case PredictionModel_Kalman:
            _mX = cv::Mat::zeros(6, 1, CV_32F);
            _mA = cv::Mat::zeros(6, 6, CV_32F);
            _mU = cv::Mat::zeros(3, 1, CV_32F);
            _mB = cv::Mat::zeros(6, 3, CV_32F);
            _mP = cv::Mat::zeros(6, 6, CV_32F);
            _mW = cv::Mat::zeros(6, 1, CV_32F);
            _mQ = cv::Mat::zeros(6, 6, CV_32F);
            _mZ = cv::Mat::zeros(6, 1, CV_32F);
            _mR = cv::Mat::zeros(6, 6, CV_32F);
            _mH = cv::Mat::eye(6, 6, CV_32F);
            _mC = cv::Mat::eye(6, 6, CV_32F);
            _mI = cv::Mat::eye(6, 6, CV_32F);

            for (int i = 0; i < 6; i++)
            {
                _mR.at<SLfloat>(i, i) = 1.0f;
            }

            _mH.at<SLfloat>(3, 3) = 0.0f;
            _mH.at<SLfloat>(4, 4) = 0.0f;
            _mH.at<SLfloat>(5, 5) = 0.0f;

            _mP.at<SLfloat>(0, 0) = 1000.0f;
            _mP.at<SLfloat>(1, 1) = 1000.0f;
            _mP.at<SLfloat>(2, 2) = 1000.0f;
            _mP.at<SLfloat>(3, 3) = 1000.0f;
            _mP.at<SLfloat>(4, 4) = 1000.0f;
            _mP.at<SLfloat>(5, 5) = 1000.0f;
            break;
    }
}

void SLCVStateEstimator::kalmanUpdateNextStateFunction(float dt)
{
    _mA.at<SLfloat>(0, 0) = 1.0f;
    _mA.at<SLfloat>(0, 3) = dt;
    _mA.at<SLfloat>(1, 1) = 1.0f;
    _mA.at<SLfloat>(1, 4) = dt;
    _mA.at<SLfloat>(2, 2) = 1.0f;
    _mA.at<SLfloat>(2, 5) = dt;
    _mA.at<SLfloat>(3, 3) = 1.0f;
    _mA.at<SLfloat>(4, 4) = 1.0f;
    _mA.at<SLfloat>(5, 5) = 1.0f;

    float positionAccelerationDelta = 0.5f * (dt*dt);
    _mB.at<SLfloat>(0, 0) = positionAccelerationDelta;
    _mB.at<SLfloat>(1, 1) = positionAccelerationDelta;
    _mB.at<SLfloat>(2, 2) = positionAccelerationDelta;
    _mB.at<SLfloat>(3, 0) = dt;
    _mB.at<SLfloat>(4, 1) = dt;
    _mB.at<SLfloat>(5, 2) = dt;
}

void SLCVStateEstimator::kalmanPredict()
{
#if _ANDROID
    sensorQueue.acceleration(_mU.at<SLfloat>(0, 0), _mU.at<SLfloat>(1, 0), _mU.at<SLfloat>(2, 0));
#endif

    // Step 1: predict state
    _mX = _mA * _mX + _mB * _mU + _mW;

    // Step 2: predict covariance
    _mP = _mA * _mP * _mA.t() + _mQ;
}

void SLCVStateEstimator::kalmanUpdate(StateAndTime state)
{
    // Step 4: calculate kalman gain
    cv::Mat K = (_mP * _mH.t()) / (_mH * _mP * _mH.t() + _mR);

    // Step 5: prepare measurement
    SLVec3f sVec = state.state.translation();
    cv::Mat measurement = cv::Mat::zeros(6, 1, CV_32F);
    measurement.at<SLfloat>(0, 0) = sVec.x;
    measurement.at<SLfloat>(1, 0) = sVec.y;
    measurement.at<SLfloat>(2, 0) = sVec.z;
    cv::Mat Y = cv::Mat::zeros(6, 1, CV_32F);
    Y = _mC * measurement + _mZ;

    // Step 6: calculate new state
    _mX = _mX + K * (Y - _mH * _mX);

    // Step 7: calculate new covariance
    _mP = (_mI - K * _mH) * _mP;
}

SLMat4f SLCVStateEstimator::getPose()
{
    SLMat4f result;
  
    StateAndTime state, previousState;
    bool stateUpdated;
  
    {
        std::lock_guard<std::mutex> guard(_poseLock);
        state = _state;
        stateUpdated = _stateUpdated;
    }

    _dT = duration_cast<seconds>(SLClock::now() - state.time).count();
    SLVec3f rVec;

    switch (_predictionModel) {
        case PredictionModel_Kalman:
            kalmanUpdateNextStateFunction(_dT);

            if (_initialUpdate)
            {
                kalmanPredict();
            }

            if (stateUpdated)
            {
                kalmanUpdate(state);
                _initialUpdate = true;

                {
                    std::lock_guard<std::mutex> guard(_poseLock);
                    _stateUpdated = false;
                }
            }

            result.translate(SLVec3f(_mX.at<SLfloat>(0, 0), _mX.at<SLfloat>(1, 0), _mX.at<SLfloat>(2, 0)));
            state.state.toEulerAnglesZYX(rVec.z, rVec.y, rVec.x);
            result.fromEulerAnglesXYZ(rVec.x, rVec.y, rVec.z);
            break;

        case PredictionModel_Latest:
            result = state.state;
            break;
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

SLVec3f SLCVStateEstimator::acceleration()
{
    SLVec3f result = SLVec3f(_mU.at<SLfloat>(0, 0),
                             _mU.at<SLfloat>(1, 0),
                             _mU.at<SLfloat>(2, 0));

    return result;
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

float SLCVStateEstimator::dT()
{
    float result = _dT;
    return result;
}

SLint64 SLCVStateEstimator::dTc()
{
    SLint64 result = _dTc;
    return result;
}
