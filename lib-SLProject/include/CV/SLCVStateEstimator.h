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

#define STATE_ESTIMATOR_MAX_STATE_COUNT 10

#include <stdafx.h>

#if _ANDROID
#include <AppDemoAndroidSensorQueue.h>
#endif

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

    enum PredictionModel
    {
        PredictionModel_None,
        PredictionModel_Latest,
        PredictionModel_Kalman
    };

    SLCVStateEstimator(PredictionModel predictionModel);
  
    SLMat4f getPose();
    void updatePose(const SLMat4f& slMat, const SLTimePoint& time);
    SLVec3f dP();
    SLVec3f dR();
    float dT();
    SLint64 dTc();
    SLVec3f acceleration();

private:
    PredictionModel _predictionModel;
    StateAndTime _state;
    StateAndTime _previousState;
    DeltaToPrevious _deltas[STATE_ESTIMATOR_MAX_STATE_COUNT];
    SLVec3f _summedTranslationDelta;
    SLVec3f _summedRotationDelta;
    bool _stateUpdated = false;
    bool _initialUpdate = false;
    int _deltaIndex = -1;
    int _deltaCount = 0;
    float _dT;
    SLint64 _dTc;
    std::mutex _poseLock;

    // Kalman filter
    /*
     * state Matrix _mX layout:
     * s := position
     * v := velocity
     * s.x --- --- --- --- ---
     * --- s.y --- --- --- ---
     * --- --- s.z --- --- ---
     * --- --- --- v.x --- ---
     * --- --- --- --- v.y ---
     * --- --- --- --- --- v.z
     */
    cv::Mat _mX; // state
    cv::Mat _mA; // position and velocity update
    cv::Mat _mU; // acceleration measurement
    cv::Mat _mB; // acceleration update
    cv::Mat _mP; // state covariance
    cv::Mat _mW; // state prediction error
    cv::Mat _mQ; // covariance prediction error
    cv::Mat _mZ; // measurement error
    cv::Mat _mR; // measurement uncertainty
    cv::Mat _mH; // transformation matrix
    cv::Mat _mC; // transformation matrix
    cv::Mat _mI; // identity matrix

    void kalmanUpdateNextStateFunction(float dt);
    void kalmanPredict();
    void kalmanUpdate(StateAndTime state);

#if _ANDROID
    AppDemoAndroidSensorQueue sensorQueue;
#endif
};

#endif
