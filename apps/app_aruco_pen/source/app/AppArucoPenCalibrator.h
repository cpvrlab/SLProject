#ifndef SLPROJECT_APPARUCOPENCALIBRATOR_H
#define SLPROJECT_APPARUCOPENCALIBRATOR_H

#include <cv/CVCamera.h>
#include <cv/CVCalibrationEstimator.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <app/AppArucoPenSceneView.h>

class AppArucoPenCalibrator
{
private:
    CVCalibrationEstimator* _calibrationEstimator = nullptr;
    bool _processedCalibResult = false;

public:
    bool grab = false;

    ~AppArucoPenCalibrator();

    void reset();
    void update(CVCamera* ac, SLScene* s, SLSceneView* sv);
    void init(CVCamera* ac, AppArucoPenSceneView* aapSv);

};

#endif // SLPROJECT_APPARUCOPENCALIBRATOR_H
