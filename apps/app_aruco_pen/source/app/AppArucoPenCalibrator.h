//#############################################################################
//  File:      AppArucoPenCalibrator.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_APPARUCOPENCALIBRATOR_H
#define SLPROJECT_APPARUCOPENCALIBRATOR_H

#include <cv/CVCamera.h>
#include <cv/CVCalibrationEstimator.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <app/AppArucoPenSceneView.h>
#include <CVCaptureProvider.h>

//-----------------------------------------------------------------------------
class AppArucoPenCalibrator
{
private:
    CVCalibrationEstimator* _calibrationEstimator = nullptr;
    bool                    _processedCalibResult = false;

public:
    bool grab = false;

    ~AppArucoPenCalibrator();

    void reset();
    void update(CVCamera*    ac,
                SLScene*     s,
                SLSceneView* sv);
    void init(CVCamera*             ac,
              AppArucoPenSceneView* aapSv);

    static void calcExtrinsicParams(CVCaptureProvider* provider);
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_APPARUCOPENCALIBRATOR_H
