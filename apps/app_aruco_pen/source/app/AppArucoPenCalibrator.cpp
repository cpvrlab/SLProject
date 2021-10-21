//#############################################################################
//  File:      AppArucoPenCalibrator.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "AppArucoPenCalibrator.h"

#include <AppDemo.h>
#include <app/AppArucoPen.h>

#include <CVCapture.h>

#include <string>
#include <sstream>

//-----------------------------------------------------------------------------
AppArucoPenCalibrator::~AppArucoPenCalibrator()
{
    delete _calibrationEstimator;
}
//-----------------------------------------------------------------------------
void AppArucoPenCalibrator::reset()
{
    delete _calibrationEstimator;
    _calibrationEstimator = nullptr;
}
//-----------------------------------------------------------------------------
void AppArucoPenCalibrator::update(CVCaptureProvider* provider,
                                   SLScene*           s,
                                   SLSceneView*       sv)
{
    auto* aapSv = dynamic_cast<AppArucoPenSceneView*>(sv);

    try
    {
        if (!_calibrationEstimator)
        {
            init(&provider->camera(), aapSv);
        }

        if (_calibrationEstimator->isStreaming())
        {
            _calibrationEstimator->updateAndDecorate(provider->lastFrameBGR(),
                                                     provider->lastFrameGray(),
                                                     aapSv->grab);
            // reset grabbing switch
            aapSv->grab = false;

            stringstream ss;
            ss << "Click on the screen to create a calibration photo. Created "
               << _calibrationEstimator->numCapturedImgs() << " of " << _calibrationEstimator->numImgsToCapture();
            s->info(ss.str());
        }
        else if (_calibrationEstimator->isBusyExtracting())
        {
            // also reset grabbing, user has to click again
            aapSv->grab = false;
            _calibrationEstimator->updateAndDecorate(provider->lastFrameBGR(),
                                                     provider->lastFrameGray(),
                                                     false);
            s->info("Busy extracting corners, please wait with grabbing ...");
        }
        else if (_calibrationEstimator->isCalculating())
        {
            _calibrationEstimator->updateAndDecorate(provider->lastFrameBGR(),
                                                     provider->lastFrameGray(),
                                                     false);
            s->info("Calculating calibration, please wait ...");
        }
        else if (_calibrationEstimator->isDone())
        {
            if (!_processedCalibResult)
            {
                if (_calibrationEstimator->calibrationSuccessful())
                {
                    _processedCalibResult          = true;
                    provider->camera().calibration = _calibrationEstimator->getCalibration();

                    std::string camUID            = AppArucoPen::instance().currentCaptureProvider()->uid();
                    string      mainCalibFilename = "camCalib_" + camUID + ".xml";
                    std::string errorMsg;

                    if (!provider->camera().calibration.save(AppDemo::calibFilePath, mainCalibFilename))
                    {
                        errorMsg += " Saving calibration failed!";
                    }

                    s->info("Calibration successful." + errorMsg);
                }
                else
                {
                    s->info(("Calibration failed!"));
                }

                s->onLoad(s, sv, SID_VideoTrackArucoCubeMain);
            }
        }
        else if (_calibrationEstimator->isDoneCaptureAndSave())
        {
            s->info(("Capturing done!"));
        }
    }
    catch (CVCalibrationEstimatorException& e)
    {
        log("SLProject", e.what());
        s->info("Exception during calibration! Please restart!");
    }
}
//-----------------------------------------------------------------------------
void AppArucoPenCalibrator::init(CVCamera*             ac,
                                 AppArucoPenSceneView* aapSv)
{
    _calibrationEstimator = new CVCalibrationEstimator(AppDemo::calibrationEstimatorParams,
                                                       CVCapture::instance()->activeCamSizeIndex,
                                                       ac->mirrorH(),
                                                       ac->mirrorV(),
                                                       ac->type(),
                                                       Utils::ComputerInfos::get(),
                                                       AppDemo::calibIniPath,
                                                       AppDemo::externalPath,
                                                       AppDemo::exePath);
    // clear grab request from sceneview
    aapSv->grab           = false;
    _processedCalibResult = false;
}
//-----------------------------------------------------------------------------
