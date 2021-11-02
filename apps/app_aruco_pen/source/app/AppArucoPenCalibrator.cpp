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
void AppArucoPenCalibrator::update(CVCamera*    ac,
                                   SLScene*     s,
                                   SLSceneView* sv)
{
    auto* aapSv = dynamic_cast<AppArucoPenSceneView*>(sv);

    try
    {
        if (!_calibrationEstimator)
        {
            init(ac, aapSv);
        }

        if (_calibrationEstimator->isStreaming())
        {
            _calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, aapSv->grab);
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
            _calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, false);
            s->info("Busy extracting corners, please wait with grabbing ...");
        }
        else if (_calibrationEstimator->isCalculating())
        {
            _calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, false);
            s->info("Calculating calibration, please wait ...");
        }
        else if (_calibrationEstimator->isDone())
        {
            if (!_processedCalibResult)
            {
                if (_calibrationEstimator->calibrationSuccessful())
                {
                    _processedCalibResult = true;
                    ac->calibration       = _calibrationEstimator->getCalibration();

                    calcExtrinsicParams(AppArucoPen::instance().currentCaptureProvider());

                    std::string camUID            = AppArucoPen::instance().currentCaptureProvider()->uid();
                    string      mainCalibFilename = "camCalib_" + camUID + ".xml";
                    std::string errorMsg;

                    if (!ac->calibration.save(AppDemo::calibFilePath, mainCalibFilename))
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
void AppArucoPenCalibrator::calcExtrinsicParams(CVCaptureProvider* provider)
{
    SL_LOG("Calculating extrinsic parameters...");

    CVSize     boardSize(8, 5);
    CVVPoint2f corners2D;
    int        flags = cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_ADAPTIVE_THRESH;

    if (!cv::findChessboardCorners(provider->lastFrameGray(),
                                   boardSize,
                                   corners2D,
                                   flags))
    {
        SL_LOG("ERROR: Failed to calculate extrinsic parameters: Chessboard not detected");
        return;
    }

    cv::cornerSubPix(provider->lastFrameGray(),
                     corners2D,
                     cv::Size(11, 11),
                     cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                      30,
                                      0.0001));

    CVVVec3f boardPoints3D;
    float    squareSize = 0.029f;

    for (int y = boardSize.height - 1; y >= 0; --y)
        for (int x = 0; x < boardSize.width; ++x)
            boardPoints3D.push_back(CVPoint3f((float)x * squareSize, (float)y * squareSize, 0));

    CVMat rVec;
    CVMat tVec;
    bool  solved;

    solved = cv::solvePnP(CVMat(boardPoints3D),
                          CVMat(corners2D),
                          provider->camera().calibration.cameraMat(),
                          provider->camera().calibration.distortion(),
                          rVec,
                          tVec,
                          false,
                          cv::SOLVEPNP_ITERATIVE);

    if (!solved)
    {
        SL_LOG("ERROR: Failed to calculate extrinsic parameters: Couldn't solve PnP");
        return;
    }

    provider->camera().calibration.rvec = rVec;
    provider->camera().calibration.tvec = tVec;

    SL_LOG("Extrinsic parameters calculated!");
}
//-----------------------------------------------------------------------------