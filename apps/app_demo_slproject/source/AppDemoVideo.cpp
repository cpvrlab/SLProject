//#############################################################################
//  File:      AppDemoVideo.cpp
//  Date:      August 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLScene.h>
#include <SLSceneView.h>
#include <CVCapture.h>
#include <CVTracked.h>
#include <CVTrackedAruco.h>
#include <SLGLTexture.h>
#include <CVCalibrationEstimator.h>
#include <AppDemoSceneView.h>
#include <AppDemo.h>
#include <GlobalTimer.h>
#include <Profiler.h>

#ifndef SL_EMSCRIPTEN
#    include <FtpUtils.h>
#endif

//-----------------------------------------------------------------------------
/*! Global pointer for the video texture defined in AppDemoLoad for video scenes
 It gets updated in the following onUpdateTracking routine */
SLGLTexture* videoTexture = nullptr;

/*! Global pointer for a tracker that is set in AppDemoLoad for video scenes
 It gets updated in the following onUpdateTracking routine */
CVTracked* tracker = nullptr;

/*! Global pointer to a node that from witch the tracker changes the pose.
 it gets updated in the following onUpdateTracking routine */
SLNode* trackedNode = nullptr;

//-----------------------------------------------------------------------------
// always update scene camera fovV from calibration because the calibration may have
// been adapted in adjustForSL after a change of aspect ratio!
// Attention: The active scene view camera may be a different one that the tracking camera
// but we have to update the tracking camera only!
void updateTrackingSceneCamera(CVCamera* ac)
{
    PROFILE_FUNCTION();

    if (trackedNode && typeid(*trackedNode) == typeid(SLCamera))
    {
        SLCamera* trackingCam = dynamic_cast<SLCamera*>(trackedNode);
        trackingCam->fov(ac->calibration.cameraFovVDeg());
    }
}
//-----------------------------------------------------------------------------
// CVCalibrationEstimator* calibrationEstimator = nullptr;
void runCalibrationEstimator(CVCamera* ac, SLScene* s, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    AppDemoSceneView* adSv                 = dynamic_cast<AppDemoSceneView*>(sv);
    static bool       processedCalibResult = false;
    try
    {
        if (!AppDemo::calibrationEstimator)
        {
            AppDemo::calibrationEstimator = new CVCalibrationEstimator(AppDemo::calibrationEstimatorParams,
                                                                       CVCapture::instance()->activeCamSizeIndex,
                                                                       ac->mirrorH(),
                                                                       ac->mirrorV(),
                                                                       ac->type(),
                                                                       Utils::ComputerInfos::get(),
                                                                       AppDemo::calibIniPath,
                                                                       AppDemo::externalPath,
                                                                       AppDemo::exePath);

            // clear grab request from sceneview
            adSv->grab           = false;
            processedCalibResult = false;
        }

        if (AppDemo::calibrationEstimator->isStreaming())
        {
            AppDemo::calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame,
                                                             CVCapture::instance()->lastFrameGray,
                                                             adSv->grab);
            // reset grabbing switch
            adSv->grab = false;

            stringstream ss;
            ss << "Click on the screen to create a calibration photo. Created "
               << AppDemo::calibrationEstimator->numCapturedImgs()
               << " of " << AppDemo::calibrationEstimator->numImgsToCapture();
            s->info(ss.str());
        }
        else if (AppDemo::calibrationEstimator->isBusyExtracting())
        {
            // also reset grabbing, user has to click again
            adSv->grab = false;
            AppDemo::calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame,
                                                             CVCapture::instance()->lastFrameGray,
                                                             false);
            s->info("Busy extracting corners, please wait with grabbing ...");
        }
        else if (AppDemo::calibrationEstimator->isCalculating())
        {
            AppDemo::calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame,
                                                             CVCapture::instance()->lastFrameGray,
                                                             false);
            s->info("Calculating calibration, please wait ...");
        }
        else if (AppDemo::calibrationEstimator->isDone())
        {
            if (!processedCalibResult)
            {
                if (AppDemo::calibrationEstimator->calibrationSuccessful())
                {
                    processedCalibResult = true;
                    ac->calibration      = AppDemo::calibrationEstimator->getCalibration();

                    std::string computerInfo      = Utils::ComputerInfos::get();
                    string      mainCalibFilename = "camCalib_" + computerInfo + "_main.xml";
                    string      scndCalibFilename = "camCalib_" + computerInfo + "_scnd.xml";
                    std::string errorMsg;
                    if (ac->calibration.save(AppDemo::calibFilePath, mainCalibFilename))
                    {

#ifndef SL_EMSCRIPTEN
                        if (!FtpUtils::uploadFile(AppDemo::calibFilePath,
                                                  mainCalibFilename,
                                                  AppDemo::CALIB_FTP_HOST,
                                                  AppDemo::CALIB_FTP_USER,
                                                  AppDemo::CALIB_FTP_PWD,
                                                  AppDemo::CALIB_FTP_DIR,
                                                  errorMsg))
                        {
                            Utils::log("WAIApp", errorMsg.c_str());
                        }
#endif
                    }
                    else
                    {
                        errorMsg += " Saving calibration failed!";
                    }

                    s->info("Calibration successful." + errorMsg);
                }
                else
                {
                    s->info(("Calibration failed!"));
                }
            }
        }
        else if (AppDemo::calibrationEstimator->isDoneCaptureAndSave())
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
//! logic that ensures that we have a valid calibration state
void ensureValidCalibration(CVCamera* ac, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    // we have to make sure calibration process is stopped if someone stops calibrating
    if (AppDemo::calibrationEstimator)
    {
        delete AppDemo::calibrationEstimator;
        AppDemo::calibrationEstimator = nullptr;
    }

    if (ac->calibration.state() == CS_uncalibrated)
    {
        // Try to read device lens and sensor information
        string strF = AppDemo::deviceParameter["DeviceLensFocalLength"];
        string strW = AppDemo::deviceParameter["DeviceSensorPhysicalSizeW"];
        string strH = AppDemo::deviceParameter["DeviceSensorPhysicalSizeH"];
        if (!strF.empty() && !strW.empty() && !strH.empty())
        {
            float devF = strF.empty() ? 0.0f : stof(strF);
            float devW = strW.empty() ? 0.0f : stof(strW);
            float devH = strH.empty() ? 0.0f : stof(strH);

            // Changes the state to CS_guessed
            ac->calibration = CVCalibration(devW,
                                            devH,
                                            devF,
                                            cv::Size(CVCapture::instance()->lastFrame.cols,
                                                     CVCapture::instance()->lastFrame.rows),
                                            ac->mirrorH(),
                                            ac->mirrorV(),
                                            ac->type(),
                                            Utils::ComputerInfos::get());
        }
        else
        {
            // make a guess using frame size and a guessed field of view
            ac->calibration = CVCalibration(cv::Size(CVCapture::instance()->lastFrame.cols,
                                                     CVCapture::instance()->lastFrame.rows),
                                            60.0,
                                            ac->mirrorH(),
                                            ac->mirrorV(),
                                            ac->type(),
                                            Utils::ComputerInfos::get());
        }
    }
}
//-----------------------------------------------------------------------------
//! Implements the update per frame for video update and feature tracking
/*! This routine is called once per frame before any other update within the
 the main rendering loop (see: AppDemoMainGLFW::onPaint or GLES3View::onDrawFrame).
 See the documentation within SLCVTracked and in all of its inheritants.
*/
bool onUpdateVideo()
{
    PROFILE_FUNCTION();

    if (AppDemo::sceneViews.empty())
        return false;

    SLScene*     s  = AppDemo::scene;
    SLSceneView* sv = AppDemo::sceneViews[0];

    if (CVCapture::instance()->videoType() != VT_NONE &&
        !CVCapture::instance()->lastFrame.empty())
    {
        SLfloat trackingTimeStartMS = GlobalTimer::timeMS();

        CVCamera* ac = CVCapture::instance()->activeCamera;

        if (AppDemo::sceneID == SID_VideoCalibrateMain ||
            AppDemo::sceneID == SID_VideoCalibrateScnd)
        {
            runCalibrationEstimator(ac, s, sv);
        }
        else
        {
            ensureValidCalibration(ac, sv);
            // Attention: Always update scene camera fovV from calibration because the calibration may have
            // been adapted in adjustForSL after a change of aspect ratio!
            // The active scene view camera may be a different one that the tracking camera
            // but we have to update the tracking camera only!
            updateTrackingSceneCamera(ac);

            if (tracker && trackedNode)
            {
                bool foundPose = tracker->track(CVCapture::instance()->lastFrameGray,
                                                CVCapture::instance()->lastFrame,
                                                &ac->calibration);
                if (foundPose)
                {
                    // clang-format off
                    // convert matrix type CVMatx44f to SLMat4f
                    CVMatx44f cvOVM = tracker->objectViewMat();
                    SLMat4f glOVM(cvOVM.val[0], cvOVM.val[1], cvOVM.val[2], cvOVM.val[3],
                                  cvOVM.val[4], cvOVM.val[5], cvOVM.val[6], cvOVM.val[7],
                                  cvOVM.val[8], cvOVM.val[9], cvOVM.val[10],cvOVM.val[11],
                                  cvOVM.val[12],cvOVM.val[13],cvOVM.val[14],cvOVM.val[15]);
                    // clang-format on

                    // set the object matrix depending if the
                    // tracked node is attached to a camera or not
                    if (typeid(*trackedNode) == typeid(SLCamera))
                    {
                        trackedNode->om(glOVM.inverted());
                        trackedNode->setDrawBitsRec(SL_DB_HIDDEN, true);
                    }
                    else
                    {
                        // see comments in CVTracked::calcObjectMatrix
                        trackedNode->om(sv->camera()->om() * glOVM);
                        trackedNode->setDrawBitsRec(SL_DB_HIDDEN, false);
                    }
                }
                else
                    trackedNode->setDrawBitsRec(SL_DB_HIDDEN, false);
            }

            // Update info text only for chessboard scene
            if (AppDemo::sceneID == SID_VideoCalibrateMain ||
                AppDemo::sceneID == SID_VideoCalibrateScnd ||
                AppDemo::sceneID == SID_VideoTrackChessMain ||
                AppDemo::sceneID == SID_VideoTrackChessScnd)
            {
                SLfloat      fovH = ac->calibration.cameraFovHDeg();
                SLfloat      err  = ac->calibration.reprojectionError();
                stringstream ss; // info line text
                ss << "Tracking Chessboard on " << (CVCapture::instance()->videoType() == VT_MAIN ? "main " : "scnd. ") << "camera. ";
                if (ac->calibration.state() == CS_calibrated)
                    ss << "FOVH: " << fovH << ", error: " << err;
                else
                    ss << "Not calibrated. FOVH guessed: " << fovH << " degrees.";
                s->info(ss.str());
            }
        }

        //...................................................................
        // copy image to video texture
        if (videoTexture)
        {
            if (ac->calibration.state() == CS_calibrated && ac->showUndistorted())
            {
                CVMat undistorted;
                ac->calibration.remap(CVCapture::instance()->lastFrame, undistorted);

                // CVCapture::instance()->videoTexture()->copyVideoImage(undistorted.cols,
                videoTexture->copyVideoImage(undistorted.cols,
                                             undistorted.rows,
                                             CVCapture::instance()->format,
                                             undistorted.data,
                                             undistorted.isContinuous(),
                                             true);
            }
            else
            {
                // CVCapture::instance()->videoTexture()->copyVideoImage(CVCapture::instance()->lastFrame.cols,
                videoTexture->copyVideoImage(CVCapture::instance()->lastFrame.cols,
                                             CVCapture::instance()->lastFrame.rows,
                                             CVCapture::instance()->format,
                                             CVCapture::instance()->lastFrame.data,
                                             CVCapture::instance()->lastFrame.isContinuous(),
                                             true);
            }
        }
        else
            SL_WARN_MSG("No video texture to copy to.");

#ifndef SL_EMSCRIPTEN
        CVTracked::trackingTimesMS.set(GlobalTimer::timeMS() - trackingTimeStartMS);
#endif

        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
