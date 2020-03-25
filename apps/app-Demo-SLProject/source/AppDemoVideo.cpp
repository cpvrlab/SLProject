//#############################################################################
//  File:      AppDemoVideo.cpp
//  Author:    Marcus Hudritsch
//  Date:      August 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//#include <SLApplication.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <CVCapture.h>
#include <CVTrackedAruco.h>
#include <SLGLTexture.h>
#include <CVCalibrationEstimator.h>
#include <AppDemoSceneView.h>
#include <SLApplication.h>
#include <FtpUtils.h>

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
//always update scene camera fov from calibration because the calibration may have
//been adapted in adjustForSL after a change of aspect ratio!
//Attention: The active scene view camera may be a different one that the tracking camera
//but we have to update the tracking camera only!
void updateTrackingSceneCamera(CVCamera* ac)
{
    if (trackedNode && typeid(*trackedNode) == typeid(SLCamera))
    {
        SLCamera* trackingCam = static_cast<SLCamera*>(trackedNode);
        trackingCam->fov(ac->calibration.cameraFovVDeg());
    }
}
//-----------------------------------------------------------------------------
//CVCalibrationEstimator* calibrationEstimator = nullptr;

void runCalibrationEstimator(CVCamera* ac, SLScene* s, SLSceneView* sv)
{
    AppDemoSceneView* adSv                 = static_cast<AppDemoSceneView*>(sv);
    static bool       processedCalibResult = false;
    try
    {
        if (!SLApplication::calibrationEstimator)
        {
            SLApplication::calibrationEstimator = new CVCalibrationEstimator(SLApplication::calibrationEstimatorParams,
                                                                             CVCapture::instance()->activeCamSizeIndex,
                                                                             ac->mirrorH(),
                                                                             ac->mirrorV(),
                                                                             ac->type(),
                                                                             SLApplication::getComputerInfos(),
                                                                             SLApplication::calibIniPath,
                                                                             SLApplication::externalPath);

            //clear grab request from sceneview
            adSv->grab           = false;
            processedCalibResult = false;
        }

        if (SLApplication::calibrationEstimator->isStreaming())
        {
            SLApplication::calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, adSv->grab);
            //reset grabbing switch
            adSv->grab = false;

            stringstream ss;
            ss << "Click on the screen to create a calibration photo. Created "
               << SLApplication::calibrationEstimator->numCapturedImgs() << " of " << SLApplication::calibrationEstimator->numImgsToCapture();
            s->info(ss.str());
        }
        else if (SLApplication::calibrationEstimator->isBusyExtracting())
        {
            //also reset grabbing, user has to click again
            adSv->grab = false;
            SLApplication::calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, false);
            s->info("Busy extracting corners, please wait with grabbing ...");
        }
        else if (SLApplication::calibrationEstimator->isCalculating())
        {
            SLApplication::calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, false);
            s->info("Calculating calibration, please wait ...");
        }
        else if (SLApplication::calibrationEstimator->isDone())
        {
            if (!processedCalibResult)
            {
                if (SLApplication::calibrationEstimator->calibrationSuccessful())
                {
                    processedCalibResult = true;
                    ac->calibration      = SLApplication::calibrationEstimator->getCalibration();

                    std::string computerInfo      = SLApplication::getComputerInfos();
                    string      mainCalibFilename = "camCalib_" + computerInfo + "_main.xml";
                    string      scndCalibFilename = "camCalib_" + computerInfo + "_scnd.xml";
                    std::string errorMsg;
                    if (ac->calibration.save(SLApplication::calibFilePath, mainCalibFilename))
                    {

                        if (!FtpUtils::uploadFile(SLApplication::calibFilePath,
                                                  mainCalibFilename,
                                                  SLApplication::CALIB_FTP_HOST,
                                                  SLApplication::CALIB_FTP_USER,
                                                  SLApplication::CALIB_FTP_PWD,
                                                  SLApplication::CALIB_FTP_DIR,
                                                  errorMsg))
                        {
                            Utils::log("WAIApp", errorMsg.c_str());
                        }
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
        else if (SLApplication::calibrationEstimator->isDoneCaptureAndSave())
        {
            s->info(("Capturing done!"));
        }
    }
    catch (CVCalibrationEstimatorException& e)
    {
        log("WAIApp", e.what());
        s->info("Exception during calibration! Please restart!");
    }
}

//-----------------------------------------------------------------------------
//! logic that ensures that we have a valid calibration state
void ensureValidCalibration(CVCamera* ac, SLSceneView* sv)
{
    //we have to make sure calibration process is stopped if someone stopps calibrating
    if (SLApplication::calibrationEstimator)
    {
        delete SLApplication::calibrationEstimator;
        SLApplication::calibrationEstimator = nullptr;
    }

    if (ac->calibration.state() == CS_uncalibrated)
    {
        // Try to read device lens and sensor information
        string strF = SLApplication::deviceParameter["DeviceLensFocalLength"];
        string strW = SLApplication::deviceParameter["DeviceSensorPhysicalSizeW"];
        string strH = SLApplication::deviceParameter["DeviceSensorPhysicalSizeH"];
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
                                            SLApplication::getComputerInfos());
        }
        else
        {
            //make a guess using frame size and a guessed field of view
            ac->calibration = CVCalibration(cv::Size(CVCapture::instance()->lastFrame.cols,
                                                     CVCapture::instance()->lastFrame.rows),
                                            60.0,
                                            ac->mirrorH(),
                                            ac->mirrorV(),
                                            ac->type(),
                                            SLApplication::getComputerInfos());
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
    SLScene*     s  = SLApplication::scene;
    SLSceneView* sv = s->sceneView(0);

    if (CVCapture::instance()->videoType() != VT_NONE && !CVCapture::instance()->lastFrame.empty())
    {
        SLfloat trackingTimeStartMS = SLApplication::timeMS();

        CVCamera* ac = CVCapture::instance()->activeCamera;

        if (SLApplication::sceneID == SID_VideoCalibrateMain ||
            SLApplication::sceneID == SID_VideoCalibrateScnd)
        {
            runCalibrationEstimator(ac, s, sv);
        }
        else
        {
            ensureValidCalibration(ac, sv);
            //Attention: Always update scene camera fov from calibration because the calibration may have
            //been adapted in adjustForSL after a change of aspect ratio!
            //The active scene view camera may be a different one that the tracking camera
            //but we have to update the tracking camera only!
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
            if (SLApplication::sceneID == SID_VideoCalibrateMain ||
                SLApplication::sceneID == SID_VideoCalibrateScnd ||
                SLApplication::sceneID == SID_VideoTrackChessMain ||
                SLApplication::sceneID == SID_VideoTrackChessScnd)
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
        //copy image to video texture
        if (videoTexture)
        {
            if (ac->calibration.state() == CS_calibrated && ac->showUndistorted())
            {
                CVMat undistorted;
                ac->calibration.remap(CVCapture::instance()->lastFrame, undistorted);

                //CVCapture::instance()->videoTexture()->copyVideoImage(undistorted.cols,
                videoTexture->copyVideoImage(undistorted.cols,
                                             undistorted.rows,
                                             CVCapture::instance()->format,
                                             undistorted.data,
                                             undistorted.isContinuous(),
                                             true);
            }
            else
            {
                //CVCapture::instance()->videoTexture()->copyVideoImage(CVCapture::instance()->lastFrame.cols,
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

        CVTracked::trackingTimesMS.set(SLApplication::timeMS() - trackingTimeStartMS);
        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
