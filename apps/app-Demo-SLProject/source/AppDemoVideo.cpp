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
CVCalibrationEstimator* calibrationEstimator = nullptr;

void runCalibrationEstimator(CVCalibration* ac, SLScene* s, SLSceneView* sv)
{
    AppDemoSceneView* adSv = static_cast<AppDemoSceneView*>(sv);

    if (!calibrationEstimator)
    {
        calibrationEstimator = new CVCalibrationEstimator(ac->calibrationFlags(),
                                                          CVCapture::instance()->activeCamSizeIndex,
                                                          SLApplication::calibrationEstimatorParams.mirrorH,
                                                          SLApplication::calibrationEstimatorParams.mirrorV,
                                                          ac->camType());
        //clear grab request from sceneview
        adSv->grab = false;
    }

    if (calibrationEstimator->isStreaming())
    {
        calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, adSv->grab);
        //reset grabbing switch
        adSv->grab = false;

        stringstream ss;
        ss << "Click on the screen to create a calibration photo. Created "
           << calibrationEstimator->numCapturedImgs() << " of " << calibrationEstimator->numImgsToCapture();
        s->info(ss.str());
    }
    else if (calibrationEstimator->isBusyExtracting())
    {
        //also reset grabbing, user has to click again
        adSv->grab = false;
        calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, false);
        s->info("Busy extracting corners, please wait with grabbing ...");
    }
    else if (calibrationEstimator->isCalculating())
    {
        calibrationEstimator->updateAndDecorate(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, false);
        s->info("Calculating calibration, please wait ...");
    }
    else if (calibrationEstimator->isDone())
    {
        //overwrite current calibration
        if (calibrationEstimator->calibrationSuccessful())
        {
            *ac = calibrationEstimator->getCalibration();

            std::string computerInfo      = SLApplication::getComputerInfos();
            string      mainCalibFilename = "camCalib_" + computerInfo + "_main.xml";
            string      scndCalibFilename = "camCalib_" + computerInfo + "_scnd.xml";
            ac->save(mainCalibFilename, mainCalibFilename);
            //update scene camera
            sv->camera()->fov(ac->cameraFovVDeg());
            cv::Mat scMat = ac->cameraMatUndistorted();
            sv->camera()->intrinsics((float)scMat.at<double>(0, 0),
                                     (float)scMat.at<double>(1, 1),
                                     (float)scMat.at<double>(0, 2),
                                     (float)scMat.at<double>(1, 2));
            s->info("Calibration successful.");
        }
        else
        {
            s->info("Calibration failed!");
        }

        //free estimator instance
        delete calibrationEstimator;
        calibrationEstimator = nullptr;

        if (SLApplication::sceneID == SID_VideoCalibrateMain)
            s->onLoad(s, sv, SID_VideoTrackChessMain);
        else
            s->onLoad(s, sv, SID_VideoTrackChessScnd);
    }
}
//-----------------------------------------------------------------------------
//! logic that ensures that we have a valid calibration state
void ensureValidCalibration(CVCalibration* ac, SLSceneView* sv)
{
    //we have to make sure calibration process is stopped if someone stopps calibrating
    if (calibrationEstimator)
    {
        delete calibrationEstimator;
        calibrationEstimator = nullptr;
    }

    if (ac->state() == CS_uncalibrated)
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
            *ac = CVCalibration(devW,
                                devH,
                                devF,
                                cv::Size(CVCapture::instance()->lastFrame.cols,
                                         CVCapture::instance()->lastFrame.rows),
                                SLApplication::calibrationEstimatorParams.mirrorH,
                                SLApplication::calibrationEstimatorParams.mirrorV,
                                ac->camType());
        }
        else
        {
            //make a guess using frame size and a guessed field of view
            *ac = CVCalibration(cv::Size(CVCapture::instance()->lastFrame.cols,
                                         CVCapture::instance()->lastFrame.rows),
                                60.0,
                                SLApplication::calibrationEstimatorParams.mirrorH,
                                SLApplication::calibrationEstimatorParams.mirrorV,
                                ac->camType());
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

        CVCalibration* ac = CVCapture::instance()->activeCalib;

        if (SLApplication::sceneID == SID_VideoCalibrateMain ||
            SLApplication::sceneID == SID_VideoCalibrateScnd)
        {
            runCalibrationEstimator(ac, s, sv);
        }
        else
        {
            ensureValidCalibration(ac, sv);
            //always update scene camera fov from calibration because the calibration may have
            //been adapted in adjustForSL after a change of aspect ratio
            sv->camera()->fov(ac->cameraFovVDeg());

            if (tracker && trackedNode)
            {
                bool foundPose = tracker->track(CVCapture::instance()->lastFrameGray,
                                                CVCapture::instance()->lastFrame,
                                                ac);
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
                SLfloat      fovH = ac->cameraFovHDeg();
                SLfloat      err  = ac->reprojectionError();
                stringstream ss; // info line text
                ss << "Tracking Chessboard on " << (CVCapture::instance()->videoType() == VT_MAIN ? "main " : "scnd. ") << "camera. ";
                if (ac->state() == CS_calibrated)
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
            if (ac->state() == CS_calibrated && ac->showUndistorted())
            {
                CVMat undistorted;
                ac->remap(CVCapture::instance()->lastFrame, undistorted);

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
