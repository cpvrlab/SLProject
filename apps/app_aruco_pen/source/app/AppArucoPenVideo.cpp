//#############################################################################
//  File:      AppArucoPenVideo.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//#include <AppDemo.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <CVCapture.h>
#include <cv/CVTrackedAruco.h>
#include <SLGLTexture.h>
#include <cv/CVCalibrationEstimator.h>
#include <apps/app_aruco_pen/source/app/AppArucoPenSceneView.h>
#include <AppDemo.h>
#include <FtpUtils.h>
#include <GlobalTimer.h>
#include <SLProjectScene.h>
#include <Instrumentor.h>

#include <app/AppArucoPen.h>

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
//! logic that ensures that we have a valid calibration state
void ensureValidCalibration(CVCamera* ac, SLSceneView* sv)
{
    PROFILE_FUNCTION();

    // we have to make sure calibration process is stopped if someone stopps calibrating
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

        CVCamera* ac = &AppArucoPen::instance().currentCaptureProvider()->camera();

        if (AppDemo::sceneID == SID_VideoCalibrateMain)
        {
            AppArucoPen::instance().calibrator().update(ac, s, sv);
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
                AppDemo::sceneID == SID_VideoTrackChessMain)
            {
                SLfloat      fovH = ac->calibration.cameraFovHDeg();
                SLfloat      err  = ac->calibration.reprojectionError();
                stringstream ss; // info line text
                ss << "Tracking Chessboard on main camera. ";
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

        CVTracked::trackingTimesMS.set(GlobalTimer::timeMS() - trackingTimeStartMS);
        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
