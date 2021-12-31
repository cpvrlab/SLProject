//#############################################################################
//  File:      AppArucoPenVideo.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLScene.h>
#include <SLSceneView.h>
#include <CVCapture.h>
#include <cv/CVTrackedAruco.h>
#include <AppDemo.h>
#include <FtpUtils.h>
#include <GlobalTimer.h>
#include <Instrumentor.h>

#include <app/AppArucoPen.h>
#include <ArucoPen.h>

//-----------------------------------------------------------------------------
void updateTrackingSceneCamera(CVCamera* ac)
{
    PROFILE_FUNCTION();

    SLNode* trackedNode = AppArucoPen::instance().trackedNode;
    if (trackedNode && typeid(*trackedNode) == typeid(SLCamera))
    {
        auto* trackingCam = dynamic_cast<SLCamera*>(trackedNode);
        trackingCam->fov(ac->calibration.cameraFovVDeg());
    }
}
//-----------------------------------------------------------------------------
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
void trackVideo(CVCaptureProvider* provider)
{
    PROFILE_FUNCTION();

    SLSceneView* sv = AppDemo::sceneViews[0];
    CVCamera*    ac = &provider->camera();

    ensureValidCalibration(ac, sv);
    updateTrackingSceneCamera(ac);

    bool trackingResult = AppArucoPen::instance().arucoPen().trackingSystem()->track(provider);
    if (!trackingResult)
    {
        AppArucoPen::instance().trackedNode->setDrawBitsRec(SL_DB_HIDDEN, false);
        return;
    }

    if (AppDemo::sceneID == SID_VirtualArucoPen)
    {
        // clang-format off
        CVMatx44f cvWM = AppArucoPen::instance().arucoPen().trackingSystem()->worldMatrix();
        SLMat4f glWM(cvWM.val[0], cvWM.val[1], cvWM.val[2], cvWM.val[3],
                     cvWM.val[4], cvWM.val[5], cvWM.val[6], cvWM.val[7],
                     cvWM.val[8], cvWM.val[9], cvWM.val[10],cvWM.val[11],
                     cvWM.val[12],cvWM.val[13],cvWM.val[14],cvWM.val[15]);
        // clang-format on

        SLNode* trackedNode = AppArucoPen::instance().trackedNode;
        trackedNode->om(glWM);
        trackedNode->setDrawBitsRec(SL_DB_HIDDEN, false);
    }
    else
    {
        TrackingSystem* trackingSystem = AppArucoPen::instance().arucoPen().trackingSystem();
        if (typeid(*trackingSystem) != typeid(TrackingSystemArucoCube)) return;

        // clang-format off
        // convert matrix type CVMatx44f to SLMat4f
        CVTracked* tracker = AppArucoPen::instance().trackers().at(provider);
        CVMatx44f cvOVM = tracker->objectViewMat();
        SLMat4f glOVM(cvOVM.val[0], cvOVM.val[1], cvOVM.val[2], cvOVM.val[3],
                      cvOVM.val[4], cvOVM.val[5], cvOVM.val[6], cvOVM.val[7],
                      cvOVM.val[8], cvOVM.val[9], cvOVM.val[10],cvOVM.val[11],
                      cvOVM.val[12],cvOVM.val[13],cvOVM.val[14],cvOVM.val[15]);
        // clang-format on

        SLNode* trackedNode = AppArucoPen::instance().trackedNode;

        if (typeid(*trackedNode) == typeid(SLCamera))
        {
            trackedNode->om(glOVM.inverted());
            trackedNode->setDrawBitsRec(SL_DB_HIDDEN, true);
        }
        else
        {
            CVCalibration* calib = &ac->calibration;
            if (!calib->rvec.empty() && !calib->tvec.empty())
            {
                CVMatx44f extrinsic = CVTracked::createGLMatrix(calib->tvec, calib->rvec);
                // clang-format off
                extrinsic = CVMatx44f(extrinsic.val[ 1], extrinsic.val[ 2], extrinsic.val[ 0], extrinsic.val[3],
                                      extrinsic.val[ 5], extrinsic.val[ 6], extrinsic.val[ 4], extrinsic.val[7],
                                      extrinsic.val[ 9], extrinsic.val[10], extrinsic.val[ 8], extrinsic.val[11],
                                      0.0f,              0.0f,             0.0f,               1.0f);
                SLMat4f glExtrinsic(extrinsic.val[0], extrinsic.val[1], extrinsic.val[2], extrinsic.val[3],
                                    extrinsic.val[4], extrinsic.val[5], extrinsic.val[6], extrinsic.val[7],
                                    extrinsic.val[8], extrinsic.val[9], extrinsic.val[10],extrinsic.val[11],
                                    extrinsic.val[12],extrinsic.val[13],extrinsic.val[14],extrinsic.val[15]);
                // clang-format on
                sv->camera()->om(glExtrinsic.inverted());
            }

            // see comments in CVTracked::calcObjectMatrix
            trackedNode->om(sv->camera()->om() * glOVM);
            trackedNode->setDrawBitsRec(SL_DB_HIDDEN, false);
        }
    }
}
//-----------------------------------------------------------------------------
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
        if (AppDemo::sceneID != SID_VirtualArucoPen)
        {
            if (AppArucoPen::instance().videoTexture)
            {
                if (ac->calibration.state() == CS_calibrated && ac->showUndistorted())
                {
                    CVMat undistorted;
                    ac->calibration.remap(CVCapture::instance()->lastFrame, undistorted);

                    // CVCapture::instance()->videoTexture()->copyVideoImage(undistorted.cols,
                    AppArucoPen::instance().videoTexture->copyVideoImage(undistorted.cols,
                                                                         undistorted.rows,
                                                                         CVCapture::instance()->format,
                                                                         undistorted.data,
                                                                         undistorted.isContinuous(),
                                                                         true);
                }
                else
                {
                    AppArucoPen::instance().videoTexture->copyVideoImage(CVCapture::instance()->lastFrame.cols,
                                                                         CVCapture::instance()->lastFrame.rows,
                                                                         CVCapture::instance()->format,
                                                                         CVCapture::instance()->lastFrame.data,
                                                                         CVCapture::instance()->lastFrame.isContinuous(),
                                                                         true);
                }
            }
            else
                SL_WARN_MSG("No video texture to copy to.");
        }

        CVTracked::trackingTimesMS.set(GlobalTimer::timeMS() - trackingTimeStartMS);
        return true;
    }

    return false;
}
//-----------------------------------------------------------------