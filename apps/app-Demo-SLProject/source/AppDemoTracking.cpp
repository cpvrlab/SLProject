//#############################################################################
//  File:      AppDemoTracking.cpp
//  Author:    Marcus Hudritsch
//  Date:      August 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLApplication.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLCVCapture.h>
#include <SLCVTrackedAruco.h>

//-----------------------------------------------------------------------------
//! Implements the update per frame for feature tracking and video update
/*! This routine is called once per frame before any other update within the
 the main rendering loop (see: AppDemoMainGLFW::onPaint or GLES3View::onDrawFrame).
 See the documentation within SLCVTracked and in all of its inheritants.
*/
bool onUpdateTracking()
{
    SLScene*     s  = SLApplication::scene;
    SLSceneView* sv = s->sv(0);

    if (SLCVCapture::instance()->videoType() != VT_NONE && !SLCVCapture::instance()->lastFrame.empty())
    {
        SLfloat          trackingTimeStartMS = SLApplication::timeMS();
        SLCVCalibration* ac                  = SLApplication::activeCalib;

        // Invalidate calibration if camera input aspect doesn't match output
        SLfloat calibWdivH              = ac->imageAspectRatio();
        SLbool  aspectRatioDoesNotMatch = SL_abs(sv->scrWdivH() - calibWdivH) > 0.01f;
        if (aspectRatioDoesNotMatch && ac->state() == CS_calibrated)
        {
            ac->clear();
        }

        stringstream ss; // info line text

        //.....................................................................
        if (ac->state() == CS_uncalibrated)
        {
            if (SLApplication::sceneID == SID_VideoCalibrateMain ||
                SLApplication::sceneID == SID_VideoCalibrateScnd)
            {
                ac->state(CS_calibrateStream);
            }
            else
            { // Changes the state to CS_guessed
                ac->createFromGuessedFOV(SLCVCapture::instance()->lastFrame.cols,
                                         SLCVCapture::instance()->lastFrame.rows);
                sv->camera()->fov(ac->cameraFovVDeg());
            }
        }
        else //..............................................................
          if (ac->state() == CS_calibrateStream || ac->state() == CS_calibrateGrab)
        {
            ac->findChessboard(SLCVCapture::instance()->lastFrame, SLCVCapture::instance()->lastFrameGray, true);
            int imgsToCap = ac->numImgsToCapture();
            int imgsCaped = ac->numCapturedImgs();

            //update info line
            if (imgsCaped < imgsToCap)
                ss << "Click on the screen to create a calibration photo. Created "
                   << imgsCaped << " of " << imgsToCap;
            else
            {
                ss << "Calculating calibration, please wait ...";
                ac->state(CS_startCalculating);
            }
            s->info(ss.str());
        }
        else //..............................................................
          if (ac->state() == CS_startCalculating)
        {
            if (ac->calculate())
            {
                sv->camera()->fov(ac->cameraFovVDeg());
                if (SLApplication::sceneID == SID_VideoCalibrateMain)
                    s->onLoad(s, sv, SID_VideoTrackChessMain);
                else
                    s->onLoad(s, sv, SID_VideoTrackChessScnd);
            }
        }
        else if (ac->state() == CS_calibrated || ac->state() == CS_guessed) //......
        {
            SLCVTrackedAruco::trackAllOnce = true;

            // track all trackers in the first sceneview
            for (auto tracker : SLCVTracked::trackers)
            {
                tracker->track(SLCVCapture::instance()->lastFrameGray,
                               SLCVCapture::instance()->lastFrame,
                               ac,
                               SLCVTracked::showDetection,
                               sv);
            }

            // Update info text only for chessboard scene
            if (SLApplication::sceneID == SID_VideoCalibrateMain ||
                SLApplication::sceneID == SID_VideoCalibrateScnd ||
                SLApplication::sceneID == SID_VideoTrackChessMain ||
                SLApplication::sceneID == SID_VideoTrackChessScnd)
            {
                SLfloat fovH = ac->cameraFovHDeg();
                SLfloat err  = ac->reprojectionError();
                ss << "Tracking Chessboard on " << (SLCVCapture::instance()->videoType() == VT_MAIN ? "main " : "scnd. ") << "camera. ";
                if (ac->state() == CS_calibrated)
                    ss << "FOVH: " << fovH << ", error: " << err;
                else
                    ss << "Not calibrated. FOVH guessed: " << fovH << " degrees.";
                s->info(ss.str());
            }
        } //...................................................................

        //copy image to video texture
        if (ac->state() == CS_calibrated && ac->showUndistorted())
        {
            SLCVMat undistorted;
            ac->remap(SLCVCapture::instance()->lastFrame, undistorted);

            SLCVCapture::instance()->videoTexture()->copyVideoImage(undistorted.cols,
                                                                    undistorted.rows,
                                                                    SLCVCapture::instance()->format,
                                                                    undistorted.data,
                                                                    undistorted.isContinuous(),
                                                                    true);
        }
        else
        {
            SLCVCapture::instance()->videoTexture()->copyVideoImage(SLCVCapture::instance()->lastFrame.cols,
                                                                    SLCVCapture::instance()->lastFrame.rows,
                                                                    SLCVCapture::instance()->format,
                                                                    SLCVCapture::instance()->lastFrame.data,
                                                                    SLCVCapture::instance()->lastFrame.isContinuous(),
                                                                    true);
        }

        SLCVTracked::trackingTimesMS.set(SLApplication::timeMS() - trackingTimeStartMS);
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
