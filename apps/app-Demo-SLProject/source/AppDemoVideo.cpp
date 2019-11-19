//#############################################################################
//  File:      AppDemoVideo.cpp
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
#include <CVCapture.h>
#include <CVTrackedAruco.h>
#include <SLGLTexture.h>

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
        SLfloat        trackingTimeStartMS = SLApplication::timeMS();
        CVCalibration* ac                  = CVCapture::instance()->activeCalib;

        // Invalidate calibration if viewport aspect doesn't match calibration aspect ratio
        SLfloat calibWdivH              = ac->imageAspectRatio();
        SLbool  aspectRatioDoesNotMatch = Utils::abs(sv->viewportWdivH() - calibWdivH) > 0.01f;
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
            {
                // Try to read device lens and sensor information
                string strF = SLApplication::deviceParameter["DeviceLensFocalLength"];
                string strW = SLApplication::deviceParameter["DeviceSensorPhysicalSizeW"];
                string strH = SLApplication::deviceParameter["DeviceSensorPhysicalSizeH"];
                float  devF = strF.empty() ? 0.0f : stof(strF);
                float  devW = strW.empty() ? 0.0f : stof(strW);
                float  devH = strH.empty() ? 0.0f : stof(strH);
                ac->devFocalLength(devF);
                ac->devSensorSizeW(devW);
                ac->devSensorSizeH(devH);

                // Changes the state to CS_guessed
                ac->createFromGuessedFOV(CVCapture::instance()->lastFrame.cols,
                                         CVCapture::instance()->lastFrame.rows);
                sv->camera()->fov(ac->cameraFovVDeg());
            }
        }
        else if (ac->state() == CS_calibrateStream || ac->state() == CS_calibrateGrab)
        {
            ac->findChessboard(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrameGray, true);
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
        else if (ac->state() == CS_startCalculating)
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
                SLfloat fovH = ac->cameraFovHDeg();
                SLfloat err  = ac->reprojectionError();
                ss << "Tracking Chessboard on " << (CVCapture::instance()->videoType() == VT_MAIN ? "main " : "scnd. ") << "camera. ";
                if (ac->state() == CS_calibrated)
                    ss << "FOVH: " << fovH << ", error: " << err;
                else
                    ss << "Not calibrated. FOVH guessed: " << fovH << " degrees.";
                s->info(ss.str());
            }
        } //...................................................................

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
