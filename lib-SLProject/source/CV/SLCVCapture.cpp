//#############################################################################
//  File:      SLCVCapture.cpp
//  Purpose:   OpenCV Capture Device
//  Author:    Marcus Hudritsch
//  Date:      June 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#include <SLScene.h>
#include <SLCVCapture.h>

//-----------------------------------------------------------------------------
// Global static variables
Mat             SLCVCapture::lastFrame;
SLPixelFormat   SLCVCapture::format;
VideoCapture    SLCVCapture::_captureDevice;
//-----------------------------------------------------------------------------
//! Opens the capture device and returns the frame size
SLVec2i SLCVCapture::open(int deviceNum)
{
    try
    {
        _captureDevice.open(deviceNum);

        if (!_captureDevice.isOpened())
            return SLVec2i::ZERO;

        if (SL::noTestIsRunning())
            SL_LOG("Capture devices created.\n");

        int w = (int)_captureDevice.get(CV_CAP_PROP_FRAME_WIDTH);
        int h = (int)_captureDevice.get(CV_CAP_PROP_FRAME_HEIGHT);
        cout << "CV_CAP_PROP_FRAME_WIDTH : " << w << endl;
        cout << "CV_CAP_PROP_FRAME_HEIGHT: " << h << endl;
        return SLVec2i(w, h);
    }
    catch (exception e)
    {
        SL_LOG("Exception during OpenCV video capture creation\n");
    }
    return SLVec2i::ZERO;
}
//-----------------------------------------------------------------------------
//! Grabs a new frame from the capture device and copies it to the SLScene
void SLCVCapture::grabAndCopyToSL()
{
    try
    {   if (_captureDevice.isOpened())
        {
           if (!_captureDevice.read(lastFrame))
                return;

            // Set the according OpenGL format
            switch (lastFrame.type())
            {   case CV_8UC1: format = PF_luminance; break;
                case CV_8UC3: format = PF_bgr; break;
                case CV_8UC4: format = PF_bgra; break;
                default: SL_EXIT_MSG("OpenCV image format not supported");
            }

            // OpenGL ES doesn't support BGR or BGRA
            //cv::cvtColor(lastFrame, lastFrame, CV_BGR2RGB);
            //cv::flip(lastFrame, lastFrame, 0);

            SLScene::current->copyVideoImage(lastFrame.cols,
                                             lastFrame.rows,
                                             format,
                                             lastFrame.data,
                                             true);
        }
        else
        {   static bool logOnce = true;
            if (logOnce)
            {
                if (SL::noTestIsRunning())
                    SL_LOG("OpenCV: Unable to create capture device.\n");
                logOnce = false;
            }
        }
    }
    catch (exception e)
    {
        SL_LOG("Exception during OpenCV video capture creation\n");
    }
}
//-----------------------------------------------------------------------------
//! Copies the last frame to the SLScene
void SLCVCapture::copyFrameToSL()
{
    SLScene::current->copyVideoImage(lastFrame.cols,
                                     lastFrame.rows,
                                     format,
                                     lastFrame.data,
                                     true);
}
//-----------------------------------------------------------------------------
