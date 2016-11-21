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
#include <SLSceneView.h>
#include <SLCVCapture.h>

//-----------------------------------------------------------------------------
// Global static variables
SLCVMat             SLCVCapture::lastFrame;
SLPixelFormat       SLCVCapture::format;
cv::VideoCapture    SLCVCapture::_captureDevice;
//-----------------------------------------------------------------------------
//! Opens the capture device and returns the frame size
SLVec2i SLCVCapture::open(SLint deviceNum)
{
    try
    {
        _captureDevice.open(deviceNum);

        if (!_captureDevice.isOpened())
            return SLVec2i::ZERO;

        if (SL::noTestIsRunning())
            SL_LOG("Capture devices created.\n");

        SLint w = (int)_captureDevice.get(CV_CAP_PROP_FRAME_WIDTH);
        SLint h = (int)_captureDevice.get(CV_CAP_PROP_FRAME_HEIGHT);
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
    SLScene* s = SLScene::current;

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

            // Crop input image if it doesn't match the screens aspect ratio
            if (s->usesVideoAsBckgrnd())
            {
                SLint width = 0;    // width in pixels of the destination image
                SLint height = 0;   // height in pixels of the destination image
                SLint cropH = 0;    // crop height in pixels of the source image 
                SLint cropW = 0;    // crop width in pixels of the source image 

                SLfloat inWdivH = (SLfloat)lastFrame.cols / (SLfloat)lastFrame.rows;
                SLfloat outWdivH = s->sceneViews()[0]->scrWdivH();

                // Check for cropping
                if (SL_abs(inWdivH - outWdivH) > 0.01f)
                {   if (inWdivH > outWdivH) // crop input image left & right 
                    {   width = (SLint)((SLfloat)lastFrame.rows * outWdivH);
                        height = lastFrame.rows;
                        cropW = (SLint)((SLfloat)(lastFrame.cols - width) * 0.5f);    
                    } else // crop input image at top & bottom
                    {   width = lastFrame.cols;
                        height = (SLint)((SLfloat)lastFrame.cols / outWdivH);
                        cropH = (SLint)((SLfloat)(lastFrame.rows - height) * 0.5f);
                    }
                    lastFrame(SLCVRect(cropW, cropH, width, height)).copyTo(lastFrame);
                    //imwrite("AfterCropping.bmp", lastFrame);
                }
            }

            // Use the datastart pointer to access the region of interest
            s->copyVideoImage(lastFrame.cols,
                              lastFrame.rows,
                              format,
                              lastFrame.data,
                              lastFrame.isContinuous(),
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
