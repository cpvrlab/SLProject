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
SLCVMat             SLCVCapture::lastFrameGray;
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

        _captureDevice.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        _captureDevice.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

        return SLVec2i(w, h);
    }
    catch (exception e)
    {
        SL_LOG("Exception during OpenCV video capture creation\n");
    }
    return SLVec2i::ZERO;
}
//-----------------------------------------------------------------------------
/*! Grabs a new frame from the OpenCV capture device and adjusts it for SL
*/
void SLCVCapture::grabAndAdjustForSL()
{
    try
    {   if (_captureDevice.isOpened())
        {
           if (!_captureDevice.read(lastFrame))
                return;

           adjustForSL();
        }
        else
        {   static bool logOnce = true;
            if (logOnce)
            {
                if (SL::noTestIsRunning())
                    SL_LOG("OpenCV: Capture Device is not open!\n");
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
//! Crops the last captured frame and copies it to the SLScene video texture
void SLCVCapture::adjustForSL()
{
    SLScene* s = SLScene::current;

    try
    {   // Set the according OpenGL format
        switch (lastFrame.type())
        {   case CV_8UC1: format = PF_red; break;
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

        // Create grayscale version in case of coloured lastFrame
        cv::cvtColor(lastFrame, lastFrameGray, cv::COLOR_BGR2GRAY);

        // Do not copy into the video texture here.
        // It is done in SLScene:onUpdate
    }
    catch (exception e)
    {
        SL_LOG("Exception in SLCVCapture::adjustForSL: %s\n", e.what());
    }
}
//-----------------------------------------------------------------------------
void SLCVCapture::loadIntoLastFrame(const SLint width,
                                    const SLint height,
                                    const SLPixelFormat format,
                                    const SLuchar* data,
                                    const SLbool isContinuous,
                                    const SLbool isTopLeft)
{
    // Set the according OpenCV format
    SLint cvType = 0, bpp = 0;
    
    switch (format)
    {   case PF_luminance:  {cvType = CV_8UC1; bpp = 1; break;}
        case PF_bgr:        {cvType = CV_8UC3; bpp = 3; break;}
        case PF_rgb:        {cvType = CV_8UC3; bpp = 3; break;}
        case PF_bgra:       {cvType = CV_8UC4; bpp = 4; break;}
        case PF_rgba:       {cvType = CV_8UC4; bpp = 4; break;}
        default: SL_EXIT_MSG("Pixel format not supported");
    }

    // calculate padding NO. of stride bytes (= step in OpenCV terminology)
    size_t stride = 0;
    if (!isContinuous)
    {
        SLint bitsPerPixel = bpp * 8;
        SLint bpl = ((width * bitsPerPixel + 31) / 32) * 4;
        stride = bpl - width * bpp;
    }

    SLCVCapture::lastFrame = SLCVMat(height, width, cvType, (void*)data, stride);
}
//------------------------------------------------------------------------------
