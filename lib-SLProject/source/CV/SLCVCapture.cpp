//#############################################################################
//  File:      SLCVCapture.cpp
//  Purpose:   OpenCV Capture Device
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLCVCapture.h>

//-----------------------------------------------------------------------------
// Global static variables
SLCVMat             SLCVCapture::lastFrame;
SLCVMat             SLCVCapture::lastFrameGray;
SLPixelFormat       SLCVCapture::format;
cv::VideoCapture    SLCVCapture::_captureDevice;
SLCVSize            SLCVCapture::captureSize;
SLfloat             SLCVCapture::startCaptureTimeMS;
SLbool              SLCVCapture::hasSecondaryCamera = true;
SLint               SLCVCapture::requestedSizeIndex = 0;
//-----------------------------------------------------------------------------
//! Opens the capture device and returns the frame size
SLVec2i SLCVCapture::open(SLint deviceNum)
{
    try
    {   _captureDevice.open(deviceNum);

        if (!_captureDevice.isOpened())
            return SLVec2i::ZERO;

        if (SL::noTestIsRunning())
            SL_LOG("Capture devices created.\n");

        SLint w = (int)_captureDevice.get(CV_CAP_PROP_FRAME_WIDTH);
        SLint h = (int)_captureDevice.get(CV_CAP_PROP_FRAME_HEIGHT);
        cout << "CV_CAP_PROP_FRAME_WIDTH : " << w << endl;
        cout << "CV_CAP_PROP_FRAME_HEIGHT: " << h << endl;

        //_captureDevice.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        //_captureDevice.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

        hasSecondaryCamera = false;

        return SLVec2i(w, h);
    }
    catch (exception e)
    {
        SL_LOG("Exception during OpenCV video capture creation\n");
    }
    return SLVec2i::ZERO;
}

SLVec2i SLCVCapture::open(SLstring filePath)
{
    try
    {   _captureDevice.open(filePath);

        if (!_captureDevice.isOpened())
            return SLVec2i::ZERO;

        if (SL::noTestIsRunning())
            SL_LOG("Capture devices created.\n");

        SLint w = (int)_captureDevice.get(CV_CAP_PROP_FRAME_WIDTH);
        SLint h = (int)_captureDevice.get(CV_CAP_PROP_FRAME_HEIGHT);
        cout << "CV_CAP_PROP_FRAME_WIDTH : " << w << endl;
        cout << "CV_CAP_PROP_FRAME_HEIGHT: " << h << endl;

        //_captureDevice.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        //_captureDevice.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

        hasSecondaryCamera = false;

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
    SLCVCapture::startCaptureTimeMS = SLScene::current->timeMilliSec();

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
//! Does all adjustments needed for the SLScene::_videoTexture
/*! SLCVCapture::adjustForSL processes the following adjustments for all input
image no matter with what the where captured:
\n
1) Crops the input image if it doesn't match the screens aspect ratio. The
input image mostly does't fit the aspect of the output screen aspect. If the
input image is too high we crop it on top and bottom, if it is too wide we
crop it on the sides.
\n
2) Some cameras toward a face mirror the image and some do not. If a input
image should be mirrored or not is stored in SLCVCalibration::_isMirroredH
(H for horizontal) and SLCVCalibration::_isMirroredV (V for vertical).
\n
3) Many of the further processing steps are faster done on grayscale images.
We therefore create a copy that is grayscale converted.
*/
void SLCVCapture::adjustForSL()
{
    SLScene* s = SLScene::current;
    format = SLCVImage::cv2glPixelFormat(lastFrame.type());

    // Set capture size before cropping
    captureSize = lastFrame.size();

    /////////////////
    // 1) Cropping //
    /////////////////

    // Cropping is done almost always done.
    // So this is Android image copy loop #2

    SLfloat inWdivH = (SLfloat)lastFrame.cols / (SLfloat)lastFrame.rows;
    SLfloat outWdivH = s->sceneViews()[0]->scrWdivH();

    if (SL_abs(inWdivH - outWdivH) > 0.01f)
    {   SLint width = 0;    // width in pixels of the destination image
        SLint height = 0;   // height in pixels of the destination image
        SLint cropH = 0;    // crop height in pixels of the source image
        SLint cropW = 0;    // crop width in pixels of the source image
        
        if (inWdivH > outWdivH) // crop input image left & right
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

    //////////////////
    // 2) Mirroring //
    //////////////////

    // Mirroring is done for most selfie cameras.
    // So this is Android image copy loop #3

    if (SLScene::current->activeCalib()->isMirroredH())
    {   SLCVMat mirrored;
        if (SLScene::current->activeCalib()->isMirroredV())
             cv::flip(SLCVCapture::lastFrame, mirrored,-1);
        else cv::flip(SLCVCapture::lastFrame, mirrored, 1);
        SLCVCapture::lastFrame = mirrored;
    } else
    if (SLScene::current->activeCalib()->isMirroredV())
    {   SLCVMat mirrored;
        if (SLScene::current->activeCalib()->isMirroredH())
             cv::flip(SLCVCapture::lastFrame, mirrored,-1);
        else cv::flip(SLCVCapture::lastFrame, mirrored, 0);
        SLCVCapture::lastFrame = mirrored;
    }

    /////////////////////////
    // 3) Create grayscale //
    /////////////////////////

    // Creating a grayscale version from an YUV input source is stupid.
    // We just could take the Y channel.
    // Android image copy loop #4

    cv::cvtColor(lastFrame, lastFrameGray, cv::COLOR_BGR2GRAY);

    // Do not copy into the video texture here. It is done in SLScene:onUpdate

    s->captureTimesMS().set(s->timeMilliSec() - SLCVCapture::startCaptureTimeMS);
}
//-----------------------------------------------------------------------------
/*! This method is called by iOS and Android projects that capture their video
cameras on their own. We only adjust the color space. See the app-Demo-iOS and
app-Demo-Android projects for the usage.
*/
void SLCVCapture::loadIntoLastFrame(const SLint width,
                                    const SLint height,
                                    const SLPixelFormat format,
                                    const SLuchar* data,
                                    const SLbool isContinuous)
{
    SLCVCapture::startCaptureTimeMS = SLScene::current->timeMilliSec();

    // treat Android YUV to RGB conversion special
    if (format == PF_yuv_420_888)
    {
        SLCVMat yuv(height + height / 2, width, CV_8UC1, (void*)data);

        // Android image copy loop #1
        cvtColor(yuv, SLCVCapture::lastFrame, CV_YUV2RGB_NV21, 3);
    }
    else
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
            stride = (size_t)(bpl - width * bpp);
        }

        SLCVCapture::lastFrame = SLCVMat(height, width, cvType, (void*)data, stride);
    }
    
    adjustForSL();
}
//------------------------------------------------------------------------------
//! Copies and converts the video image in YUV_420 format to RGB and Grayscale
/*! SLCVCapture::copyYUVPlanes copies and converts the video image in YUV_420
format to the RGB image in SLCVCapture::lastFrame and the Y channel the grayscale
image in SLCVCapture::lastFrameGray.\n
In the YUV_420 format only the luminosity channel Y has the full resolution
(one byte per pixel). The color channels U and V are subsampled and have only
one byte per 4 pixel. See also https://en.wikipedia.org/wiki/Chroma_subsampling
\n
In addition the routine crops and mirrors the image if needed. So the following
processing steps should be done hopefully in a single loop:
\n
1) Crops the input image if it doesn't match the screens aspect ratio. The
input image mostly does't fit the aspect of the output screen aspect. If the
input image is too high we crop it on top and bottom, if it is too wide we
crop it on the sides.
\n
2) Some cameras toward a face mirror the image and some do not. If a input
image should be mirrored or not is stored in SLCVCalibration::_isMirroredH
(H for horizontal) and SLCVCalibration::_isMirroredV (V for vertical).
\n
3) The most expensive part of course is the color space conversion from the
YUV to RGB conversion. According to Wikipedia the conversion is defined as:
\n
- C = 1.164*(Y-16); D = U-128; E = V-128
- R = clip(round(C + 1.596*E))
- G = clip(round(C - 0.391*D - 0.813*E))
- B = clip(round(C + 2.018*D))
\n
A faster integer version with bit shifting is:\n
- C = 298*(Y-16)+128; D = U-128; E = V-128
- R = clip((C + 409*E) >> 8)
- G = clip((C - 100*D - 208*E) >> 8)
- B = clip((C + 516*D) >> 8)
\n
4) Many of the image processing tasks are faster done on grayscale images.
We therefore create a copy of the y-channel into SLCVCapture::lastFrameGray.
\n
\param srcW         Source image width in pixel
\param srcH         Source image height in pixel
\param y            Pointer to first byte of the top left pixel of the y-plane
\param ySize        Size in bytes of the y-plane (must be srcW x srcH)
\param yPixStride   Offest in bytes to the next pixel in the y-plane
\param yLineStride  Offest in bytes to the next line in the y-plane
\param u            Pointer to first byte of the top left pixel of the u-plane
\param uSize        Size in bytes of the u-plane
\param uPixStride   Offest in bytes to the next pixel in the u-plane
\param uLineStride  Offest in bytes to the next line in the u-plane
\param v            Pointer to first byte of the top left pixel of the v-plane
\param vSize        Size in bytes of the v-plane
\param vPixStride   Offest in bytes to the next pixel in the v-plane
\param vLineStride  Offest in bytes to the next line in the v-plane
*/
void SLCVCapture::copyYUVPlanes(int srcW, int srcH,
                                SLuchar* y, int ySize, int yPixStride, int yLineStride,
                                SLuchar* u, int uSize, int uPixStride, int uLineStride,
                                SLuchar* v, int vSize, int vPixStride, int vLineStride)
{
    // pointer to the active scene
    SLScene* s = SLScene::current;

    // Set the start time to measure the MS for the whole conversion
    SLCVCapture::startCaptureTimeMS = s->timeMilliSec();

    // input image aspect ratio
    SLfloat srcWdivH = (SLfloat)srcW / srcH;

    // output image aspect ratio = aspect of the always landscape screen
    SLfloat dstWdivH = s->sceneViews()[0]->scrWdivH();

    SLint dstW = srcW;  // width in pixels of the destination image
    SLint dstH = srcH;  // height in pixels of the destination image
    SLint cropH = 0;    // crop height in pixels of the source image
    SLint cropW = 0;    // crop width in pixels of the source image

    // Crop image if source and destination aspect is not the same
    if (SL_abs(srcWdivH - dstWdivH) > 0.01f)
    {
        if (srcWdivH > dstWdivH) // crop input image left & right
        {   dstW  = (SLint)((SLfloat)srcH * dstWdivH);
            dstH  = srcH;
            cropW = (SLint)((SLfloat)(srcW - dstW) * 0.5f);
        } else // crop input image at top & bottom
        {   dstW  = srcW;
            dstH  = (SLint)((SLfloat)srcW / dstWdivH);
            cropH = (SLint)((SLfloat)(srcH - dstH) * 0.5f);
        }
    }

    // Get the infos if the destination image must be mirrored
    bool mirrorH = s->activeCalib()->isMirroredH();
    bool mirrorV = s->activeCalib()->isMirroredV();

    /*
    Now do if possible only one loop over the source image to fill up the
    RGB image in SLCVCapture::lastFrame and the grayscale image in
    SLCVCapture::lastFrameGray.

    In the OpenCV docs:
    http://docs.opencv.org/3.1.0/db/da5/tutorial_how_to_scan_images.html
    the fasted way to iterate over an image matrix is in plain old C:

    Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
    {
        // accept only char type matrices
        CV_Assert(I.depth() == CV_8U);
        int channels = I.channels();
        int nRows = I.rows;
        int nCols = I.cols * channels;
        if (I.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }
        int i,j;
        uchar* p;
        for( i = 0; i < nRows; ++i)
        {
            p = I.ptr<uchar>(i);
            for ( j = 0; j < nCols; ++j)
            {
                p[j] = table[p[j]];
            }
        }
        return I;
    }
    */

    // ???

    // Stop the capture time displayed in the statistics info
    s->captureTimesMS().set(s->timeMilliSec() - SLCVCapture::startCaptureTimeMS);
}
//------------------------------------------------------------------------------
