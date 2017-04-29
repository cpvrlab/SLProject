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
#include <pthread.h>

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

        // calculate padding NO. of destStride bytes (= step in OpenCV terminology)
        size_t destStride = 0;
        if (!isContinuous)
        {
            SLint bitsPerPixel = bpp * 8;
            SLint bpl = ((width * bitsPerPixel + 31) / 32) * 4;
            destStride = (size_t)(bpl - width * bpp);
        }

        SLCVCapture::lastFrame = SLCVMat(height, width, cvType, (void*)data, destStride);
    }

    adjustForSL();
}


inline void yuv2rbg(SLubyte y, SLubyte u, SLubyte v,
                    SLubyte& r, SLubyte& g, SLubyte& b)
{
    // Conversion from:
    // https://de.wikipedia.org/wiki/YUV-Farbmodell
    //float c = 1.164f*(float)(yVal-16);
    //float d = (float)(uVal-128);
    //float e = (float)(vVal-128);
    //r = clipFToUInt8(c + 1.596f*e);
    //g = clipFToUInt8(c - 0.391f*d - 0.813f*e);
    //b = clipFToUInt8(c + 2.018f*d);

    // Conversion from:
    // http://www.wordsaretoys.com/2013/10/18/making-yuv-conversion-a-little-faster
    // I've multiplied each floating point constant by 1024 and truncated it.
    // Now I can add/subtract the scaled integers, and apply a bit shift right to
    // divide each result by 1024
    int e = v - 128;
    int d = u - 128;
    int a0 = 1192 * (y - 16);
    int a1 = 1634 * e;
    int a2 = 832 * e;
    int a3 = 400 * d;
    int a4 = 2066 * d;
    r = (SLubyte)SL_clamp(     (a0 + a1) >> 10, 0, 255);
    g = (SLubyte)SL_clamp((a0 - a2 - a3) >> 10, 0, 255);
    b = (SLubyte)SL_clamp(     (a0 + a4) >> 10, 0, 255);

    // Similar as above with multiplication and division by 2^8
    //int c = 298*(yVal - 16) + 128;
    //int d = uVal - 128;
    //int e = vVal - 128;
    //r = (SLubyte)SL_clamp(        (c + 409*e) >> 8, 0, 255);
    //g = (SLubyte)SL_clamp((c - 100*d - 208*e) >> 8, 0, 255);
    //b = (SLubyte)SL_clamp(        (c + 516*d) >> 8, 0, 255);

    //r = clipFToUInt8(yVal + 1.40f*(vVal-128));
    //g = clipFToUInt8(yVal - 0.34f*(uVal-128) - 0.71f*(vVal-128));
    //b = clipFToUInt8(yVal + 1.77f*(uVal-128));
}

inline SLubyte clipFToUInt8(float val)
{
    float result = val;

    if (result > 255.0f)
        result = 255.0f;

    if (result < 0.0f)
        result = 0.0f;

    return(SLubyte)result;
}

struct jd_color
{
    SLubyte b, g, r;
};

struct jd_threadInfoHeader
{
    int destPixSize;
    int grayPixSize;
    int yPixSize;
    int uPixSize;
    int vPixSize;
    int destStride;
    int grayStride;
    int yStride;
    int uStride;
    int vStride;
};

struct jd_threadInfo
{
    jd_threadInfoHeader *header;
    int rowCount;
    int colCount;
    SLubyte *destRow;
    SLubyte *grayRow;
    SLubyte *yRow;
    SLubyte *uRow;
    SLubyte *vRow;
};

void* copyToBuffer(void *arg)
{
    jd_threadInfo *info = (jd_threadInfo *)arg;

    int rowCount = info->rowCount;
    int colCount = info->colCount;

    jd_threadInfoHeader *header = info->header;
    int destPixSize = header->destPixSize;
    int grayPixSize = header->grayPixSize;
    int yPixSize = header->yPixSize;
    int uPixSize = header->uPixSize;
    int vPixSize = header->vPixSize;
    int destStride = header->destStride;
    int grayStride = header->grayStride;
    int yStride = header->yStride;
    int uStride = header->uStride;
    int vStride = header->vStride;

    for (int row = 0; row < rowCount; ++row)
    {
        jd_color* pixel = (jd_color *)info->destRow;
        SLubyte* grayPix = info->grayRow;
        SLubyte* yPix = info->yRow;
        SLubyte* uPix = info->uRow;
        SLubyte* vPix = info->vRow;

        for (int col = 0; col < colCount; ++col)
        {
            yuv2rbg(*yPix, *uPix, *vPix, pixel->r,pixel->g, pixel->b);
            *grayPix = *yPix;

            pixel += destPixSize;
            grayPix += grayPixSize;
            yPix += yPixSize;

            //if (col % 2) {
            if (col & 1) {
                uPix += uPixSize;
                vPix += vPixSize;
            }
        }

        info->destRow += destStride;
        info->grayRow += grayStride;

        info->yRow += yStride;

        //if (row % 2) {
        if (row & 1) {
            info->uRow += uStride;
            info->vRow += vStride;
        }
    }

    return 0;
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
\param yPixStride   Offset in bytes to the next pixel in the y-plane
\param yLineStride  Offset in bytes to the next line in the y-plane
\param u            Pointer to first byte of the top left pixel of the u-plane
\param uSize        Size in bytes of the u-plane
\param uPixStride   Offset in bytes to the next pixel in the u-plane
\param uLineStride  Offset in bytes to the next line in the u-plane
\param v            Pointer to first byte of the top left pixel of the v-plane
\param vSize        Size in bytes of the v-plane
\param vPixStride   Offset in bytes to the next pixel in the v-plane
\param vLineStride  Offset in bytes to the next line in the v-plane
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
    {   if (srcWdivH > dstWdivH) // crop input image left & right
        {   dstW  = (SLint)((SLfloat)srcH * dstWdivH);
            dstH  = srcH;
            cropW = (SLint)((SLfloat)(srcW - dstW) * 0.5f);
        }
        else // crop input image at top & bottom
        {   dstW  = srcW;
            dstH  = (SLint)((SLfloat)srcW / dstWdivH);
            cropH = (SLint)((SLfloat)(srcH - dstH) * 0.5f);
        }
    }

    // Get the infos if the destination image must be mirrored
    bool mirrorH = s->activeCalib()->isMirroredH();
    bool mirrorV = s->activeCalib()->isMirroredV();

    // Create output color and grayscale images
    lastFrame = SLCVMat(dstH, dstW, CV_8UC(3));
    lastFrameGray = SLCVMat(dstH, dstW, CV_8UC(1));
    format = SLCVImage::cv2glPixelFormat(lastFrame.type());

    // Bugfix on some devices with wrong pixel strides
    if (yLineStride==uLineStride && uPixStride==1)
    {   uPixStride = 2;
        vPixStride = 2;
    }

    SLubyte* destRow = lastFrame.data;
    SLubyte* grayRow = lastFrameGray.data;

    int destPixSize = 3;
    int destStride = dstW * destPixSize;
    int grayPixSize = 1;
    int grayStride = dstW * grayPixSize;

    int destRowStride = dstW * destPixSize;
    int grayRowStride = dstW;
    if (mirrorH) {
        destRow += (dstH - 1) * destStride;
        grayRow += (dstH - 1) * grayStride;
        destRowStride *= -1;
        grayRowStride *= -1;
    }

    int destColStride = 1;
    int grayColStride = grayPixSize;
    if (mirrorV) {
        destRow += (destStride - destPixSize);
        grayRow += (grayStride - grayPixSize);
        destColStride *= -1;
        grayColStride *= -1;
    }

    int halfCropH = cropH/2;
    int halfCropW = cropW/2;

    SLubyte* yRow = y +     cropH*yLineStride +     cropW*yPixStride;
    SLubyte* uRow = u + halfCropH*uLineStride + halfCropW*uPixStride;
    SLubyte* vRow = v + halfCropH*vLineStride + halfCropW*vPixStride;

    jd_threadInfoHeader tHeader;
    tHeader.destPixSize = destColStride;
    tHeader.grayPixSize = grayColStride;
    tHeader.yPixSize    = yPixStride;
    tHeader.uPixSize    = uPixStride;
    tHeader.vPixSize    = vPixStride;
    tHeader.destStride  = destRowStride;
    tHeader.grayStride  = grayRowStride;
    tHeader.yStride     = yLineStride;
    tHeader.uStride     = uLineStride;
    tHeader.vStride     = vLineStride;

    int threadNum = 4;
    pthread_t threads[threadNum];
    jd_threadInfo threadInfos[threadNum];
    int rowsPerThread = dstH / (threadNum + 1);
    int halfRowsPerThread = (int)(rowsPerThread*0.5f);
    int rowsHandled = 0;

    for(int i = 0; i < threadNum; i++)
    {
        jd_threadInfo* info = threadInfos + i;
        info->header   = &tHeader;
        info->destRow  = destRow;
        info->grayRow  = grayRow;
        info->yRow     = yRow;
        info->uRow     = uRow;
        info->vRow     = vRow;
        info->rowCount = rowsPerThread;
        info->colCount = dstW;

        pthread_t* t = threads + i;
        pthread_create(t, NULL, copyToBuffer, info);
        rowsHandled += rowsPerThread;

        destRow += destRowStride *     rowsPerThread;
        grayRow += grayRowStride *     rowsPerThread;
        yRow    += yLineStride   *     rowsPerThread;
        uRow    += uLineStride   * halfRowsPerThread;
        vRow    += vLineStride   * halfRowsPerThread;
    }

    jd_threadInfo infoMain;
    infoMain.header   = &tHeader;
    infoMain.destRow  = destRow;
    infoMain.grayRow  = grayRow;
    infoMain.yRow     = yRow;
    infoMain.uRow     = uRow;
    infoMain.vRow     = vRow;
    infoMain.rowCount = (dstH - rowsHandled);
    infoMain.colCount = dstW;

    copyToBuffer(&infoMain);

    // Join all threads to continue single threaded
    for(int i = 0; i < threadNum; i++)
    {   pthread_t *t = threads + i;
        pthread_join(*t, NULL);
    }

    // Stop the capture time displayed in the statistics info
    s->captureTimesMS().set(s->timeMilliSec() - SLCVCapture::startCaptureTimeMS);
}
//------------------------------------------------------------------------------
