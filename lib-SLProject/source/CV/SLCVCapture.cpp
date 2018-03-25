//#############################################################################
//  File:      SLCVCapture.cpp
//  Purpose:   OpenCV Capture Device
//  Authors:   Michael Goettlicher, Marcus Hudritsch, Jan Dellsperger
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.4 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <SLApplication.h>
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
SLstring            SLCVCapture::videoDefaultPath = "../_data/videos/";
SLstring            SLCVCapture::videoFilename = "";
SLbool              SLCVCapture::videoLoops = true;
//-----------------------------------------------------------------------------
//! Opens the capture device and returns the frame size
/* This so far called in SLScene::onAfterLoad if a scene uses a live video by
setting the the SLScene::_videoType to VT_MAIN or VT_SCND.
*/
SLVec2i SLCVCapture::open(SLint deviceNum)
{
    try
    {   _captureDevice.open(deviceNum);

        if (!_captureDevice.isOpened())
            return SLVec2i::ZERO;
        
        SL_LOG("Capture devices created.\n");

        SLint w = (int)_captureDevice.get(CV_CAP_PROP_FRAME_WIDTH);
        SLint h = (int)_captureDevice.get(CV_CAP_PROP_FRAME_HEIGHT);
        SL_LOG("CV_CAP_PROP_FRAME_WIDTH : %d\n", w);
        SL_LOG("CV_CAP_PROP_FRAME_HEIGHT: %d\n", h);

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
//! Opens the video file instead of a camera feed.
/* This so far called in SLScene::onAfterLoad if a scene uses a video by
setting the the SLScene::_videoType to VT_FILE.
*/
SLVec2i SLCVCapture::openFile()
{
    try
    {   // Load the file directly
        if (!SLFileSystem::fileExists(videoFilename))
        {   videoFilename = videoDefaultPath + videoFilename;
            if (!SLFileSystem::fileExists(videoFilename))
            {   SLstring msg = "SLCVCapture::openFile: File not found: " + videoFilename;
                SL_EXIT_MSG(msg.c_str());
            }
        }

        _captureDevice.open(videoFilename);

        if (!_captureDevice.isOpened())
        {
            SL_LOG("SLCVCapture::openFile: Failed to open video file.");
            return SLVec2i::ZERO;
        }

        SL_LOG("Capture devices created with video.\n");

        SLint w = (int)_captureDevice.get(CV_CAP_PROP_FRAME_WIDTH);
        SLint h = (int)_captureDevice.get(CV_CAP_PROP_FRAME_HEIGHT);
        SL_LOG("CV_CAP_PROP_FRAME_WIDTH : %d\n", w);
        SL_LOG("CV_CAP_PROP_FRAME_HEIGHT: %d\n", h);

        hasSecondaryCamera = false;

        return SLVec2i(w, h);
    }
    catch (exception e)
    {
        SL_LOG("SLCVCapture::openFile: Exception during OpenCV video capture creation with video file\n");
    }
    return SLVec2i::ZERO;
}
//-----------------------------------------------------------------------------
void SLCVCapture::release()
{
    if (_captureDevice.isOpened())
        _captureDevice.release();

    videoFilename = "";
}
//-----------------------------------------------------------------------------
/*! Grabs a new frame from the OpenCV capture device or video file and calls
SLCVCapture::adjustForSL. This function can also be called by Android or iOS
app for grabbing a frame of a video file. Android and iOS use their own
capture functionality.
*/
void SLCVCapture::grabAndAdjustForSL()
{
    SLCVCapture::startCaptureTimeMS = SLApplication::scene->timeMilliSec();

    try
    {   if (_captureDevice.isOpened())
        {
            if (!_captureDevice.read(lastFrame))
            {
                // Try to loop the video
                if (videoFilename != "" && videoLoops)
                {   _captureDevice.set(CV_CAP_PROP_POS_FRAMES, 0);
                    if (!_captureDevice.read(lastFrame))
                        return;
                }
                else return;
            }

            adjustForSL();
        }
        else
        {   static bool logOnce = true;
            if (logOnce)
            {   SL_LOG("OpenCV: Capture device or video file is not open!\n");
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
images no matter with what they where captured:
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
    SLScene* s = SLApplication::scene;
    format = SLCVImage::cv2glPixelFormat(lastFrame.type());

    // Set capture size before cropping
    captureSize = lastFrame.size();

    /////////////////
    // 1) Cropping //
    /////////////////

    // Cropping is done almost always.
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

    if (SLApplication::activeCalib->isMirroredH())
    {   SLCVMat mirrored;
        if (SLApplication::activeCalib->isMirroredV())
            cv::flip(SLCVCapture::lastFrame, mirrored,-1);
        else cv::flip(SLCVCapture::lastFrame, mirrored, 1);
        SLCVCapture::lastFrame = mirrored;
    } else
    if (SLApplication::activeCalib->isMirroredV())
    {   SLCVMat mirrored;
        if (SLApplication::activeCalib->isMirroredH())
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
    SLCVCapture::startCaptureTimeMS = SLApplication::scene->timeMilliSec();

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

        // calculate padding NO. of bgrRowOffset bytes (= step in OpenCV terminology)
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
//-----------------------------------------------------------------------------
//! YUV to RGB image infos. Offset value can be negative for mirrored copy.
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
}
//-----------------------------------------------------------------------------
//! YUV to RGB image infos. Offset value can be negative for mirrored copy.
struct colorBGR
{
    SLubyte b, g, r;
};
//-----------------------------------------------------------------------------
//! YUV to RGB image infos. Offset value can be negative for mirrored copy.
struct YUV2RGB_ImageInfo
{
    int bgrColOffest;   //!< offset in bytes to the next bgr pixel (column)
    int grayColOffest;  //!< offset in bytes to the next gray pixel (column)
    int yColOffest;     //!< offset in bytes to the next y pixel (column)
    int uColOffest;     //!< offset in bytes to the next u pixel (column)
    int vColOffset;     //!< offset in bytes to the next v pixel (column)
    int bgrRowOffset;   //!< offset in bytes to the next bgr row
    int grayRowOffset;  //!< offset in bytes to the next grayscale row
    int yRowOffset;     //!< offset in bytes to the y value of the next row
    int uRowOffset;     //!< offset in bytes to the u value of the next row
    int vRowOffest;     //!< offset in bytes to the v value of the next row
};
//-----------------------------------------------------------------------------
//! YUV to RGB image block infos that are different per thread
struct YUV2RGB_BlockInfo
{
    YUV2RGB_ImageInfo *imageInfo;   //!< Pointer to the image info
    int     rowCount;   //!< Num. of rows in block
    int     colCount;   //!< Num. of columns in block
    SLubyte* bgrRow;    //!< Pointer to the bgr row
    SLubyte* grayRow;   //!< Pointer to the grayscale row
    SLubyte* yRow;      //!< Pointer to the y value row
    SLubyte* uRow;      //!< Pointer to the u value row
    SLubyte* vRow;      //!< Pointer to the v value row
};
//-----------------------------------------------------------------------------
//! YUV to RGB conversion function called by multiple threads
/*!
/param info image block information struct with thread specific information
*/
void* convertYUV2RGB(YUV2RGB_BlockInfo* block)
{
    YUV2RGB_ImageInfo* image = block->imageInfo;

    for (int row = 0; row < block->rowCount; ++row)
    {
        colorBGR* bgrCol = (colorBGR *)block->bgrRow;
        SLubyte* grayCol = block->grayRow;
        SLubyte* yCol    = block->yRow;
        SLubyte* uCol    = block->uRow;
        SLubyte* vCol    = block->vRow;

        // convert 2 pixels in the inner loop
        for (int col = 0; col < block->colCount; col+=2)
        {
            yuv2rbg(*yCol, *uCol, *vCol, bgrCol->r,bgrCol->g, bgrCol->b);
            *grayCol = *yCol;

            bgrCol  += image->bgrColOffest;
            grayCol += image->grayColOffest;
            yCol    += image->yColOffest;

            yuv2rbg(*yCol, *uCol, *vCol, bgrCol->r,bgrCol->g, bgrCol->b);
            *grayCol = *yCol;

            bgrCol  += image->bgrColOffest;
            grayCol += image->grayColOffest;
            yCol    += image->yColOffest;

            uCol    += image->uColOffest;
            vCol    += image->vColOffset;
        }

        block->bgrRow  += image->bgrRowOffset;
        block->grayRow += image->grayRowOffset;
        block->yRow    += image->yRowOffset;

        // if odd row
        if (row & 1)
        {   block->uRow += image->uRowOffset;
            block->vRow += image->vRowOffest;
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
\param srcW        Source image width in pixel
\param srcH        Source image height in pixel
\param y           Pointer to first byte of the top left pixel of the y-plane
\param yBytes      Size in bytes of the y-plane (must be srcW x srcH)
\param yColOffset  Offset in bytes to the next pixel in the y-plane
\param yRowOffset  Offset in bytes to the next line in the y-plane
\param u           Pointer to first byte of the top left pixel of the u-plane
\param uBytes      Size in bytes of the u-plane
\param uColOffset  Offset in bytes to the next pixel in the u-plane
\param uRowOffset  Offset in bytes to the next line in the u-plane
\param v           Pointer to first byte of the top left pixel of the v-plane
\param vBytes      Size in bytes of the v-plane
\param vColOffset  Offset in bytes to the next pixel in the v-plane
\param vRowOffset  Offset in bytes to the next line in the v-plane
*/
void SLCVCapture::copyYUVPlanes(int srcW, int srcH,
                                SLuchar* y, int yBytes, int yColOffset, int yRowOffset,
                                SLuchar* u, int uBytes, int uColOffset, int uRowOffset,
                                SLuchar* v, int vBytes, int vColOffset, int vRowOffset)
{
    // pointer to the active scene
    SLScene* s = SLApplication::scene;

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
    bool mirrorH = SLApplication::activeCalib->isMirroredH();
    bool mirrorV = SLApplication::activeCalib->isMirroredV();

    // Create output color (BGR) and grayscale images
    lastFrame     = SLCVMat(dstH, dstW, CV_8UC(3));
    lastFrameGray = SLCVMat(dstH, dstW, CV_8UC(1));
    format        = SLCVImage::cv2glPixelFormat(lastFrame.type());

    // Bugfix on some devices with wrong pixel offsets
    if (yRowOffset==uRowOffset && uColOffset==1)
    {   uColOffset = 2;
        vColOffset = 2;
    }

    SLubyte* bgrRow  = lastFrame.data;
    SLubyte* grayRow = lastFrameGray.data;

    int bgrColBytes  = 3;
    int bgrRowBytes  = dstW * bgrColBytes;
    int grayColBytes = 1;
    int grayRowBytes = dstW * grayColBytes;

    // Adjust the offsets depending on the horizontal mirroring
    int bgrRowOffset  = dstW * bgrColBytes;
    int grayRowOffset = dstW;
    if (mirrorH) {
        bgrRow  += (dstH - 1) * bgrRowBytes;
        grayRow += (dstH - 1) * grayRowBytes;
        bgrRowOffset  *= -1;
        grayRowOffset *= -1;
    }

    // Adjust the offsets depending on the vertical mirroring
    int bgrColOffset = 1;
    int grayColOffset = grayColBytes;
    if (mirrorV) {
        bgrRow  += (bgrRowBytes - bgrColBytes);
        grayRow += (grayRowBytes - grayColBytes);
        bgrColOffset  *= -1;
        grayColOffset *= -1;
    }

    // Set source buffer pointers
    int halfCropH = cropH/2;
    int halfCropW = cropW/2;
    SLubyte* yRow = y +     cropH*yRowOffset +     cropW*yColOffset;
    SLubyte* uRow = u + halfCropH*uRowOffset + halfCropW*uColOffset;
    SLubyte* vRow = v + halfCropH*vRowOffset + halfCropW*vColOffset;

    // Set the information common for all thread blocks
    YUV2RGB_ImageInfo imageInfo;
    imageInfo.bgrColOffest  = bgrColOffset;
    imageInfo.grayColOffest = grayColOffset;
    imageInfo.yColOffest    = yColOffset;
    imageInfo.uColOffest    = uColOffset;
    imageInfo.vColOffset    = vColOffset;
    imageInfo.bgrRowOffset  = bgrRowOffset;
    imageInfo.grayRowOffset = grayRowOffset;
    imageInfo.yRowOffset    = yRowOffset;
    imageInfo.uRowOffset    = uRowOffset;
    imageInfo.vRowOffest    = vRowOffset;

    // Prepare the threads (hyperthreads seam to be unefficient on ARM)
    const int threadNum = 4; //SL_max(thread::hardware_concurrency(), 1U);
    vector<thread> threads;
    YUV2RGB_BlockInfo threadInfos[threadNum];
    int rowsPerThread = dstH / (threadNum + 1);
    int halfRowsPerThread = (int)(rowsPerThread*0.5f);
    int rowsHandled = 0;

    // Launch threadNum-1 threads on different blocks of the image
    for(int i = 0; i < threadNum-1; i++)
    {
        YUV2RGB_BlockInfo* info = threadInfos + i;
        info->imageInfo = &imageInfo;
        info->bgrRow    = bgrRow;
        info->grayRow   = grayRow;
        info->yRow      = yRow;
        info->uRow      = uRow;
        info->vRow      = vRow;
        info->rowCount  = rowsPerThread;
        info->colCount  = dstW;

        ////////////////////////////////////////////////
        threads.push_back(thread(convertYUV2RGB, info));
        ////////////////////////////////////////////////

        rowsHandled += rowsPerThread;

        bgrRow  += bgrRowOffset  *     rowsPerThread;
        grayRow += grayRowOffset *     rowsPerThread;
        yRow    += yRowOffset    *     rowsPerThread;
        uRow    += uRowOffset    * halfRowsPerThread;
        vRow    += vRowOffset    * halfRowsPerThread;
    }

    // Launch the last block on the main thread
    YUV2RGB_BlockInfo infoMain;
    infoMain.imageInfo = &imageInfo;
    infoMain.bgrRow    = bgrRow;
    infoMain.grayRow   = grayRow;
    infoMain.yRow      = yRow;
    infoMain.uRow      = uRow;
    infoMain.vRow      = vRow;
    infoMain.rowCount  = (dstH - rowsHandled);
    infoMain.colCount  = dstW;

    convertYUV2RGB(&infoMain);

    // Join all threads to continue single threaded
    for(auto& thread : threads) thread.join();

    // Stop the capture time displayed in the statistics info
    s->captureTimesMS().set(s->timeMilliSec() - SLCVCapture::startCaptureTimeMS);
}
//------------------------------------------------------------------------------
