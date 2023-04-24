//#############################################################################
//  File:      CVCapture.cpp
//  Purpose:   OpenCV Capture Device
//  Authors:   Michael Goettlicher, Marcus Hudritsch, Jan Dellsperger
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/*
The OpenCV library version 3.4 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVCamera.h>
#include <algorithm> // std::max
#include <CVCapture.h>
#include <CVImage.h>
#include <Utils.h>
#include <FtpUtils.h>
#include <AppDemo.h>
#include <Profiler.h>

//-----------------------------------------------------------------------------
CVCapture* CVCapture::_instance = nullptr;
//-----------------------------------------------------------------------------
//! Private constructor
CVCapture::CVCapture()
  : mainCam(CVCameraType::FRONTFACING),
    scndCam(CVCameraType::BACKFACING),
    videoFileCam(CVCameraType::VIDEOFILE)
{
    startCaptureTimeMS = 0.0f;
#ifdef APP_USES_CVCAPTURE
    hasSecondaryCamera = false;
#else
    hasSecondaryCamera = true;
#endif
    videoFilename      = "";
    videoLoops         = true;
    fps                = 1;
    frameCount         = 0;
    activeCamSizeIndex = -1;
    activeCamera       = nullptr;
    _captureTimesMS.init(60, 0);
}
//-----------------------------------------------------------------------------
//! Private constructor
CVCapture::~CVCapture()
{
    release();
    if (CVCapture::_instance)
    {
        delete CVCapture::_instance;
        CVCapture::_instance = nullptr;
    }
}
//-----------------------------------------------------------------------------
//! Opens the capture device and returns the frame size
/* This so far called in start if a scene uses a live video by
setting the CVCapture::videoType to VT_MAIN. On desktop systems the webcam
is the only and main camera.
*/
CVSize2i CVCapture::open(int deviceNum)
{
#ifndef SL_EMSCRIPTEN
    try
    {
        _captureDevice.open(deviceNum);

        if (!_captureDevice.isOpened())
            return CVSize2i(0, 0);

        Utils::log("SLProject", "Capture devices created.");
        //_captureDevice.set(cv::CAP_PROP_FRAME_WIDTH, 1440);
        //_captureDevice.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        int w = (int)_captureDevice.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)_captureDevice.get(cv::CAP_PROP_FRAME_HEIGHT);
        // Utils::log("SLProject", "CV_CAP_PROP_FRAME_WIDTH : %d", w);
        // Utils::log("SLProject", "CV_CAP_PROP_FRAME_HEIGHT: %d", h);

        hasSecondaryCamera = false;
        fps                = (float)_captureDevice.get(cv::CAP_PROP_FPS);
        frameCount         = 0;

        // Set one camera size entry
        CVCapture::camSizes.clear();
        CVCapture::camSizes.push_back(CVSize(w, h));
        CVCapture::activeCamSizeIndex = 0;

        return CVSize2i(w, h);
    }
    catch (exception& e)
    {
        Utils::log("SLProject", "Exception during OpenCV video capture creation: %s", e.what());
    }
#else
    WebCameraFacing facing;
    if (_videoType == VT_MAIN)
        facing = WebCameraFacing::BACK;
    else if (_videoType == VT_SCND)
        facing = WebCameraFacing::FRONT;
    _webCamera.open(facing);

    // We can't query the actual resolution of the camera because that is considered a security risk.
    // Therefore, we list some common resolutions. If the camera doesn't support the requested resolution,
    // the browser will simply switch to a supported one.
    camSizes           = {CVSize2i(640, 480),
                          CVSize2i(1280, 720),
                          CVSize2i(1920, 1080)};
    activeCamSizeIndex = 0;
#endif

    return CVSize2i(0, 0);
}
//-----------------------------------------------------------------------------
//! Opens the video file instead of a camera feed.
/* This so far called in CVCapture::start if a scene uses a video by
setting the the CVCapture::videoType to VT_FILE.
*/
CVSize2i CVCapture::openFile()
{
#ifndef SL_EMSCRIPTEN
    try
    { // Load the file directly
        if (!Utils::fileExists(videoFilename))
        {
            string msg = "CVCapture::openFile: File not found: " + videoFilename;
            Utils::exitMsg("SLProject", msg.c_str(), __LINE__, __FILE__);
        }

        _captureDevice.open(videoFilename);

        if (!_captureDevice.isOpened())
        {
            Utils::log("SLProject", "CVCapture::openFile: Failed to open video file.");
            return CVSize2i(0, 0);
        }

        // Utils::log("SLProject", "Capture devices created with video.");

        int w = (int)_captureDevice.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)_captureDevice.get(cv::CAP_PROP_FRAME_HEIGHT);
        // Utils::log("SLProject", "CV_CAP_PROP_FRAME_WIDTH : %d", w);
        // Utils::log("SLProject", "CV_CAP_PROP_FRAME_HEIGHT: %d", h);

        hasSecondaryCamera = false;
        fps                = (float)_captureDevice.get(cv::CAP_PROP_FPS);
        frameCount         = (int)_captureDevice.get(cv::CAP_PROP_FRAME_COUNT);

        return CVSize2i(w, h);
    }
    catch (exception& e)
    {
        Utils::log("SLProject", "CVCapture::openFile: Exception during OpenCV video capture creation with video file: %s", e.what());
    }
#endif

    return CVSize2i(0, 0);
}
//-----------------------------------------------------------------------------
//! starts the video capturing
void CVCapture::start(float viewportWdivH)
{
#if defined(SL_EMSCRIPTEN)
    open(VT_MAIN);
#elif defined(APP_USES_CVCAPTURE)
    if (_videoType != VT_NONE)
    {
        if (!isOpened())
        {
            CVSize2i videoSize;
            if (_videoType == VT_FILE && !videoFilename.empty())
                videoSize = openFile();
            else
                videoSize = open(0);

            if (videoSize != CVSize2i(0, 0))
            {
                grabAndAdjustForSL(viewportWdivH);
            }
        }
    }
#else
    if (_videoType == VT_FILE && !videoFilename.empty())
    {
        if (!isOpened())
        {
            CVSize2i videoSize = openFile();
        }
    }
#endif
}
//-----------------------------------------------------------------------------
bool CVCapture::isOpened()
{
#ifndef SL_EMSCRIPTEN
    return _captureDevice.isOpened();
#else
    return _webCamera.isOpened();
#endif
}
//-----------------------------------------------------------------------------
void CVCapture::release()
{
#ifndef SL_EMSCRIPTEN
    if (_captureDevice.isOpened())
        _captureDevice.release();
#else
    if (_webCamera.isOpened())
        _webCamera.close();
#endif

    videoFilename = "";
}
//-----------------------------------------------------------------------------
/*! Grabs a new frame from the OpenCV capture device or video file and calls
CVCapture::adjustForSL. This function can also be called by Android or iOS
app for grabbing a frame of a video file. Android and iOS use their own
capture functionality.
If viewportWdivH is negative the viewport aspect will be adapted to the video
aspect ratio.
*/
bool CVCapture::grabAndAdjustForSL(float viewportWdivH)
{
    PROFILE_FUNCTION();

    CVCapture::startCaptureTimeMS = _timer.elapsedTimeInMilliSec();

#ifndef SL_EMSCRIPTEN
    try
    {
        if (_captureDevice.isOpened())
        {
            if (!_captureDevice.read(lastFrame))
            {
                // Try to loop the video
                if (!videoFilename.empty() && videoLoops)
                {
                    _captureDevice.set(cv::CAP_PROP_POS_FRAMES, 0);
                    if (!_captureDevice.read(lastFrame))
                        return false;
                }
                else
                    return false;
            }
#    if defined(ANDROID)
            // Convert BGR to RGB on mobile phones
            cvtColor(CVCapture::lastFrame, CVCapture::lastFrame, cv::COLOR_BGR2RGB, 3);
#    endif
            adjustForSL(viewportWdivH);
        }
        else
        {
            static bool logOnce = true;
            if (logOnce)
            {
                Utils::log("SLProject", "OpenCV: Capture device or video file is not open!");
                logOnce = false;
                return false;
            }
        }
    }
    catch (exception& e)
    {
        Utils::log("SLProject", "Exception during OpenCV video capture creation: %s", e.what());
        return false;
    }
#else
    if (!_webCamera.isOpened())
    {
        SL_LOG("Web camera is not open!");
        return false;
    }

    if (activeCamera->camSizeIndex() != -1)
        _webCamera.setSize(camSizes[activeCamera->camSizeIndex()]);

    lastFrame = _webCamera.read();
    adjustForSL(viewportWdivH);
#endif

    return true;
}
//-----------------------------------------------------------------------------
/*! This method is called by iOS and Android projects that capture their video
cameras on their own. We only adjust the color space. See the app_demo_slproject/ios and
app_demo_slproject/android projects for the usage.
*/
void CVCapture::loadIntoLastFrame(const float           viewportWdivH,
                                  const int             width,
                                  const int             height,
                                  const CVPixelFormatGL newFormat,
                                  const uchar*          data,
                                  const bool            isContinuous)
{
    CVCapture::startCaptureTimeMS = _timer.elapsedTimeInMilliSec();

    // treat Android YUV to RGB conversion special
    if (newFormat == PF_yuv_420_888)
    {
        CVMat yuv(height + height / 2, width, CV_8UC1, (void*)data);

        // Android image copy loop #1
        cvtColor(yuv, CVCapture::lastFrame, cv::COLOR_YUV2RGB_NV21, 3);
    }
    // convert 4 channel images to 3 channel
    else if (newFormat == PF_bgra || format == PF_rgba)
    {
        CVMat rgba(height, width, CV_8UC4, (void*)data);
        cvtColor(rgba, CVCapture::lastFrame, cv::COLOR_RGBA2RGB, 3);
    }
    else
    {
        // Set the according OpenCV format
        int cvType = 0, bpp = 0;

        switch (newFormat)
        {
            case PF_luminance:
            {
                cvType = CV_8UC1;
                bpp    = 1;
                break;
            }
            case PF_bgr:
            {
                cvType = CV_8UC3;
                bpp    = 3;
                break;
            }
            case PF_rgb:
            {
                cvType = CV_8UC3;
                bpp    = 3;
                break;
            }
            case PF_bgra:
            {
                cvType = CV_8UC4;
                bpp    = 4;
                break;
            }
            case PF_rgba:
            {
                cvType = CV_8UC4;
                bpp    = 4;
                break;
            }
            default: Utils::exitMsg("SLProject", "Pixel format not supported", __LINE__, __FILE__);
        }

        // calculate padding NO. of bgrRowOffset bytes (= step in OpenCV terminology)
        size_t destStride = 0;
        if (!isContinuous)
        {
            int bitsPerPixel = bpp * 8;
            int bpl          = ((width * bitsPerPixel + 31) / 32) * 4;
            destStride       = (size_t)(bpl - width * bpp);
        }

        CVCapture::lastFrame = CVMat(height, width, cvType, (void*)data, destStride);
    }

    adjustForSL(viewportWdivH);
}
//-----------------------------------------------------------------------------
//! Does all adjustments needed for the videoTexture
/*! CVCapture::adjustForSL processes the following adjustments for all input
images no matter with what they where captured:
\n
1) Crops the input image if it doesn't match the screens aspect ratio. The
input image mostly does't fit the aspect of the output screen aspect. If the
input image is too high we crop it on top and bottom, if it is too wide we
crop it on the sides.
If viewportWdivH is negative the viewport aspect will be adapted to the video
aspect ratio. No cropping will be applied.
\n
2) Some cameras toward a face mirror the image and some do not. If a input
image should be mirrored or not is stored in CVCalibration::_isMirroredH
(H for horizontal) and CVCalibration::_isMirroredV (V for vertical).
\n
3) Many of the further processing steps are faster done on grayscale images.
We therefore create a copy that is grayscale converted.
*/
void CVCapture::adjustForSL(float viewportWdivH)
{
    PROFILE_FUNCTION();

    format = CVImage::cvType2glPixelFormat(lastFrame.type());

    //////////////////////////////////////
    // 1) Check if capture size changed //
    //////////////////////////////////////

    // Get capture size before cropping
    captureSize = lastFrame.size();

    // Determine active size index if unset or changed
    if (!camSizes.empty())
    {
        CVSize activeSize(0, 0);

        if (activeCamSizeIndex >= 0 && activeCamSizeIndex < (int)camSizes.size())
            activeSize = camSizes[(uint)activeCamSizeIndex];

        if (activeCamSizeIndex == -1 || captureSize != activeSize)
        {
            for (unsigned long i = 0; i < camSizes.size(); ++i)
            {
                if (camSizes[i] == captureSize)
                {
                    activeCamSizeIndex = (int)i;
                    break;
                }
            }
        }
    }

    //////////////////////////////////////////////////////////////////
    // 2) Crop Video image to the aspect ratio of OpenGL background //
    //////////////////////////////////////////////////////////////////

    // Cropping is done almost always.
    // So this is Android image copy loop #2

    float inWdivH = (float)lastFrame.cols / (float)lastFrame.rows;
    // viewportWdivH is negative the viewport aspect will be the same
    float outWdivH = viewportWdivH < 0.0f ? inWdivH : viewportWdivH;

    if (Utils::abs(inWdivH - outWdivH) > 0.01f)
    {
        int width  = 0; // width in pixels of the destination image
        int height = 0; // height in pixels of the destination image
        int cropH  = 0; // crop height in pixels of the source image
        int cropW  = 0; // crop width in pixels of the source image
        int wModulo4;
        int hModulo4;

        if (inWdivH > outWdivH) // crop input image left & right
        {
            width  = (int)((float)lastFrame.rows * outWdivH);
            height = lastFrame.rows;
            cropW  = (int)((float)(lastFrame.cols - width) * 0.5f);

            // Width must be devidable by 4
            wModulo4 = width % 4;
            if (wModulo4 == 1) width--;
            if (wModulo4 == 2)
            {
                cropW++;
                width -= 2;
            }
            if (wModulo4 == 3) width++;
        }
        else // crop input image at top & bottom
        {
            width  = lastFrame.cols;
            height = (int)((float)lastFrame.cols / outWdivH);
            cropH  = (int)((float)(lastFrame.rows - height) * 0.5f);

            // Height must be dividable by 4
            hModulo4 = height % 4;
            if (hModulo4 == 1) height--;
            if (hModulo4 == 2)
            {
                cropH++;
                height -= 2;
            }
            if (hModulo4 == 3) height++;
        }

        lastFrame(CVRect(cropW, cropH, width, height)).copyTo(lastFrame);
        // imwrite("AfterCropping.bmp", lastFrame);
    }

    //////////////////
    // 3) Mirroring //
    //////////////////

    // Mirroring is done for most selfie cameras.
    // So this is Android image copy loop #3

    if (activeCamera->calibration.isMirroredH())
    {
        CVMat mirrored;
        if (activeCamera->calibration.isMirroredV())
            cv::flip(lastFrame, mirrored, -1);
        else
            cv::flip(lastFrame, mirrored, 1);
        lastFrame = mirrored;
    }
    else if (activeCamera->calibration.isMirroredV())
    {
        CVMat mirrored;
        if (activeCamera->calibration.isMirroredH())
            cv::flip(lastFrame, mirrored, -1);
        else
            cv::flip(lastFrame, mirrored, 0);
        lastFrame = mirrored;
    }

    /////////////////////////
    // 4) Create grayscale //
    /////////////////////////

    // Creating a grayscale version from an YUV input source is stupid.
    // We just could take the Y channel.
    // Android image copy loop #4

    if (!lastFrame.empty())
        cv::cvtColor(lastFrame, lastFrameGray, cv::COLOR_BGR2GRAY);

#ifndef SL_EMSCRIPTEN
    // Reset calibrated image size
    if (lastFrame.size() != activeCamera->calibration.imageSize())
    {
        activeCamera->calibration.adaptForNewResolution(lastFrame.size(), true);
    }
#endif

    _captureTimesMS.set(_timer.elapsedTimeInMilliSec() - startCaptureTimeMS);
}
//-----------------------------------------------------------------------------
//! YUV to RGB image infos. Offset value can be negative for mirrored copy.
inline void
yuv2rbg(uchar y, uchar u, uchar v, uchar& r, uchar& g, uchar& b)
{
    // Conversion from:
    // https://de.wikipedia.org/wiki/YUV-Farbmodell
    // float c = 1.164f*(float)(yVal-16);
    // float d = (float)(uVal-128);
    // float e = (float)(vVal-128);
    // r = clipFToUInt8(c + 1.596f*e);
    // g = clipFToUInt8(c - 0.391f*d - 0.813f*e);
    // b = clipFToUInt8(c + 2.018f*d);

    // Conversion from:
    // http://www.wordsaretoys.com/2013/10/18/making-yuv-conversion-a-little-faster
    // I've multiplied each floating point constant by 1024 and truncated it.
    // Now I can add/subtract the scaled integers, and apply a bit shift right to
    // divide each result by 1024
    int e  = v - 128;
    int d  = u - 128;
    int a0 = 1192 * (y - 16);
    int a1 = 1634 * e;
    int a2 = 832 * e;
    int a3 = 400 * d;
    int a4 = 2066 * d;
    r      = (uchar)Utils::clamp((a0 + a1) >> 10, 0, 255);
    g      = (uchar)Utils::clamp((a0 - a2 - a3) >> 10, 0, 255);
    b      = (uchar)Utils::clamp((a0 + a4) >> 10, 0, 255);
}
//-----------------------------------------------------------------------------
//! YUV to RGB image infos. Offset value can be negative for mirrored copy.
struct colorBGR
{
    uchar b, g, r;
};
//-----------------------------------------------------------------------------
//! YUV to RGB image infos. Offset value can be negative for mirrored copy.
struct YUV2RGB_ImageInfo
{
    int bgrColOffest;  //!< offset in bytes to the next bgr pixel (column)
    int grayColOffest; //!< offset in bytes to the next gray pixel (column)
    int yColOffest;    //!< offset in bytes to the next y pixel (column)
    int uColOffest;    //!< offset in bytes to the next u pixel (column)
    int vColOffset;    //!< offset in bytes to the next v pixel (column)
    int bgrRowOffset;  //!< offset in bytes to the next bgr row
    int grayRowOffset; //!< offset in bytes to the next grayscale row
    int yRowOffset;    //!< offset in bytes to the y value of the next row
    int uRowOffset;    //!< offset in bytes to the u value of the next row
    int vRowOffest;    //!< offset in bytes to the v value of the next row
};
//-----------------------------------------------------------------------------
//! YUV to RGB image block infos that are different per thread
struct YUV2RGB_BlockInfo
{
    YUV2RGB_ImageInfo* imageInfo; //!< Pointer to the image info
    int                rowCount;  //!< Num. of rows in block
    int                colCount;  //!< Num. of columns in block
    uchar*             bgrRow;    //!< Pointer to the bgr row
    uchar*             grayRow;   //!< Pointer to the grayscale row
    uchar*             yRow;      //!< Pointer to the y value row
    uchar*             uRow;      //!< Pointer to the u value row
    uchar*             vRow;      //!< Pointer to the v value row
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
        colorBGR* bgrCol  = (colorBGR*)block->bgrRow;
        uchar*    grayCol = block->grayRow;
        uchar*    yCol    = block->yRow;
        uchar*    uCol    = block->uRow;
        uchar*    vCol    = block->vRow;

        // convert 2 pixels in the inner loop
        for (int col = 0; col < block->colCount; col += 2)
        {
            yuv2rbg(*yCol, *uCol, *vCol, bgrCol->r, bgrCol->g, bgrCol->b);
            *grayCol = *yCol;

            bgrCol += image->bgrColOffest;
            grayCol += image->grayColOffest;
            yCol += image->yColOffest;

            yuv2rbg(*yCol, *uCol, *vCol, bgrCol->r, bgrCol->g, bgrCol->b);
            *grayCol = *yCol;

            bgrCol += image->bgrColOffest;
            grayCol += image->grayColOffest;
            yCol += image->yColOffest;

            uCol += image->uColOffest;
            vCol += image->vColOffset;
        }

        block->bgrRow += image->bgrRowOffset;
        block->grayRow += image->grayRowOffset;
        block->yRow += image->yRowOffset;

        // if odd row
        if (row & 1)
        {
            block->uRow += image->uRowOffset;
            block->vRow += image->vRowOffest;
        }
    }

    return nullptr;
}
//------------------------------------------------------------------------------
//! Copies and converts the video image in YUV_420 format to RGB and Grayscale
/*! CVCapture::copyYUVPlanes copies and converts the video image in YUV_420
format to the RGB image in CVCapture::lastFrame and the Y channel the grayscale
image in CVCapture::lastFrameGray.\n
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
image should be mirrored or not is stored in CVCalibration::_isMirroredH
(H for horizontal) and CVCalibration::_isMirroredV (V for vertical).
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
We therefore create a copy of the y-channel into CVCapture::lastFrameGray.
\n
@param scrWdivH    aspect ratio width / height
@param srcW        Source image width in pixel
@param srcH        Source image height in pixel
@param y           Pointer to first byte of the top left pixel of the y-plane
@param yBytes      Size in bytes of the y-plane (must be srcW x srcH)
@param yColOffset  Offset in bytes to the next pixel in the y-plane
@param yRowOffset  Offset in bytes to the next line in the y-plane
@param u           Pointer to first byte of the top left pixel of the u-plane
@param uBytes      Size in bytes of the u-plane
@param uColOffset  Offset in bytes to the next pixel in the u-plane
@param uRowOffset  Offset in bytes to the next line in the u-plane
@param v           Pointer to first byte of the top left pixel of the v-plane
@param vBytes      Size in bytes of the v-plane
@param vColOffset  Offset in bytes to the next pixel in the v-plane
@param vRowOffset  Offset in bytes to the next line in the v-plane
*/
void CVCapture::copyYUVPlanes(float  scrWdivH,
                              int    srcW,
                              int    srcH,
                              uchar* y,
                              int    yBytes,
                              int    yColOffset,
                              int    yRowOffset,
                              uchar* u,
                              int    uBytes,
                              int    uColOffset,
                              int    uRowOffset,
                              uchar* v,
                              int    vBytes,
                              int    vColOffset,
                              int    vRowOffset)
{
    // Set the start time to measure the MS for the whole conversion
    CVCapture::startCaptureTimeMS = _timer.elapsedTimeInMilliSec();

    // input image aspect ratio
    float imgWdivH = (float)srcW / (float)srcH;

    int dstW  = srcW; // width in pixels of the destination image
    int dstH  = srcH; // height in pixels of the destination image
    int cropH = 0;    // crop height in pixels of the source image
    int cropW = 0;    // crop width in pixels of the source image

    // Crop image if source and destination aspect is not the same
    if (Utils::abs(imgWdivH - scrWdivH) > 0.01f)
    {
        if (imgWdivH > scrWdivH) // crop input image left & right
        {
            dstW  = (int)((float)srcH * scrWdivH);
            dstH  = srcH;
            cropW = (int)((float)(srcW - dstW) * 0.5f);
        }
        else // crop input image at top & bottom
        {
            dstW  = srcW;
            dstH  = (int)((float)srcW / scrWdivH);
            cropH = (int)((float)(srcH - dstH) * 0.5f);
        }
    }

    // Get the infos if the destination image must be mirrored
    bool mirrorH = CVCapture::activeCamera->mirrorH();
    bool mirrorV = CVCapture::activeCamera->mirrorV();

    // Create output color (BGR) and grayscale images
    lastFrame     = CVMat(dstH, dstW, CV_8UC(3));
    lastFrameGray = CVMat(dstH, dstW, CV_8UC(1));
    format        = CVImage::cvType2glPixelFormat(lastFrame.type());

    // Bugfix on some devices with wrong pixel offsets
    if (yRowOffset == uRowOffset && uColOffset == 1)
    {
        uColOffset = 2;
        vColOffset = 2;
    }

    uchar* bgrRow  = lastFrame.data;
    uchar* grayRow = lastFrameGray.data;

    int bgrColBytes  = 3;
    int bgrRowBytes  = dstW * bgrColBytes;
    int grayColBytes = 1;
    int grayRowBytes = dstW * grayColBytes;

    // Adjust the offsets depending on the horizontal mirroring
    int bgrRowOffset  = dstW * bgrColBytes;
    int grayRowOffset = dstW;
    if (mirrorH)
    {
        bgrRow += (dstH - 1) * bgrRowBytes;
        grayRow += (dstH - 1) * grayRowBytes;
        bgrRowOffset *= -1;
        grayRowOffset *= -1;
    }

    // Adjust the offsets depending on the vertical mirroring
    int bgrColOffset  = 1;
    int grayColOffset = grayColBytes;
    if (mirrorV)
    {
        bgrRow += (bgrRowBytes - bgrColBytes);
        grayRow += (grayRowBytes - grayColBytes);
        bgrColOffset *= -1;
        grayColOffset *= -1;
    }

    // Set source buffer pointers
    int    halfCropH = cropH / 2;
    int    halfCropW = cropW / 2;
    uchar* yRow      = y + cropH * yRowOffset + cropW * yColOffset;
    uchar* uRow      = u + halfCropH * uRowOffset + halfCropW * uColOffset;
    uchar* vRow      = v + halfCropH * vRowOffset + halfCropW * vColOffset;

    // Set the information common for all thread blocks
    YUV2RGB_ImageInfo imageInfo{};
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
    const int         threadNum = 4; // std::max(thread::hardware_concurrency(), 1U);
    vector<thread>    threads;
    YUV2RGB_BlockInfo threadInfos[threadNum];
    int               rowsPerThread     = dstH / (threadNum + 1);
    int               halfRowsPerThread = (int)((float)rowsPerThread * 0.5f);
    int               rowsHandled       = 0;

    // Launch threadNum-1 threads on different blocks of the image
    for (int i = 0; i < threadNum - 1; i++)
    {
        YUV2RGB_BlockInfo* info = threadInfos + i;
        info->imageInfo         = &imageInfo;
        info->bgrRow            = bgrRow;
        info->grayRow           = grayRow;
        info->yRow              = yRow;
        info->uRow              = uRow;
        info->vRow              = vRow;
        info->rowCount          = rowsPerThread;
        info->colCount          = dstW;

        ////////////////////////////////////////////////
        threads.emplace_back(thread(convertYUV2RGB, info));
        ////////////////////////////////////////////////

        rowsHandled += rowsPerThread;

        bgrRow += bgrRowOffset * rowsPerThread;
        grayRow += grayRowOffset * rowsPerThread;
        yRow += yRowOffset * rowsPerThread;
        uRow += uRowOffset * halfRowsPerThread;
        vRow += vRowOffset * halfRowsPerThread;
    }
    // Launch the last block on the main thread
    YUV2RGB_BlockInfo infoMain{};
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
    for (auto& thread : threads)
        thread.join();

    // Stop the capture time displayed in the statistics info
    _captureTimesMS.set(_timer.elapsedTimeInMilliSec() - startCaptureTimeMS);
}
//-----------------------------------------------------------------------------
//! Setter for video type also sets the active calibration
/*! The CVCapture instance has up to three video camera calibrations, one
for a main camera (CVCapture::mainCam), one for the selfie camera on
mobile devices (CVCapture::scndCam) and one for video file simulation
(CVCapture::videoFileCam). The member CVCapture::activeCamera
references the active one.
*/
void CVCapture::videoType(CVVideoType vt)
{
    _videoType = vt;
    _captureTimesMS.set(0.0f);

    if (vt == VT_SCND)
    {
        if (hasSecondaryCamera)
            activeCamera = &scndCam;
        else // fallback if there is no secondary camera we use main setup
        {
            _videoType   = VT_MAIN;
            activeCamera = &mainCam;
        }
    }
    else if (vt == VT_FILE)
        activeCamera = &videoFileCam;
    else
    {
        activeCamera = &mainCam;
        if (vt == VT_NONE)
        {
            release();
            _captureTimesMS.init(60, 0.0f);
        }
    }
}
//-----------------------------------------------------------------------------
void CVCapture::loadCalibrations(const string& computerInfo,
                                 const string& configPath)
{
    string mainCalibFilename = "camCalib_" + computerInfo + "_main.xml";
    string scndCalibFilename = "camCalib_" + computerInfo + "_scnd.xml";

    // load opencv camera calibration for main and secondary camera
#if defined(APP_USES_CVCAPTURE)

    // try to download from ftp if no calibration exists locally
    string fullPathAndFilename = Utils::unifySlashes(configPath) + mainCalibFilename;
    if (Utils::fileExists(fullPathAndFilename))
    {
        if (!mainCam.calibration.load(configPath, mainCalibFilename, true))
        {
            // instantiate a guessed calibration
            // mainCam.calibration = CVCalibration()
        }
    }
    else
    {
        /*
        //todo: move this download call out of cvcaputure (during refactoring of this class)
        string errorMsg;
        if (!FtpUtils::downloadFileLatestVersion(AppDemo::calibFilePath,
                                              mainCalibFilename,
                                              AppDemo::CALIB_FTP_HOST,
                                              AppDemo::CALIB_FTP_USER,
                                              AppDemo::CALIB_FTP_PWD,
                                              AppDemo::CALIB_FTP_DIR,
                                              errorMsg))
        {
         Utils::log("SLProject", errorMsg.c_str());
        }
        */
    }

    activeCamera       = &mainCam;
    hasSecondaryCamera = false;
#else
    /*
    //todo: move this download call out of cvcaputure (during refactoring of this class)
    string errorMsg;
    if (!FtpUtils::downloadFile(AppDemo::calibFilePath,
                                mainCalibFilename,
                                AppDemo::CALIB_FTP_HOST,
                                AppDemo::CALIB_FTP_USER,
                                AppDemo::CALIB_FTP_PWD,
                                AppDemo::CALIB_FTP_DIR,
                                errorMsg))
    {
        Utils::log("SLProject", errorMsg.c_str());
    }
    //todo: move this download call out of cvcaputure (during refactoring of this class)
    if (!FtpUtils::downloadFile(AppDemo::calibFilePath,
                                scndCalibFilename,
                                AppDemo::CALIB_FTP_HOST,
                                AppDemo::CALIB_FTP_USER,
                                AppDemo::CALIB_FTP_PWD,
                                AppDemo::CALIB_FTP_DIR,
                                errorMsg))
    {
        Utils::log("SLProject", errorMsg.c_str());
    }
    */
    mainCam.calibration.load(configPath, mainCalibFilename, true);
    scndCam.calibration.load(configPath, scndCalibFilename, true);
    activeCamera       = &mainCam;
    hasSecondaryCamera = true;
#endif
}
//-----------------------------------------------------------------------------
/*! Sets the with and height of a camera size at index sizeIndex.
If sizeIndexMax changes the vector in CVCapture gets cleared and resized.
*/
void CVCapture::setCameraSize(int sizeIndex,
                              int sizeIndexMax,
                              int width,
                              int height)
{
    if ((uint)sizeIndexMax != camSizes.size())
    {
        camSizes.clear();
        camSizes.resize((uint)sizeIndexMax);
    }
    camSizes[(uint)sizeIndex].width  = width;
    camSizes[(uint)sizeIndex].height = height;
}
//-----------------------------------------------------------------------------
//! Moves the current frame position in a video file.
void CVCapture::moveCapturePosition(int n)
{
#ifndef SL_EMSCRIPTEN
    if (_videoType != VT_FILE) return;

    int frameIndex = (int)_captureDevice.get(cv::CAP_PROP_POS_FRAMES);
    frameIndex += n;

    if (frameIndex < 0) frameIndex = 0;
    if (frameIndex > frameCount) frameIndex = frameCount;

    _captureDevice.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
#endif
}
//-----------------------------------------------------------------------------
//! Returns the next frame index number
int CVCapture::nextFrameIndex()
{
#ifndef SL_EMSCRIPTEN
    int result = 0;

    if (_videoType == VT_FILE)
        result = (int)_captureDevice.get(cv::CAP_PROP_POS_FRAMES);

    return result;
#else
    return 0;
#endif
}
//-----------------------------------------------------------------------------
int CVCapture::videoLength()
{
#ifndef SL_EMSCRIPTEN
    int result = 0;

    if (_videoType == VT_FILE)
        result = (int)_captureDevice.get(cv::CAP_PROP_FRAME_COUNT);

    return result;
#else
    return 0;
#endif
}
//------------------------------------------------------------------------------
