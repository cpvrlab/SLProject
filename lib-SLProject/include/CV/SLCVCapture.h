//#############################################################################
//  File:      SLCVCapture
//  Purpose:   OpenCV Capture Device
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVCAPTURE_H
#define SLCVCAPTURE_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <SLCV.h>
#include <SLEnums.h>
#include <SLVec2.h>
#include <SLGLTexture.h>
#include <opencv2/opencv.hpp>

//-----------------------------------------------------------------------------
//! Encapsulation of the OpenCV Capture Device and holder of the last frame.
/*! It holds a static image for the last captured color frame and a grayscale
version as well as a single static instance of the OpenCV capture device.
\n
The live video image grabbing is not mandatory and can be replaced by the the
top level application with its own video grabbing functionality. This is e.g.
used in the iOS or Android examples. 
The SLCVCapture::lastFrame and SLCVCapture::lastFrameGray are on the other
hand used in all applications as the buffer for the last captured image.\n
Alternatively SLCVCapture can open a video file by a given videoFilename.
This feature can be used across all platforms.
For more information on video and capture see:\n
https://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html
*/
class SLCVCapture
{
    public:
    static SLVec2i      open(SLint deviceNum);
    static SLVec2i      openFile();
    static void         grabAndAdjustForSL(SLfloat outWdivH);
    static void         adjustForSL(SLfloat outWdivH);
    static SLbool       isOpened() { return _captureDevice.isOpened(); }
    static void         release();
    static void         loadIntoLastFrame(const SLint         camWidth,
                                          const SLint         camHeight,
                                          const SLPixelFormat srcPixelFormat,
                                          const SLuchar*      data,
                                          const SLbool        isContinuous);
    static void         copyYUVPlanes(int      srcW,
                                      int      srcH,
                                      SLuchar* y,
                                      int      ySize,
                                      int      yPixStride,
                                      int      yLineStride,
                                      SLuchar* u,
                                      int      uSize,
                                      int      uPixStride,
                                      int      uLineStride,
                                      SLuchar* v,
                                      int      vSize,
                                      int      vPixStride,
                                      int      vLineStride,
                                      SLfloat  dstWdivH);
    static void         videoType(SLVideoType vt);
    static SLVideoType  videoType() { return _videoType; }
    static void         init();
    static SLGLTexture* videoTexture() { return _videoTexture; }
    static SLGLTexture* videoTextureErr() { return _videoTextureErr; }
    static void         setVideoTexture();

    static SLCVMat       lastFrame;          //!< last frame grabbed in RGB
    static SLCVMat       lastFrameGray;      //!< last frame in grayscale
    static SLPixelFormat format;             //!< SL pixel format
    static SLCVSize      captureSize;        //!< size of captured frame
    static SLfloat       startCaptureTimeMS; //!< start time of capturing in ms
    static SLbool        hasSecondaryCamera; //!< flag if device has secondary camera
    static SLstring      videoDefaultPath;   //!< default path for video files
    static SLstring      videoFilename;      //!< video filename to load
    static SLbool        videoLoops;         //!< flag if video should loop
    static SLdouble      fps;

    /*! A requestedSizeIndex of -1 returns on Android the default size of 640x480.
    This is the default size index if the camera resolutions are unknown.
    */
    static SLCVVSize camSizes;           //!< All possible camera sizes
    static SLint     activeCamSizeIndex; //!< Currently active camera size index

    private:
    static cv::VideoCapture _captureDevice; //!< OpenCV capture device

    static SLVideoType  _videoType;       //!< Flag for using the live video image
    static SLGLTexture* _videoTexture;    //!< Texture for live video image
    static SLGLTexture* _videoTextureErr; //!< Texture for live video error
};
//-----------------------------------------------------------------------------
#endif // SLCVCAPTURE_H
