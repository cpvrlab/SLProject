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

#include <stdafx.h>
#include <SLCV.h>
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
{   public:
    static  SLVec2i         open                (SLint deviceNum);
    static  SLVec2i         openFile            ();
    static  void            grabAndAdjustForSL  ();
    static  void            adjustForSL         ();
    static  SLbool          isOpened            () {return _captureDevice.isOpened();}
    static  void            release             ();
    static  void            loadIntoLastFrame   (const SLint camWidth,
                                                 const SLint camHeight,
                                                 const SLPixelFormat srcPixelFormat,
                                                 const SLuchar* data,
                                                 const SLbool isContinuous);
    static  void            copyYUVPlanes       (int srcW, int srcH,
                                                 SLuchar* y, int ySize,
                                                 int yPixStride, int yLineStride,
                                                 SLuchar* u, int uSize,
                                                 int uPixStride, int uLineStride,
                                                 SLuchar* v, int vSize,
                                                 int vPixStride, int vLineStride);

    static  SLCVMat         lastFrame;          //!< last frame grabbed
    static  SLCVMat         lastFrameGray;      //!< last frame in grayscale
    static  SLPixelFormat   format;             //!< SL pixel format
    static  SLCVSize        captureSize;        //!< size of captured frame
    static  SLfloat         startCaptureTimeMS; //!< start time of capturing in ms
    static  SLbool          hasSecondaryCamera; //!< flag if device has secondary camera
    static  SLstring        videoDefaultPath;   //!< default path for video files
    static  SLstring        videoFilename;      //!< video filename to load
    static  SLbool          videoLoops;         //!< flag if video should loop

    /*! A requestedSizeIndex of 0 returns on Android the default size of 640x480.
    If this size is not available the median element of the available sizes array is returned.
    An index of -n return the n-th smaller one. \n
    An index of +n return the n-th bigger one.\n
    This requestedSizeIndex has only an influence right now on Android.
    On desktop systen OpenCV gets the max. available resolution.
    On iOS the resolution is hardcoded to 640 x 480.*/
    static  SLint           requestedSizeIndex;

private:
    static  cv::VideoCapture _captureDevice;    //!< OpenCV capture device
};
//-----------------------------------------------------------------------------
#endif // SLCVCAPTURE_H

