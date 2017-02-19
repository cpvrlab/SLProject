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
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/

#include <stdafx.h>
#include <SLCV.h>
#include <opencv2/opencv.hpp>

//-----------------------------------------------------------------------------
//! Encapsulation of the OpenCV Capture Device
/*! It holds a static image for the last captured color frame and a grayscale
version as well as a single static instance of the OpenCV capture device. The
The live video image grabbing is not mandatory and can be replaced by the the
top level application with its own video grabbing functionality. This is e.g.
used in the iOS or Android examples. 
The SLCVCapture::lastFrame and SLCVCapture::lastFrameGray are on the other
hand used in all applications as the buffer for the last captured image.
*/
class SLCVCapture
{   public:
    static  SLVec2i         open                (SLint deviceNum);
    static  void            grabAndAdjustForSL  ();
    static  void            adjustForSL         ();
    static  SLbool          isOpened            () {return _captureDevice.isOpened();}
    static  void            release             () {_captureDevice.release();}
    static  void            loadIntoLastFrame   (const SLint camWidth,
                                                 const SLint camHeight,
                                                 const SLPixelFormat srcPixelFormat,
                                                 const SLuchar* data,
                                                 const SLbool isContinuous);

    static  SLCVMat         lastFrame;          //!< last frame grabbed
    static  SLCVMat         lastFrameGray;      //!< last frame in grayscale
    static  SLPixelFormat   format;             //!< SL pixel format
    static  SLCVSize        captureSize;        //!< size of captured fram
    static  SLfloat         startCaptureTimeMS; //!< start time of capturing in ms
    static  SLbool          hasSecondaryCamera; //!< flag if device has secondary camera

    private:
    static  cv::VideoCapture _captureDevice;    //!< OpenCV capture device
};
//-----------------------------------------------------------------------------
#endif // SLCVCAPTURE_H

