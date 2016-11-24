//#############################################################################
//  File:      SLCVCapture
//  Purpose:   OpenCV Capture Device
//  Author:    Marcus Hudritsch
//  Date:      June 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SLCVCapture_H
#define SLCVCapture_H

#include <stdafx.h>
#include <SLCV.h>
#include <opencv2/opencv.hpp>

//-----------------------------------------------------------------------------
//! Encapsulation of the OpenCV Capture Device
/*! It holds a static image for the last captured frame and single static
instance of the OpenCV capture device.
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
                                                 const SLbool isContinuous,
                                                 const SLbool isTopLeft);

    static  SLCVMat         lastFrame;      //!< last frame grabbed
    static  SLCVMat         lastFrameGray;  //!< last frame in grayscale
    static  SLPixelFormat   format;         //!< SL pixel format

    private:
    static  cv::VideoCapture _captureDevice; //!< OpenCV capture device
};
//-----------------------------------------------------------------------------
#endif
