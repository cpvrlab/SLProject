//#############################################################################
//  File:      CVCapture
//  Purpose:   OpenCV Capture Device
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVCapture_H
#define CVCapture_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVTypedefs.h>
#include <CVImage.h>
#include <Averaged.h>
#include <opencv2/opencv.hpp>
#include <CVCamera.h>
#include <HighResTimer.h>

//-----------------------------------------------------------------------------
//! Video type if multiple exist on mobile devices
enum CVVideoType
{
    VT_NONE = 0, //!< No camera needed
    VT_MAIN = 1, //!< Main camera on all on all all devices
    VT_SCND = 2, //!< Selfie camera on mobile devices
    VT_FILE = 3, //!< Loads a video from file with OpenCV
};

//-----------------------------------------------------------------------------
//! Encapsulation of the OpenCV Capture Device and holder of the last frame.
/*! It holds a static image for the last captured color frame and a grayscale
version as well as a single static instance of the OpenCV capture device.
\n
The live video image grabbing is not mandatory and can be replaced by the the
top level application with its own video grabbing functionality. This is e.g.
used in the iOS or Android examples. 
The CVCapture::lastFrame and CVCapture::lastFrameGray are on the other
hand used in all applications as the buffer for the last captured image.\n
Alternatively CVCapture can open a video file by a given videoFilename.
This feature can be used across all platforms.
For more information on video and capture see:\n
https://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html
*/
class CVCapture
{
public: //! Public static instance getter for singleton pattern
    static CVCapture* instance()
    {
        if (!_instance)
        {
            _instance = new CVCapture();
            return _instance;
        }
        else
            return _instance;
    }

    CVSize2i open(int deviceNum);
    CVSize2i openFile();
    void     start(float viewportWdivH);
    bool     grabAndAdjustForSL(float viewportWdivH);
    void     loadIntoLastFrame(float        vieportWdivH,
                               int          camWidth,
                               int          camHeight,
                               CVPixFormat  srcPixelFormat,
                               const uchar* data,
                               bool         isContinuous);
    void     adjustForSL(float viewportWdivH);
    bool     isOpened() { return _captureDevice.isOpened(); }
    void     release();
    void     copyYUVPlanes(float  scrWdivH,
                           int    srcW,
                           int    srcH,
                           uchar* y,
                           int    ySize,
                           int    yPixStride,
                           int    yLineStride,
                           uchar* u,
                           int    uSize,
                           int    uPixStride,
                           int    uLineStride,
                           uchar* v,
                           int    vSize,
                           int    vPixStride,
                           int    vLineStride);

    void        videoType(CVVideoType vt);
    CVVideoType videoType() { return _videoType; }
    int         nextFrameIndex();
    //! get number of frames in video
    int       videoLength();
    AvgFloat& captureTimesMS() { return _captureTimesMS; }
    void      loadCalibrations(const string& computerInfo,
                               const string& configPath,
                               const string& videoPath);
    void      setCameraSize(int sizeIndex,
                            int sizeIndexMax,
                            int width,
                            int height);

    void moveCapturePosition(int n);

    CVMat       lastFrame;          //!< last frame grabbed in RGB
    CVMat       lastFrameFull;      //!< last frame grabbed in RGB and full resolution
    CVMat       lastFrameGray;      //!< last frame in grayscale
    CVPixFormat format;             //!< GL pixel format
    CVSize      captureSize;        //!< size of captured frame
    float       startCaptureTimeMS; //!< start time of capturing in ms
    bool        hasSecondaryCamera; //!< flag if device has secondary camera
    string      videoDefaultPath;   //!< default path for video files
    string      videoFilename;      //!< video filename to load
    bool        videoLoops;         //!< flag if video should loop
    float       fps;
    int         frameCount;

    /*! A requestedSizeIndex of -1 returns on Android the default size of 640x480.
    This is the default size index if the camera resolutions are unknown.*/
    CVVSize camSizes;           //!< All possible camera sizes
    int     activeCamSizeIndex; //!< Currently active camera size index

    CVCamera* activeCamera; //!< Pointer to the active camera
    CVCamera  mainCam;      //!< camera representation for main video camera
    CVCamera  scndCam;      //!< camera representation for secondary video camera
    CVCamera  videoFileCam; //!< camera representation for simulation using a video file

private:
    CVCapture(); //!< private onetime constructor
    ~CVCapture();
    static CVCapture* _instance; //!< global singleton object

    CVVideoType    _videoType;      //!< Flag for using the live video image
    CVVideoCapture _captureDevice;  //!< OpenCV capture device
    AvgFloat       _captureTimesMS; //!< Averaged time for video capturing in ms
    HighResTimer   _timer;          //!< High resolution timer
};
//-----------------------------------------------------------------------------
#endif // CVCapture_H
