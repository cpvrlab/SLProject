#ifndef SENS_CAMERA_H
#define SENS_CAMERA_H

#include <opencv2/core.hpp>

struct SENSFrame
{
    SENSFrame(cv::Mat imgRGB,
              cv::Mat imgGray,
              int     captureWidth,
              int     captureHeight,
              int     cropW,
              int     cropH,
              bool    mirroredH,
              bool    mirroredV)
      : imgRGB(imgRGB),
        imgGray(imgGray),
        captureWidth(captureWidth),
        captureHeight(captureHeight),
        cropW(cropW),
        cropH(cropH),
        mirroredH(mirroredH),
        mirroredV(mirroredV)
    {
    }

    cv::Mat imgRGB;
    cv::Mat imgGray;

    const int  captureWidth;
    const int  captureHeight;
    const int  cropW;
    const int  cropH;
    const bool mirroredH;
    const bool mirroredV;
};
typedef std::shared_ptr<SENSFrame> SENSFramePtr;

class SENSCamera
{
public:
    enum class Facing
    {
        FRONT = 0,
        BACK
    };

    enum class FocusMode
    {
        CONTINIOUS_AUTO_FOCUS = 0,
        FIXED_INFINITY_FOCUS
    };

    struct Config
    {
        FocusMode focusMode = FocusMode::CONTINIOUS_AUTO_FOCUS;
        //! largest target image width (only RGB)
        int targetWidth = 0;
        //! largest target image width (only RGB)
        int targetHeight = 0;
        //! width of smaller image version (e.g. for tracking)
        int smallWidth = 0;
        //! height of smaller image version (e.g. for tracking)
        int smallHeight = 0;
        //! mirror image horizontally after capturing
        bool mirrorH = false;
        //! mirror image vertically after capturing
        bool mirrorV = false;
        //! provide scaled (smaller) version with size (smallWidth, smallHeight)
        bool provideScaledImage = false;
        //! provide gray version of small image
        bool convertToGray = false;
        //! adjust image in asynchronous thread
        bool adjustAsynchronously = false;
    };

    SENSCamera(SENSCamera::Facing facing)
      : _facing(facing)
    {
    }

    virtual void         start(const Config config)              = 0;
    virtual void         start(int width, int height) = 0;
    virtual void         stop(){};
    virtual SENSFramePtr getLatestFrame() = 0;

private:
    SENSCamera::Facing _facing;
};

#endif //SENS_CAMERA_H