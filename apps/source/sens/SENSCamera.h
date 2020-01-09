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

    SENSCamera(SENSCamera::Facing facing)
      : _facing(facing)
    {
    }

    virtual void    start(int width, int height, FocusMode focusMode) = 0;
    virtual void    stop(){};
    virtual SENSFramePtr getLatestFrame() = 0;

private:
    SENSCamera::Facing _facing;
};

#endif //SENS_CAMERA_H