#ifndef SENS_FRAME_H
#define SENS_FRAME_H

#include <opencv2/core.hpp>

//Camera frame obeject
struct SENSFrame
{
    SENSFrame(cv::Mat imgRGB,
              cv::Mat imgManip,
              int     captureWidth,
              int     captureHeight,
              int     cropW,
              int     cropH,
              bool    mirroredH,
              bool    mirroredV)
      : imgRGB(imgRGB),
        imgManip(imgManip),
        captureWidth(captureWidth),
        captureHeight(captureHeight),
        cropW(cropW),
        cropH(cropH),
        mirroredH(mirroredH),
        mirroredV(mirroredV)
    {
    }

    //! cropped input image
    cv::Mat imgRGB;
    //! scaled and maybe gray manipulated image
    cv::Mat imgManip;

    const int  captureWidth;
    const int  captureHeight;
    const int  cropW;
    const int  cropH;
    const bool mirroredH;
    const bool mirroredV;
};
typedef std::shared_ptr<SENSFrame> SENSFramePtr;

#endif //SENS_FRAME_H
