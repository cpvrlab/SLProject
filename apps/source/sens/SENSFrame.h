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
              bool    mirroredV,
              float   scaleToManip,
              cv::Mat intrinsics)
      : imgRGB(imgRGB),
        imgManip(imgManip),
        captureWidth(captureWidth),
        captureHeight(captureHeight),
        cropW(cropW),
        cropH(cropH),
        mirroredH(mirroredH),
        mirroredV(mirroredV),
        scaleToManip(scaleToManip),
        intrinsics(intrinsics) //!transfer by reference
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
    //! scale from imgManip to imgRGB
    const float scaleToManip;
    //! new intrinsics matrix (valid if intrinsicsChanged is true and can then be used to define a new calibration)
    cv::Mat intrinsics;
};
typedef std::unique_ptr<SENSFrame> SENSFramePtr;

#endif //SENS_FRAME_H
