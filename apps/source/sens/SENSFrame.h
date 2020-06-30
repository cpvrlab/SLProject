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
              cv::Mat intrinsics,
              bool    intrinsicsChanged)
      : imgRGB(imgRGB),
        imgManip(imgManip),
        captureWidth(captureWidth),
        captureHeight(captureHeight),
        cropW(cropW),
        cropH(cropH),
        mirroredH(mirroredH),
        mirroredV(mirroredV),
        scaleToManip(scaleToManip),
        intrinsics(intrinsics),
        intrinsicsChanged(intrinsicsChanged)
    {
        std::cout << "intrinsics scaled" << std::endl;
        std::cout << intrinsics << std::endl;
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
    //! camera intrinsics
    cv::Mat intrinsics;
    //! flags if camera intrinsics have changed between this frame and last frame (this is necessary for cameras that dynamically change their intrinsics e.g. on autofocus changes (iOS))
    const bool intrinsicsChanged;
};
typedef std::unique_ptr<SENSFrame> SENSFramePtr;

#endif //SENS_FRAME_H
