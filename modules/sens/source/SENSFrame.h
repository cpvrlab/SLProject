#ifndef SENS_FRAME_H
#define SENS_FRAME_H

#include <opencv2/core.hpp>
#include <SENS.h>
#include <memory>

//Camera frame obeject
struct SENSFrameBase
{
    SENSFrameBase(SENSTimePt timePt, cv::Mat imgBGR, cv::Mat intrinsics, int width, int height)
      : imgBGR(imgBGR),
        intrinsics(intrinsics),
        width(width),
        height(height),
        timePt(timePt)
    {
    }

    cv::Mat imgBGR;
    cv::Mat intrinsics;
    int     width;
    int     height;

    const SENSTimePt timePt;
};
using SENSFrameBasePtr = std::shared_ptr<SENSFrameBase>;
//typedef std::shared_ptr<SENSFrameBase> SENSFrameBasePtr;

struct SENSFrame
{
    SENSFrame(const SENSTimePt& timePt,
              cv::Mat           imgBGR,
              cv::Mat           imgManip,
              bool              mirroredH,
              bool              mirroredV,
              float             scaleToManip,
              cv::Mat           intrinsics,
              int               width,
              int               height)
      : timePt(timePt),
        imgBGR(imgBGR),
        imgManip(imgManip),
        mirroredH(mirroredH),
        mirroredV(mirroredV),
        scaleToManip(scaleToManip),
        intrinsics(intrinsics),
        width(width),
        height(height)
    {
    }

    const SENSTimePt timePt;

    //! original image (maybe cropped and scaled)
    cv::Mat imgBGR;
    //! scaled and maybe gray manipulated image
    cv::Mat imgManip;

    //const int  captureWidth;
    //const int  captureHeight;
    //const int  cropW;
    //const int  cropH;
    const bool mirroredH;
    const bool mirroredV;
    //! scale between imgManip and imgBGR
    const float scaleToManip;

    cv::Mat intrinsics;
    int     width;
    int     height;
};
using SENSFramePtr = std::shared_ptr<SENSFrame>;
//typedef std::shared_ptr<SENSFrame> SENSFramePtr;

#endif //SENS_FRAME_H
