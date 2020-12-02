#ifndef SENS_FRAME_H
#define SENS_FRAME_H

#include <opencv2/core.hpp>
#include <sens/SENS.h>

//Camera frame obeject
struct SENSFrameBase
{
    SENSFrameBase(SENSTimePt timePt, cv::Mat imgBGR, cv::Mat intrinsics)
      : imgBGR(imgBGR),
        intrinsics(intrinsics),
        timePt(timePt)
    {
    }
    
    SENSFrameBase(SENSTimePt timePt, cv::Mat imgBGR, cv::Mat intrinsics, cv::Mat camPose, bool tracking)
      : imgBGR(imgBGR),
        intrinsics(intrinsics),
        timePt(timePt),
        pose(camPose),
        tracking(tracking)
    {
    }
    
    //camera pose
    cv::Mat pose;
    bool tracking = false;
    
    //! cropped input image
    cv::Mat imgBGR;
    cv::Mat intrinsics;
    
    const SENSTimePt timePt;
};

typedef std::shared_ptr<SENSFrameBase> SENSFrameBasePtr;

struct SENSFrame
{
    SENSFrame(const SENSTimePt& timePt,
              cv::Mat           imgBGR,
              cv::Mat           imgManip,
              bool              mirroredH,
              bool              mirroredV,
              float             scaleToManip,
              cv::Mat           intrinsics,
              cv::Mat           pose,
              bool              isTracking)
      : timePt(timePt),
        imgBGR(imgBGR),
        imgManip(imgManip),
        mirroredH(mirroredH),
        mirroredV(mirroredV),
        scaleToManip(scaleToManip),
        intrinsics(intrinsics),
        pose(pose),
        tracking(isTracking)
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
    
    //camera pose
    cv::Mat pose;
    bool tracking = false;
};
typedef std::shared_ptr<SENSFrame> SENSFramePtr;

#endif //SENS_FRAME_H
