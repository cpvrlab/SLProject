#ifndef WAI_MODE_H
#define WAI_MODE_H

#include <opencv2/core.hpp>

#include <WAIMath.h>
#include <WAIHelper.h>

namespace WAI
{

enum ModeType
{
    ModeType_None,
    ModeType_Aruco,
    ModeType_ORB_SLAM2,
    ModeType_ORB_SLAM2_DATA_ORIENTED
};

class WAI_API Mode
{
    public:
    Mode(ModeType type) { _type = type; }
    ModeType getType() { return _type; }
    virtual ~Mode()                     = 0;
    virtual bool getPose(cv::Mat* pose) = 0;
    virtual void notifyUpdate()         = 0;

    private:
    ModeType _type;
};
}

#endif
