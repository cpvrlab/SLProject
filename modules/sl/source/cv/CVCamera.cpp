#include <cv/CVCamera.h>

CVCamera::CVCamera(CVCameraType type)
  : _type(type),
    calibration(type, "")
{
}
