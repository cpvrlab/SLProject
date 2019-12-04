#include <CVCamera.h>

CVCamera::CVCamera(CVCameraType type)
  : _type(type),
    _calibration(_type)
{
}
