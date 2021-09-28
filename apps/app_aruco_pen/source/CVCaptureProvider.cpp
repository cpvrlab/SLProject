#include "CVCaptureProvider.h"

CVCaptureProvider::CVCaptureProvider(SLstring uid,
                                     SLstring name,
                                     CVSize   captureSize)
  : _uid(std::move(uid)),
    _name(std::move(name)),
    _camera(CVCameraType::FRONTFACING),
    _captureSize(captureSize)
{
}