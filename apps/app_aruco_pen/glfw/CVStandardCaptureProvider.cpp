#include "CVStandardCaptureProvider.h"

CVStandardCaptureProvider::CVStandardCaptureProvider()
  : CVCaptureProvider("CV Camera"),
    _camera(CVCameraType::FRONTFACING)
{
}

CVStandardCaptureProvider::~CVStandardCaptureProvider() noexcept
{
    if (CVStandardCaptureProvider::isOpened())
    {
        CVStandardCaptureProvider::close();
    }
}

void CVStandardCaptureProvider::open()
{
    if (isOpened())
    {
        return;
    }

    _captureDevice.open(0);
}

void CVStandardCaptureProvider::grab()
{
    _captureDevice.read(_lastFrameBGR);
    cv::cvtColor(_lastFrameBGR, _lastFrameGray, cv::COLOR_BGR2GRAY);
    _captureSize = _lastFrameBGR.size();
}

void CVStandardCaptureProvider::close()
{
    if (!isOpened())
    {
        return;
    }

    _captureDevice.release();
}

SLbool CVStandardCaptureProvider::isOpened()
{
    return _captureDevice.isOpened();
}