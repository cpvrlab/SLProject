#include "CVStandardCaptureProvider.h"

CVStandardCaptureProvider::CVStandardCaptureProvider(SLint deviceIndex, CVSize captureSize)
  : CVCaptureProvider("cv_camera_" + std::to_string(deviceIndex),
                      "CV Camera " + std::to_string(deviceIndex),
                      captureSize),
    _deviceIndex(deviceIndex)
{
}

CVStandardCaptureProvider::~CVStandardCaptureProvider() noexcept
{
    if (_isOpened)
    {
        CVStandardCaptureProvider::close();
    }
}

void CVStandardCaptureProvider::open()
{
    if (_isOpened)
    {
        return;
    }

    // We should use CAP_DSHOW as the preferred API to prevent
    // a memory leak when releasing the device if the MSMF backend is used
    // (https://github.com/opencv/opencv-python/issues/198)
    // (https://stackoverflow.com/questions/53888878/cv2-warn0-terminating-async-callback-when-attempting-to-take-a-picture)
    // TODO FIXME HACK HACK HACK DON'T DO THIS
    alloca(128);
    _captureDevice.open(_deviceIndex);

    _isOpened = _captureDevice.isOpened();

    _captureDevice.set(cv::CAP_PROP_FRAME_WIDTH, captureSize().width);
    _captureDevice.set(cv::CAP_PROP_FRAME_HEIGHT, captureSize().height);
}

void CVStandardCaptureProvider::grab()
{
    if (!_isOpened)
    {
        return;
    }

    _captureDevice.read(_lastFrameBGR);
    cv::cvtColor(_lastFrameBGR, _lastFrameGray, cv::COLOR_BGR2GRAY);
}

void CVStandardCaptureProvider::close()
{
    if (!CVStandardCaptureProvider::isOpened())
    {
        return;
    }

    _captureDevice.release();
}

SLbool CVStandardCaptureProvider::isOpened()
{
    return _isOpened;
}