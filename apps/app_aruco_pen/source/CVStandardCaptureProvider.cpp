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

    // We should use CAP_DSHOW as the preferred API to prevent
    // a memory leak when releasing the device if the
    // MSMF backend is used
    // (https://github.com/opencv/opencv-python/issues/198)
    // (https://stackoverflow.com/questions/53888878/cv2-warn0-terminating-async-callback-when-attempting-to-take-a-picture)
    _captureDevice.open(_deviceIndex);
    _captureDevice.set(cv::CAP_PROP_FRAME_WIDTH, captureSize().width);
    _captureDevice.set(cv::CAP_PROP_FRAME_HEIGHT, captureSize().height);
}

void CVStandardCaptureProvider::grab()
{
    if (!isOpened())
    {
        return;
    }

    _captureDevice.read(_lastFrameBGR);
    cv::cvtColor(_lastFrameBGR, _lastFrameGray, cv::COLOR_BGR2GRAY);
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