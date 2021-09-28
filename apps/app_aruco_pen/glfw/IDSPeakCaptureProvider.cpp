#include "IDSPeakCaptureProvider.h"

#include <IDSPeakInterface.h>

IDSPeakCaptureProvider::IDSPeakCaptureProvider()
  : CVCaptureProvider("IDS Camera")
{
}

IDSPeakCaptureProvider::~IDSPeakCaptureProvider() noexcept
{
    if (IDSPeakCaptureProvider::isOpened())
    {
        IDSPeakCaptureProvider::close();
    }
}

void IDSPeakCaptureProvider::open()
{
    if (_isOpened)
    {
        return;
    }

    _isOpened = true;

    IDSPeakInterface::init();
    IDSPeakInterface::openDevice();
    IDSPeakInterface::setDeviceParameters();
    IDSPeakInterface::allocateBuffers();
    IDSPeakInterface::startCapture();
}

void IDSPeakCaptureProvider::grab()
{
    int width;
    int height;
    uint8_t* dataBGR;
    uint8_t* dataGray;
    IDSPeakInterface::captureImage(&width, &height, &dataBGR, &dataGray);

    // It's much faster if we do this ourselves instead of calling "loadIntoLastFrame"
    // We also halve the size here to save performance during tracking

    // Store the BGR image
    _lastFrameBGR = CVMat(height, width, CV_8UC3, dataBGR, 0);
    cv::resize(_lastFrameBGR, _lastFrameBGR, cv::Size(width / 2, height / 2));

    // Store the gray image
    _lastFrameGray = CVMat(height, width, CV_8UC1, dataGray, 0);
    cv::resize(_lastFrameGray, _lastFrameGray, cv::Size(width / 2, height / 2));

    _captureSize = _lastFrameBGR.size();
}

void IDSPeakCaptureProvider::close()
{
    if (!_isOpened)
    {
        return;
    }

    IDSPeakInterface::stopCapture();
    IDSPeakInterface::deallocateBuffers();
    IDSPeakInterface::uninit();

    _isOpened = false;
}

SLbool IDSPeakCaptureProvider::isOpened()
{
    return _isOpened;
}