//
// Created by vwm1 on 27/09/2021.
//

#include "IDSPeakCapture.h"

void IDSPeakCapture::start()
{
    _running = true;

    IDSPeakInterface::init();
    IDSPeakInterface::openDevice();
    IDSPeakInterface::setDeviceParameters();
    IDSPeakInterface::allocateBuffers();
    IDSPeakInterface::startCapture();
}

void IDSPeakCapture::grab()
{
    IDSPeakInterface::captureImage(&_width, &_height, &_dataBGR, &_dataGray);
}

void IDSPeakCapture::loadIntoCVCapture()
{
    // It's much faster if we do this ourselves instead of calling "loadIntoLastFrame"
    // We also halve the size here to save performance during tracking

    CVCapture::instance()->lastFrame = CVMat(_height, _width, CV_8UC3, _dataBGR, 0);
    cv::resize(CVCapture::instance()->lastFrame, CVCapture::instance()->lastFrame, cv::Size(_width / 2, _height / 2));
    CVCapture::instance()->lastFrameGray = CVMat(_height, _width, CV_8UC1, _dataGray, 0);
    cv::resize(CVCapture::instance()->lastFrameGray, CVCapture::instance()->lastFrameGray, cv::Size(_width / 2, _height / 2));
    CVCapture::instance()->format = PF_bgr;
    CVCapture::instance()->captureSize = CVCapture::instance()->lastFrame.size();
}

void IDSPeakCapture::stop()
{
    _running = false;

    IDSPeakInterface::stopCapture();
    IDSPeakInterface::deallocateBuffers();
    IDSPeakInterface::uninit();
}

IDSPeakCapture::IDSPeakCapture()
{
}

IDSPeakCapture::~IDSPeakCapture() noexcept
{
    if (_running)
    {
        stop();
    }
}