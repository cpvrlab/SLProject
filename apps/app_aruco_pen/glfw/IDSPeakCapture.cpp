//
// Created by vwm1 on 27/09/2021.
//

#include "IDSPeakCapture.h"

void IDSPeakCapture::start()
{
    running = true;

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

void IDSPeakCapture::stop()
{
    running = false;

    IDSPeakInterface::stopCapture();
    IDSPeakInterface::deallocateBuffers();
    IDSPeakInterface::uninit();
}

IDSPeakCapture::IDSPeakCapture()
{
}

IDSPeakCapture::~IDSPeakCapture() noexcept
{
    if (running)
    {
        stop();
    }
}