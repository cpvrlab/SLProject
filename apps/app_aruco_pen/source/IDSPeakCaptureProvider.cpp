//#############################################################################
//  File:      IDSPeakCaptureProvider.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "IDSPeakCaptureProvider.h"

#include <apps/app_aruco_pen/source/IDSPeakInterface.h>

#include <utility>

//-----------------------------------------------------------------------------
IDSPeakCaptureProvider::IDSPeakCaptureProvider(SLint deviceIndex, CVSize captureSize)
  : CVCaptureProvider("ids_camera_" + std::to_string(deviceIndex),
                      "IDS Camera " + std::to_string(deviceIndex),
                      std::move(captureSize)),
    _deviceIndex(deviceIndex)
{
}
//-----------------------------------------------------------------------------
IDSPeakCaptureProvider::~IDSPeakCaptureProvider() noexcept
{
    if (_isOpened)
    {
        IDSPeakCaptureProvider::close();
    }
}
//-----------------------------------------------------------------------------
void IDSPeakCaptureProvider::open()
{
    if (_isOpened)
    {
        return;
    }

    _isOpened = true;

    IDSPeakInterface::init();
    IDSPeakInterface::openDevice(_deviceIndex);
    IDSPeakInterface::setDeviceParameters();
    IDSPeakInterface::allocateBuffers();
    IDSPeakInterface::startCapture();
}
//-----------------------------------------------------------------------------
void IDSPeakCaptureProvider::grab()
{
    if (!_isOpened)
    {
        return;
    }

    int      width;
    int      height;
    uint8_t* dataBGR;
    uint8_t* dataGray;
    IDSPeakInterface::captureImage(&width, &height, &dataBGR, &dataGray);

    // It's much faster if we do this ourselves instead of calling "loadIntoLastFrame"
    // We also halve the size here to save performance during tracking

    // Store the BGR image
    _lastFrameBGR = CVMat(height, width, CV_8UC3, dataBGR, 0);
    cv::resize(_lastFrameBGR, _lastFrameBGR, captureSize());

    // Store the gray image
    _lastFrameGray = CVMat(height, width, CV_8UC1, dataGray, 0);
    cv::resize(_lastFrameGray, _lastFrameGray, captureSize());
}
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
SLbool IDSPeakCaptureProvider::isOpened()
{
    return _isOpened;
}
//-----------------------------------------------------------------------------