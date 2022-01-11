//#############################################################################
//  File:      CVCaptureProviderIDSPeak.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <CVCaptureProviderIDSPeak.h>
#include <IDSPeakInterface.h>
#include <Instrumentor.h>

#include <utility>

//-----------------------------------------------------------------------------
CVCaptureProviderIDSPeak::CVCaptureProviderIDSPeak(SLint deviceIndex, CVSize captureSize)
  : CVCaptureProvider("ids_camera_" + std::to_string(deviceIndex),
                      "IDS Camera " + std::to_string(deviceIndex),
                      std::move(captureSize)),
    _deviceIndex(deviceIndex)
{
}
//-----------------------------------------------------------------------------
CVCaptureProviderIDSPeak::~CVCaptureProviderIDSPeak() noexcept
{
    if (_isOpened)
    {
        CVCaptureProviderIDSPeak::close();
    }
}
//-----------------------------------------------------------------------------
void CVCaptureProviderIDSPeak::open()
{
    if (_isOpened)
    {
        return;
    }

    _isOpened = true;

    IDSPeakDeviceParams params;
    params.frameRate = 20.0;
    params.gain      = 1.0;
    params.gamma     = 1.05;
    params.binning   = 2;
    _device          = IDSPeakInterface::instance().openDevice(_deviceIndex, params);
}
//-----------------------------------------------------------------------------
void CVCaptureProviderIDSPeak::grab()
{
    PROFILE_FUNCTION();

    if (!_isOpened)
    {
        return;
    }

    // Acquire the frame
    IDSPeakFrame frame = _device.acquireImage();

    // Store the BGR and the gray image
    _lastFrameBGR  = CVMat(frame.height, frame.width, CV_8UC3, frame.dataBGR, 0);
    _lastFrameGray = CVMat(frame.height, frame.width, CV_8UC1, frame.dataGray, 0);

    // Resize the images if needed
    if (captureSize().width != frame.width || captureSize().height != frame.height)
    {
        cv::resize(_lastFrameBGR, _lastFrameBGR, captureSize());
        cv::resize(_lastFrameGray, _lastFrameGray, captureSize());
    }
}
//-----------------------------------------------------------------------------
void CVCaptureProviderIDSPeak::close()
{
    if (!_isOpened)
    {
        return;
    }

    _device.close();
    _isOpened = false;
}
//-----------------------------------------------------------------------------
SLbool CVCaptureProviderIDSPeak::isOpened()
{
    return _isOpened;
}
//-----------------------------------------------------------------------------
