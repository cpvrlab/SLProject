//#############################################################################
//  File:      CVCaptureProviderSpryTrack.cpp
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <CVCaptureProviderSpryTrack.h>
#include <SpryTrackInterface.h>
#include <Instrumentor.h>

#include <utility>

//-----------------------------------------------------------------------------
CVCaptureProviderSpryTrack::CVCaptureProviderSpryTrack(CVSize captureSize)
  : CVCaptureProvider("spry_track",
                      "SpryTrack",
                      std::move(captureSize))
{
}
//-----------------------------------------------------------------------------
CVCaptureProviderSpryTrack::~CVCaptureProviderSpryTrack() noexcept
{
    if (_isOpened)
    {
        CVCaptureProviderSpryTrack::close();
    }
}
//-----------------------------------------------------------------------------
void CVCaptureProviderSpryTrack::open()
{
    if (_isOpened)
    {
        return;
    }

    _isOpened = true;
    _device   = SpryTrackInterface::instance().accessDevice();
}
//-----------------------------------------------------------------------------
void CVCaptureProviderSpryTrack::grab()
{
    PROFILE_FUNCTION();

    if (!_isOpened)
    {
        return;
    }

    int      width;
    int      height;
    uint8_t* dataGray;
    _device.acquireImage(&width, &height, &dataGray);

    // Store the BGR and the gray image
    _lastFrameGray = CVMat(height, width, CV_8UC1, dataGray, 0);
    cv::cvtColor(_lastFrameGray, _lastFrameBGR, cv::COLOR_GRAY2BGR);

    // Resize the images if needed
    if (captureSize().width != width || captureSize().height != height)
    {
        cv::resize(_lastFrameBGR, _lastFrameBGR, captureSize());
        cv::resize(_lastFrameGray, _lastFrameGray, captureSize());
    }
}
//-----------------------------------------------------------------------------
void CVCaptureProviderSpryTrack::close()
{
    if (!_isOpened)
    {
        return;
    }

    _device.close();
    _isOpened = false;
}
//-----------------------------------------------------------------------------
SLbool CVCaptureProviderSpryTrack::isOpened()
{
    return _isOpened;
}
//-----------------------------------------------------------------------------
