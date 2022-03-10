//#############################################################################
//  File:      CVCaptureProviderStandard.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "CVCaptureProviderStandard.h"

#include <utility>

//-----------------------------------------------------------------------------
CVCaptureProviderStandard::CVCaptureProviderStandard(SLint deviceIndex, CVSize captureSize)
  : CVCaptureProvider("cv_camera_" + std::to_string(deviceIndex),
                      "CV Camera " + std::to_string(deviceIndex),
                      std::move(captureSize)),
    _deviceIndex(deviceIndex)
{
}
//-----------------------------------------------------------------------------
CVCaptureProviderStandard::~CVCaptureProviderStandard() noexcept
{
    if (CVCaptureProviderStandard::isOpened())
    {
        CVCaptureProviderStandard::close();
    }
}
//-----------------------------------------------------------------------------
void CVCaptureProviderStandard::open()
{
    if (isOpened())
    {
        return;
    }

    // We should use CAP_DSHOW as the preferred API to prevent
    // a memory leak when releasing the device if the MSMF backend is used
    // (https://github.com/opencv/opencv-python/issues/198)
    // (https://stackoverflow.com/questions/53888878/cv2-warn0-terminating-async-callback-when-attempting-to-take-a-picture)

    _captureDevice.setExceptionMode(true);

    try {
        _captureDevice.open(_deviceIndex);
        _captureDevice.set(cv::CAP_PROP_FRAME_WIDTH, captureSize().width);
        _captureDevice.set(cv::CAP_PROP_FRAME_HEIGHT, captureSize().height);

        _captureSize.width = (int) _captureDevice.get(cv::CAP_PROP_FRAME_WIDTH);
        _captureSize.height = (int) _captureDevice.get(cv::CAP_PROP_FRAME_HEIGHT);
    } catch (cv::Exception& e) {
        SL_LOG(e.what());
    }
}
//-----------------------------------------------------------------------------
void CVCaptureProviderStandard::grab()
{
    if (!isOpened())
    {
        return;
    }

    _captureDevice.read(_lastFrameBGR);
    cv::cvtColor(_lastFrameBGR, _lastFrameGray, cv::COLOR_BGR2GRAY);
}
//-----------------------------------------------------------------------------
void CVCaptureProviderStandard::close()
{
    if (!CVCaptureProviderStandard::isOpened())
    {
        return;
    }

    _captureDevice.release();
}
//-----------------------------------------------------------------------------
SLbool CVCaptureProviderStandard::isOpened()
{
    return _captureDevice.isOpened();
}
//-----------------------------------------------------------------------------