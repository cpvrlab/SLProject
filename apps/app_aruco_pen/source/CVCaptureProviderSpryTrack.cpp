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
constexpr int GAP_WIDTH        = 10;
constexpr int BACKGROUND_VALUE = 128;
//-----------------------------------------------------------------------------
CVCaptureProviderSpryTrack::CVCaptureProviderSpryTrack(CVSize captureSize)
  : CVCaptureProvider("spry_track",
                      "SpryTrack",
                      std::move(captureSize))
{
    _lastFrameBGR  = CVMat(_captureSize.height, _captureSize.width, CV_8UC3);
    _lastFrameGray = CVMat(_captureSize.height, _captureSize.width, CV_8UC1);
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
//    _device.enableOnboardProcessing();

    auto* marker1 = new SpryTrackMarker();
    marker1->addPoint(0.0f, 11.0f, 3.0f);
    marker1->addPoint(-15.0f, -26.0f, 3.0f);
    marker1->addPoint(16.59f, -20.95f, 3.0f);
    _device.registerMarker(marker1);
}
//-----------------------------------------------------------------------------
void CVCaptureProviderSpryTrack::grab()
{
    PROFILE_FUNCTION();

    if (!_isOpened)
    {
        return;
    }

    // Acquire the images from the spryTrack device
    SpryTrackFrame frame = _device.acquireFrame();

    // Compute the scaled image sizes and their top offset in the frame
    int   scaledWidth  = _captureSize.width / 2 - GAP_WIDTH / 2;
    float scale        = (float)scaledWidth / (float)frame.width;
    int   scaledHeight = (int)((float)frame.height * scale);
    int   top          = (int)((float)(_captureSize.height - scaledHeight) / 2.0f);

    // Convert the raw data to OpenCV matrices
    CVMat imageLeft  = CVMat(frame.height, frame.width, CV_8UC1, frame.dataGrayLeft);
    CVMat imageRight = CVMat(frame.height, frame.width, CV_8UC1, frame.dataGrayRight);

    // Resize images to fit them inside the frame
    cv::resize(imageLeft, imageLeft, CVSize(scaledWidth, scaledHeight));
    cv::resize(imageRight, imageRight, CVSize(scaledWidth, scaledHeight));

    // Fill the background with a single color
    _lastFrameGray.setTo(cv::Scalar(BACKGROUND_VALUE));

    // Copy both images to the frame
    imageLeft.copyTo(_lastFrameGray(CVRect(0,
                                           top,
                                           scaledWidth,
                                           scaledHeight)));
    imageRight.copyTo(_lastFrameGray(CVRect(_captureSize.width - scaledWidth,
                                            top,
                                            scaledWidth,
                                            scaledHeight)));

    // Convert the gray image to a BGR image
    cv::cvtColor(_lastFrameGray, _lastFrameBGR, cv::COLOR_GRAY2BGR);
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
