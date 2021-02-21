#include "SENSiOSARCore.h"

SENSiOSARCore::SENSiOSARCore()
{
    _arcoreDelegate = [[SENSiOSARCoreDelegate alloc] init];
    _available      = [_arcoreDelegate isAvailable];
}

bool SENSiOSARCore::init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray)
{
    if (!_available)
        return false;

    configure(targetWidth, targetHeight, manipWidth, manipHeight, convertManipToGray);
    return true;
}

bool SENSiOSARCore::isReady()
{
    return _arcoreDelegate != nullptr;
}

bool SENSiOSARCore::resume()
{
    bool success = false;
    if (_arcoreDelegate)
        success = [_arcoreDelegate run];

    if (success)
        _pause = false;

    return success;
}

void SENSiOSARCore::reset()
{
    if (_arcoreDelegate)
        [_arcoreDelegate reset];
}

void SENSiOSARCore::pause()
{
    if (_arcoreDelegate)
        [_arcoreDelegate pause];
    _pause = true;
}

bool SENSiOSARCore::update(cv::Mat& pose)
{
    //retrieve the latest frame from arkit delegate
    cv::Mat intrinsic;
    cv::Mat imgBGR;
    bool    isTracking;
    [_arcoreDelegate latestFrame:&pose withImg:&imgBGR AndIntrinsic:&intrinsic AndImgWidth:&_inputFrameW AndImgHeight:&_inputFrameH IsTracking:&isTracking];

    if (!imgBGR.empty())
    {
        //update the internal frame
        std::lock_guard<std::mutex> lock(_frameMutex);
        _frame = std::make_unique<SENSFrameBase>(SENSClock::now(), imgBGR, intrinsic);
    }
    else
        Utils::log("SENSiOSARCore", "frame is empty!");
    return isTracking;
}
