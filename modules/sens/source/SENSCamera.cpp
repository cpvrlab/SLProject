#include "SENSCamera.h"
#include <opencv2/imgproc.hpp>
#include <SENSUtils.h>
#include <Utils.h>

//-----------------------------------------------------------------------------
bool isEqualToOne(float value)
{
    return std::abs(1.f - value) <= 0.00001f;
}
//-----------------------------------------------------------------------------
// searches for best machting size and returns it
int SENSCameraDeviceProps::findBestMatchingConfig(cv::Size requiredSize) const
{
    if (_streamConfigs.size() == 0)
        throw SENSException(SENSType::CAM,
                            "No stream configuration available!",
                            __LINE__,
                            __FILE__);

    std::vector<std::pair<float, int>> matchingSizes;

    // calculate score for
    for (int i = 0; i < _streamConfigs.size(); ++i)
    {
        const SENSCameraStreamConfig& config = _streamConfigs[i];
        // stream size has to have minimum the size of the required size
        if (config.widthPix >= requiredSize.width &&
            config.heightPix >= requiredSize.height)
        {
            float crop        = 0.f;
            float scaleFactor = 1.f;
            // calculate necessary crop to adjust stream image to required size
            if (((float)requiredSize.width / (float)requiredSize.height) > ((float)config.widthPix / (float)config.heightPix))
            {
                scaleFactor        = (float)requiredSize.width / (float)config.widthPix;
                float heightScaled = config.heightPix * scaleFactor;
                crop += heightScaled - requiredSize.height;
                crop += (float)config.widthPix * scaleFactor - (float)requiredSize.width;
            }
            else
            {
                scaleFactor       = (float)requiredSize.height / (float)config.heightPix;
                float widthScaled = config.widthPix * scaleFactor;
                crop += widthScaled - requiredSize.width;
                crop += (float)config.heightPix * scaleFactor - (float)requiredSize.height;
            }

            float cropScaleScore = crop;
            if (!isEqualToOne(scaleFactor))
                cropScaleScore += (config.widthPix * config.heightPix);
            matchingSizes.push_back(std::make_pair(cropScaleScore, i));
        }
    }
    // sort by crop
    std::sort(matchingSizes.begin(), matchingSizes.end());

    return matchingSizes.front().second;
}
//-----------------------------------------------------------------------------
bool SENSCaptureProps::containsDeviceId(const std::string& deviceId) const
{
    return std::find_if(begin(), end(), [&](const SENSCameraDeviceProps& comp)
                        { return comp.deviceId() == deviceId; }) != end();
}
//-----------------------------------------------------------------------------
bool SENSCaptureProps::supportsCameraFacing(const SENSCameraFacing& facing) const
{
    return std::find_if(begin(), end(), [&](const SENSCameraDeviceProps& comp)
                        { return comp.facing() == facing; }) != end();
}
//-----------------------------------------------------------------------------
const SENSCameraDeviceProps* SENSCaptureProps::camPropsForDeviceId(const std::string& deviceId) const
{
    auto camPropsIt = std::find_if(begin(), end(), [&](const SENSCameraDeviceProps& comp)
                                   { return comp.deviceId() == deviceId; });
    if (camPropsIt != end())
        return &*camPropsIt;
    else
        return nullptr;
}
//-----------------------------------------------------------------------------
//! Helper function to find a camera device and stream configuration with certain characteristics.
/*! Returns a pair on nullptrs if characteristics were not fulfilled.
 * Possible reasons: the facing is not available or if the transferred width
 * and height would require an extrapolation.
 */
std::pair<const SENSCameraDeviceProps* const, const SENSCameraStreamConfig* const>
SENSCaptureProps::findBestMatchingConfig(SENSCameraFacing facing,
                                         const float      horizFov,
                                         const int        width,
                                         const int        height) const
{
    struct SortElem
    {
        // corresponding camera characteristics
        const SENSCameraDeviceProps* camChars;
        // corresponding index in stream configurations
        const SENSCameraStreamConfig* streamConfig;
        int                           streamConfigIdx;
        // cropped width (using stream configuration referenced by streamConfigIdx)
        int widthCropped;
        // cropped height (using stream configuration referenced by streamConfigIdx)
        int heightCropped;
        // horizontal and vertical field of view using cropped width
        float horizFov = -1.f;
        // scale factor between stream config and target resolutions
        float scale;

        // difference to target field of view
        float fovScore = 0.f;
        // summed crop (used as later as score to make decision)
        int cropScore;
        // score that respecs necessary scale complexity
        int scaleScore;

        void print(int idx) const
        {
            std::stringstream ss;
            ss << "Sort Elem " << idx;
            ss << " fovV: " << horizFov;
            ss << " fovV score: " << fovScore;
            ss << " crop score: " << cropScore;
            ss << " scale score " << scaleScore;
            ss << " widthCropped: " << widthCropped;
            ss << " heightCropped: " << heightCropped;
            ss << " scale: " << scale;
            ss << " width orig " << camChars->streamConfigs()[streamConfigIdx].widthPix;
            ss << " height orig " << camChars->streamConfigs()[streamConfigIdx].heightPix;
            Utils::log("SENSCaptureProps", ss.str().c_str());
        }
    };
    auto printSortElems = [](const std::vector<SortElem>& sortElems, std::string id)
    {
        std::cout << id << std::endl;
        for (int i = 0; i < sortElems.size(); ++i)
            sortElems.at(i).print(i);
    };

    std::vector<SortElem> sortElems;

    // make cropped versions of all stream configurations
    for (auto itChars = begin(); itChars != end(); ++itChars)
    {
        if (facing != itChars->facing())
            continue;

        const auto& streams              = itChars->streamConfigs();
        float       targetWidthDivHeight = (float)width / (float)height;

        for (auto itStream = streams.begin(); itStream != streams.end(); ++itStream)
        {
            const SENSCameraStreamConfig& config = *itStream;

            SortElem sortElem;
            int      cropW, cropH;
            SENS::calcCrop({config.widthPix,
                            config.heightPix},
                           targetWidthDivHeight,
                           cropW,
                           cropH,
                           sortElem.widthCropped,
                           sortElem.heightCropped);
            sortElem.scale = (float)width / (float)sortElem.widthCropped;

            if (sortElem.scale <= 1.f)
            {
                sortElem.camChars        = &*itChars;
                sortElem.streamConfigIdx = (int)(&*itStream - &*streams.begin());
                sortElem.streamConfig    = &*itStream;
                if (config.focalLengthPix > 0)
                {
                    // we have to use the cropped and unscaled width of the stream config because of the resolution of the focalLengthPix
                    sortElem.horizFov = SENS::calcFOVDegFromFocalLengthPix(config.focalLengthPix,
                                                                           sortElem.widthCropped);
                    sortElem.fovScore = std::abs(sortElem.horizFov - horizFov);
                }
                sortElem.cropScore = cropW + cropH;
                // if we have to scale add a score for it
                if (!isEqualToOne(sortElem.scale))
                    sortElem.scaleScore = width * height;
                else
                    sortElem.scaleScore = 0.f;

                sortElems.push_back(sortElem);
            }
        }
    }

    if (sortElems.size())
    {
        // sort by difference to target fovV
        std::sort(sortElems.begin(), sortElems.end(), [](const SortElem& lhs, const SortElem& rhs) -> bool
                  { return lhs.fovScore < rhs.fovScore; });
        // printSortElems(sortElems, "sortElems");

        // extract all in a range of +-3 degree compared to the closest to target fovV
        std::vector<SortElem> closeFovSortElems;
        float                 maxFovDiff = sortElems.front().fovScore + 3;
        for (SortElem elem : sortElems)
        {
            if (elem.fovScore < maxFovDiff)
                closeFovSortElems.push_back(elem);
        }

        // now extract the one with the best scale score from the remaining
        std::sort(closeFovSortElems.begin(), closeFovSortElems.end(), [](const SortElem& lhs, const SortElem& rhs) -> bool
                  { return lhs.scale > rhs.scale; });
        // printSortElems(closeFovSortElems, "closeFovSortElems");

        const SortElem& bestSortElem = closeFovSortElems.front();
        return std::make_pair(bestSortElem.camChars, bestSortElem.streamConfig);
    }
    else
    {
        return std::make_pair(nullptr, nullptr);
    }
}
//-----------------------------------------------------------------------------
void SENSBaseCamera::updateFrame(cv::Mat bgrImg,
                                 cv::Mat intrinsics,
                                 int     width,
                                 int     height,
                                 bool    intrinsicsChanged)
{
    // estimate time before running into lock
    SENSTimePt timePt = SENSClock::now();

    // inform listeners
    {
        std::lock_guard<std::mutex> lock(_listenerMutex);
        if (_listeners.size())
        {
            for (SENSCameraListener* l : _listeners)
                l->onFrame(timePt, bgrImg.clone());
        }
    }

    {
        std::lock_guard<std::mutex> lock(_frameMutex);
        _frame = std::make_shared<SENSFrameBase>(timePt,
                                                 bgrImg,
                                                 intrinsics,
                                                 width,
                                                 height);
    }
}
//-----------------------------------------------------------------------------
void SENSBaseCamera::registerListener(SENSCameraListener* listener)
{
    std::lock_guard<std::mutex> lock(_listenerMutex);
    if (std::find(_listeners.begin(), _listeners.end(), listener) == _listeners.end())
        _listeners.push_back(listener);

    if (_started)
    {
        // inform listeners about camera infos
        for (SENSCameraListener* l : _listeners)
            l->onCameraConfigChanged(_config);
    }
}
//-----------------------------------------------------------------------------
void SENSBaseCamera::unregisterListener(SENSCameraListener* listener)
{
    std::lock_guard<std::mutex> lock(_listenerMutex);
    for (auto it = _listeners.begin(); it != _listeners.end(); ++it)
    {
        if (*it == listener)
        {
            _listeners.erase(it);
            break;
        }
    }
}
//-----------------------------------------------------------------------------
SENSFrameBasePtr SENSBaseCamera::latestFrame()
{
    SENSFrameBasePtr latestFrame;

    if (!_started)
    {
        SENS_WARN("SENSBaseCamera latestFrame: Camera is not started!");
        return latestFrame;
    }

    {
        std::lock_guard<std::mutex> lock(_frameMutex);
        latestFrame = _frame;
    }

    return latestFrame;
}
//-----------------------------------------------------------------------------
void SENSBaseCamera::processStart()
{
    {
        std::lock_guard<std::mutex> lock(_listenerMutex);
        // inform listeners about camera infos
        for (SENSCameraListener* l : _listeners)
            l->onCameraConfigChanged(_config);
    }
}
//-----------------------------------------------------------------------------