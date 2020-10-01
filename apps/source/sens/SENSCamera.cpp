#include "SENSCamera.h"
#include <opencv2/imgproc.hpp>
#include <sens/SENSUtils.h>
#include <Utils.h>

bool isEqualToOne(float value)
{
    return std::abs(1.f - value) <= 0.00001f;
}

//searches for best machting size and returns it
int SENSCameraDeviceProperties::findBestMatchingConfig(cv::Size requiredSize) const
{
    if (_streamConfigs.size() == 0)
        throw SENSException(SENSType::CAM, "No stream configuration available!", __LINE__, __FILE__);

    std::vector<std::pair<float, int>> matchingSizes;

    //calculate score for
    for (int i = 0; i < _streamConfigs.size(); ++i)
    {
        const SENSCameraStreamConfig& config = _streamConfigs[i];
        //stream size has to have minimum the size of the required size
        if (config.widthPix >= requiredSize.width && config.heightPix >= requiredSize.height)
        {
            float crop        = 0.f;
            float scaleFactor = 1.f;
            //calculate necessary crop to adjust stream image to required size
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
    //sort by crop
    std::sort(matchingSizes.begin(), matchingSizes.end());

    return matchingSizes.front().second;
}

SENSFramePtr SENSCameraBase::postProcessNewFrame(cv::Mat& bgrImg, cv::Mat intrinsics, bool intrinsicsChanged)
{
    //todo: accessing config readonly should be no problem  here, as the config only changes when camera is stopped
    cv::Size inputSize = bgrImg.size();

    // Crop Video image to required aspect ratio
    int cropW = 0, cropH = 0;
    SENS::cropImage(bgrImg, (float)_config.targetWidth / (float)_config.targetHeight, cropW, cropH);

    // Mirroring
    SENS::mirrorImage(bgrImg, _config.mirrorH, _config.mirrorV);

    cv::Mat manipImg;
    float   scale = 1.0f;
    if (_config.manipWidth > 0 && _config.manipHeight > 0)
    {
        manipImg  = bgrImg;
        int cropW = 0, cropH = 0;
        SENS::cropImage(manipImg, (float)_config.manipWidth / (float)_config.manipHeight, cropW, cropH);
        scale = (float)_config.manipWidth / (float)manipImg.size().width;
        cv::resize(manipImg, manipImg, cv::Size(), scale, scale);
    }
    else if (_config.convertManipToGray)
    {
        manipImg = bgrImg;
    }

    // Create grayscale
    if (_config.convertManipToGray)
    {
        cv::cvtColor(manipImg, manipImg, cv::COLOR_BGR2GRAY);
    }

    SENSFramePtr sensFrame = std::make_unique<SENSFrame>(bgrImg,
                                                         manipImg,
                                                         inputSize.width,
                                                         inputSize.height,
                                                         cropW,
                                                         cropH,
                                                         _config.mirrorH,
                                                         _config.mirrorV,
                                                         1 / scale,
                                                         intrinsics);

    return sensFrame;
}

bool SENSCaptureProperties::containsDeviceId(const std::string& deviceId) const
{
    return std::find_if(begin(), end(), [&](const SENSCameraDeviceProperties& comp) { return comp.deviceId() == deviceId; }) != end();
}

const SENSCameraDeviceProperties* SENSCaptureProperties::camPropsForDeviceId(const std::string& deviceId) const
{
    auto camPropsIt = std::find_if(begin(), end(), [&](const SENSCameraDeviceProperties& comp) { return comp.deviceId() == deviceId; });
    if (camPropsIt != end())
        return &*camPropsIt;
    else
        return nullptr;
}

std::pair<const SENSCameraDeviceProperties* const, const SENSCameraStreamConfig* const> SENSCaptureProperties::findBestMatchingConfig(SENSCameraFacing facing,
                                                                                                                                      const float      horizFov,
                                                                                                                                      const int        width,
                                                                                                                                      const int        height) const
{
    struct SortElem
    {
        //corresponding camera characteristics
        const SENSCameraDeviceProperties* camChars;
        //corresponding index in stream configurations
        const SENSCameraStreamConfig* streamConfig;
        int                           streamConfigIdx;
        //cropped width (using stream configuration referenced by streamConfigIdx)
        int widthCropped;
        //cropped height (using stream configuration referenced by streamConfigIdx)
        int heightCropped;
        //horizontal and vertical field of view using cropped width
        float horizFov = -1.f;
        //scale factor between stream config and target resolutions
        float scale;

        //difference to target field of view
        float fovScore = 0.f;
        //summed crop (used as later as score to make decision)
        int cropScore;
        //score that respecs necessary scale complexity
        int scaleScore;

        void print(int idx) const
        {
            std::stringstream ss;
            ss << "Sort Elem " << idx;
            ss << " fov: " << horizFov;
            ss << " fov score: " << fovScore;
            ss << " crop score: " << cropScore;
            ss << " scale score " << scaleScore;
            ss << " widthCropped: " << widthCropped;
            ss << " heightCropped: " << heightCropped;
            ss << " scale: " << scale;
            ss << " width orig " << camChars->streamConfigs()[streamConfigIdx].widthPix;
            ss << " height orig " << camChars->streamConfigs()[streamConfigIdx].heightPix;
            Utils::log("SENSCaptureProperties", ss.str().c_str());
        }
    };
    auto printSortElems = [](const std::vector<SortElem>& sortElems, std::string id) {
        std::cout << id << std::endl;
        for (int i = 0; i < sortElems.size(); ++i)
            sortElems.at(i).print(i);
    };

    std::vector<SortElem> sortElems;

    //make cropped versions of all stream configurations
    for (auto itChars = begin(); itChars != end(); ++itChars)
    {
        if (facing != itChars->facing())
            continue;

        const auto& streams = itChars->streamConfigs();
        for (auto itStream = streams.begin(); itStream != streams.end(); ++itStream)
        {
            const SENSCameraStreamConfig& config               = *itStream;
            float                         targetWidthDivHeight = (float)width / (float)height;

            SortElem sortElem;
            int      cropW, cropH;
            SENS::calcCrop({config.widthPix, config.heightPix}, targetWidthDivHeight, cropW, cropH, sortElem.widthCropped, sortElem.heightCropped);
            sortElem.scale = (float)width / (float)sortElem.widthCropped;

            if (sortElem.scale <= 1.f)
            {
                sortElem.camChars        = &*itChars;
                sortElem.streamConfigIdx = (int)(&*itStream - &*streams.begin());
                sortElem.streamConfig    = &*itStream;
                if (config.focalLengthPix > 0)
                {
                    //we have to use the cropped and unscaled width of the stream config because of the resolution of the focalLengthPix
                    sortElem.horizFov = SENS::calcFOVDegFromFocalLengthPix(config.focalLengthPix, sortElem.widthCropped);
                    sortElem.fovScore = std::abs(sortElem.horizFov - horizFov);
                }
                sortElem.cropScore = cropW + cropH;
                //if we have to scale add a score for it
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
        //sort by fov score
        std::sort(sortElems.begin(), sortElems.end(), [](const SortElem& lhs, const SortElem& rhs) -> bool { return lhs.fovScore < rhs.fovScore; });
        printSortElems(sortElems, "sortElems");

        //extract all in a range of +-3 degree compared to the closest to target fov and extract the one with the
        std::vector<SortElem> closeFovSortElems;
        float                 maxFovDiff = sortElems.front().fovScore + 3;
        for (SortElem elem : sortElems)
        {
            if (elem.fovScore < maxFovDiff)
                closeFovSortElems.push_back(elem);
        }

        //now extract the
        std::sort(closeFovSortElems.begin(), closeFovSortElems.end(), [](const SortElem& lhs, const SortElem& rhs) -> bool { return lhs.scale > rhs.scale; });
        printSortElems(closeFovSortElems, "closeFovSortElems");

        const SortElem& bestSortElem = closeFovSortElems.front();
        return std::make_pair(bestSortElem.camChars, bestSortElem.streamConfig);
    }
    else
    {
        return std::make_pair(nullptr, nullptr);
    }
}

void SENSCameraBase::initCalibration(float fovDegFallbackGuess)
{
    //We make a calibration with full resolution and adjust it to the manipulated image size later if neccessary:
    //For the initial setup we have to use streamconfig values
    float horizFOVDev = fovDegFallbackGuess;
    if (_config.streamConfig.focalLengthPix > 0)
        horizFOVDev = SENS::calcFOVDegFromFocalLengthPix(_config.streamConfig.focalLengthPix, _config.streamConfig.widthPix);
    _calibration = std::make_unique<SENSCalibration>(cv::Size(_config.streamConfig.widthPix, _config.streamConfig.heightPix),
                                                     horizFOVDev,
                                                     false,
                                                     false,
                                                     SENSCameraType::BACKFACING,
                                                     Utils::ComputerInfos().get());
    //now we adapt the calibration to the target size
    if (_config.targetWidth != _config.streamConfig.widthPix || _config.targetHeight != _config.streamConfig.heightPix)
        _calibration->adaptForNewResolution({_config.targetWidth, _config.targetHeight}, false);
    
    //inform listeners about calibration
    {
        std::lock_guard<std::mutex> lock(_listenerMutex);
        for(SENSCameraListener* l : _listeners)
            l->onCalibrationChanged(*_calibration);
    }
}

void SENSCameraBase::setCalibration(SENSCalibration calibration, bool buildUndistortionMaps)
{
    if (!_started)
        throw SENSException(SENSType::CAM, "setCalibration not possible if camera is not started!", __LINE__, __FILE__);

    _calibration = std::make_unique<SENSCalibration>(calibration);
    //now we adapt the calibration to the target size
    _calibration->adaptForNewResolution({_config.targetWidth, _config.targetHeight}, buildUndistortionMaps);
    /*
    if ((_config.manipWidth > 0 && _config.manipHeight > 0) || _config.manipWidth != _config.streamConfig->widthPix || _config.manipHeight != _config.streamConfig->heightPix)
        _calibration->adaptForNewResolution({_config.manipWidth, _config.manipHeight}, buildUndistortionMaps);
    else if (_config.targetWidth != _config.streamConfig->widthPix || _config.targetHeight != _config.streamConfig->heightPix)
        _calibration->adaptForNewResolution({_config.targetWidth, _config.targetHeight}, buildUndistortionMaps);
     */
    
    {
        std::lock_guard<std::mutex> lock(_listenerMutex);
        for(SENSCameraListener* l : _listeners)
            l->onCalibrationChanged(*_calibration);
    }
}

void SENSCameraBase::registerListener(SENSCameraListener* listener)
{
    std::lock_guard<std::mutex> lock(_listenerMutex);
    if (std::find(_listeners.begin(), _listeners.end(), listener) == _listeners.end())
        _listeners.push_back(listener);
}

void SENSCameraBase::unregisterListener(SENSCameraListener* listener)
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
