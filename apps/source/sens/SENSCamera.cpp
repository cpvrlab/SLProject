#include "SENSCamera.h"
#include <opencv2/imgproc.hpp>
#include <sens/SENSUtils.h>

bool isEqualToOne(float value)
{
    return std::abs(1.f - value) <= 0.00001f;
}

//searches for best machting size and returns it
int SENSCameraCharacteristics::findBestMatchingConfig(cv::Size requiredSize) const
{
    if (_streamConfigs.size() == 0)
        throw SENSException(SENSType::CAM, "No stream configuration available!", __LINE__, __FILE__);

    std::vector<std::pair<float, int>> matchingSizes;

    //calculate score for
    for (int i = 0; i < _streamConfigs.size(); ++i)
    {
        const StreamConfig& config = _streamConfigs[i];
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

SENSFramePtr SENSCameraBase::postProcessNewFrame(cv::Mat& rgbImg)
{
    cv::Size inputSize = rgbImg.size();

    // Crop Video image to required aspect ratio
    int cropW = 0, cropH = 0;
    SENS::cropImage(rgbImg, (float)_config.targetWidth / (float)_config.targetHeight, cropW, cropH);

    // Mirroring
    //(copy here because we use no mutex to save config)
    bool mirrorH = _config.mirrorH;
    bool mirrorV = _config.mirrorV;
    SENS::mirrorImage(rgbImg, mirrorH, mirrorV);

    cv::Mat manipImg;
    if (_config.provideScaledImage)
    {
        manipImg  = rgbImg;
        int cropW = 0, cropH = 0;
        SENS::cropImage(manipImg, (float)_config.manipWidth / (float)_config.manipHeight, cropW, cropH);
        float scale = (float)manipImg.size().width / (float)_config.manipWidth;
        cv::resize(manipImg, manipImg, cv::Size(), scale, scale);
    }
    else if (_config.convertManipToGray)
    {
        manipImg = rgbImg;
    }

    // Create grayscale
    cv::Mat grayImg;
    if (_config.convertManipToGray)
    {
        cv::cvtColor(rgbImg, grayImg, cv::COLOR_BGR2GRAY);
    }

    SENSFramePtr sensFrame = std::make_shared<SENSFrame>(rgbImg, grayImg, inputSize.width, inputSize.height, cropW, cropH, mirrorH, mirrorV);
    return sensFrame;
}

std::pair<const SENSCameraCharacteristics* const, int> SENSCaptureProperties::findBestMatchingConfig(SENSCameraFacing facing, float horizFov, int width, int height) const
{
    struct SortElem
    {
        //corresponding camera characteristics
        const SENSCameraCharacteristics* camChars;
        //corresponding index in stream configurations
        int streamConfigIdx;
        //cropped width (using stream configuration referenced by streamConfigIdx)
        int widthCropped;
        //cropped height (using stream configuration referenced by streamConfigIdx)
        int heightCropped;
        //horizontal field of view using cropped width
        float horizFov;
        //scale factor between stream config and target resolutions
        float scale;

        //difference to target field of view
        float fovScore;
        //summed crop (used as later as score to make decision)
        int cropScore;
        //score that respecs necessary scale complexity
        int scaleScore;
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
            const SENSCameraCharacteristics::StreamConfig& config               = *itStream;
            float                                          targetWidthDivHeight = (float)width / (float)height;
            /*
            if ((targetWidthDivHeight) > ((float)config.widthPix / (float)config.heightPix))
            {
                sortElem.scale  = (float)width / (float)config.widthPix;
                float heightScaled = config.heightPix * sortElem.scale;
                sortElem.cropScore += heightScaled - height;
                sortElem.cropScore += (float)config.widthPix * sortElem.scale - (float)width;
            }
            else
            {
                sortElem.scale = (float)height / (float)config.heightPix;
                float widthScaled = config.widthPix * sortElem.scale;
                sortElem.cropScore += widthScaled - width;
                sortElem.cropScore += (float)config.heightPix * sortElem.scale - (float)height;
            }
             */

            SortElem sortElem;
            int      cropW, cropH;
            SENS::calcCrop({config.widthPix, config.heightPix}, targetWidthDivHeight, cropW, cropH, sortElem.widthCropped, sortElem.heightCropped);
            sortElem.scale = (float)width / (float)sortElem.widthCropped;

            if (sortElem.scale <= 1.f)
            {
                sortElem.camChars        = &*itChars;
                sortElem.streamConfigIdx = (int)(&*itStream - &*streams.begin());
                //we have to use the cropped and unscaled width of the stream config because of the resolution of the focalLengthPix
                sortElem.horizFov = SENS::calcFOVDegFromFocalLengthPix(config.focalLengthPix, sortElem.widthCropped);

                sortElem.fovScore  = std::abs(sortElem.horizFov - horizFov);
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

    //sort
    std::sort(sortElems.begin(), sortElems.end(), [](const SortElem& lhs, const SortElem& rhs) -> bool { return lhs.fovScore < rhs.fovScore; });

    int i = 0;
    for (auto it : sortElems)
    {
        std::stringstream ss;
        ss << "Sort Elem " << i;
        ss << " fov: " << it.horizFov;
        ss << " fov score: " << it.fovScore;
        ss << " widthCropped: " << it.widthCropped;
        ss << " heightCropped: " << it.heightCropped;
        ss << " scale: " << it.scale;
        std::cout << ss.str() << std::endl;
        i++;
    }

    if (sortElems.size())
    {
        const SortElem& bestSortElem = sortElems.front();
        return std::make_pair(bestSortElem.camChars, bestSortElem.streamConfigIdx);
    }
    else
    {
        return std::make_pair(nullptr, -1);
    }
}
