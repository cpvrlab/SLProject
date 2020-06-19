#include "SENSCamera.h"
#include <opencv2/imgproc.hpp>

//searches for best machting size and returns it
const SENSCameraStreamConfigs::Config& SENSCameraStreamConfigs::findBestMatchingConfig(cv::Size requiredSize) const
{
    if (_streamConfigs.size() == 0)
        throw SENSException(SENSType::CAM, "No stream configuration available!", __LINE__, __FILE__);

    std::vector<std::pair<float, int>> matchingSizes;

    //calculate score for
    for (int i = 0; i < _streamConfigs.size(); ++i)
    {
        const Config& config = _streamConfigs[i];
        //stream size has to have minimum the size of the required size
        if (config.widthPix >= requiredSize.width && config.heightPix >= requiredSize.height)
        {
            float crop = 0.f;
            //calculate necessary crop to adjust stream image to required size
            if (((float)requiredSize.width / (float)requiredSize.height) > ((float)config.widthPix / (float)config.heightPix))
            {
                float scaleFactor  = (float)requiredSize.width / (float)config.widthPix;
                float heightScaled = config.heightPix * scaleFactor;
                crop += heightScaled - requiredSize.height;
                crop += (float)config.widthPix - (float)requiredSize.width;
            }
            else
            {
                float scaleFactor = (float)requiredSize.height / (float)config.heightPix;
                float widthScaled = config.widthPix * scaleFactor;
                crop += widthScaled - requiredSize.width;
                crop += (float)config.heightPix - (float)requiredSize.height;
            }

            float cropScaleScore = crop;
            matchingSizes.push_back(std::make_pair(cropScaleScore, i));
        }
    }
    //sort by crop
    std::sort(matchingSizes.begin(), matchingSizes.end());

    return _streamConfigs[matchingSizes.front().second];
}

SENSFramePtr SENSCameraBase::postProcessNewFrame(cv::Mat& rgbImg)
{
    cv::Size inputSize = rgbImg.size();

    // Crop Video image to required aspect ratio
    int cropW = 0, cropH = 0;
    SENS::cropImage(rgbImg,(float)_config.targetWidth / (float)_config.targetHeight, cropW, cropH);

    // Mirroring
    //(copy here because we use no mutex to save config)
    bool mirrorH = _config.mirrorH;
    bool mirrorV = _config.mirrorV;
    SENS::mirrorImage(rgbImg, mirrorH, mirrorV);

    cv::Mat manipImg;
    if(_config.provideScaledImage)
    {
        manipImg = rgbImg;
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
