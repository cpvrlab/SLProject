#include "SENSCamera.h"

//searches for best machting size and returns it
cv::Size SENSCameraStreamConfigs::findBestMatchingSize(cv::Size requiredSize)
{
    if (_streamSizes.size() == 0)
        throw SENSException(SENSType::CAM, "No stream configuration available!", __LINE__, __FILE__);

    std::vector<std::pair<float, int>> matchingSizes;

    //calculate score for
    for (int i = 0; i < _streamSizes.size(); ++i)
    {
        //stream size has to have minimum the size of the required size
        if (_streamSizes[i].width >= requiredSize.width && _streamSizes[i].height >= requiredSize.height)
        {
            float crop = 0.f;
            //calculate necessary crop to adjust stream image to required size
            if (((float)requiredSize.width / (float)requiredSize.height) > ((float)_streamSizes[i].width / (float)_streamSizes[i].height))
            {
                float scaleFactor  = (float)requiredSize.width / (float)_streamSizes[i].width;
                float heightScaled = _streamSizes[i].height * scaleFactor;
                crop += heightScaled - requiredSize.height;
                crop += (float)_streamSizes[i].width - (float)requiredSize.width;
            }
            else
            {
                float scaleFactor = (float)requiredSize.height / (float)_streamSizes[i].height;
                float widthScaled = _streamSizes[i].width * scaleFactor;
                crop += widthScaled - requiredSize.width;
                crop += (float)_streamSizes[i].height - (float)requiredSize.height;
            }

            float cropScaleScore = crop;
            matchingSizes.push_back(std::make_pair(cropScaleScore, i));
        }
    }
    //sort by crop
    std::sort(matchingSizes.begin(), matchingSizes.end());

    return _streamSizes[matchingSizes.front().second];
}
