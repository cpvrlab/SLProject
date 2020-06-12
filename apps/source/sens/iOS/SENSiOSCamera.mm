#import "SENSiOSCamera.h"
#include <Utils.h>

SENSiOSCamera::SENSiOSCamera()
{
    _cameraDelegate = [[SENSiOSCameraDelegate alloc] init];
}

SENSiOSCamera::~SENSiOSCamera()
{
}

void SENSiOSCamera::start(SENSCameraConfig config)
{
    if(_started)
    {
        Utils::log("SENSiOSCamera", "Camera already started!");
        return;
    }
    [_cameraDelegate startCamera];
}

void SENSiOSCamera::start(std::string id, int width, int height)
{
    
}

void SENSiOSCamera::stop()
{
    
}

std::vector<SENSCameraCharacteristics> SENSiOSCamera::getAllCameraCharacteristics()
{
    std::vector<SENSCameraCharacteristics> characteristicsVec;
    {
        SENSCameraCharacteristics characteristics;
        characteristics.cameraId = "0";
        characteristics.streamConfig.add(cv::Size(640, 480));
        
        characteristicsVec.push_back(characteristics);
    }
    return characteristicsVec;
}

SENSFramePtr SENSiOSCamera::getLatestFrame()
{
    SENSFramePtr frame;
    return frame;
}
