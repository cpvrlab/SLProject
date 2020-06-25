#import "SENSiOSCamera.h"
#include <Utils.h>
#include <functional>
#include <sens/SENSUtils.h>

SENSiOSCamera::SENSiOSCamera()
{
    _cameraDelegate = [[SENSiOSCameraDelegate alloc] init];
    [_cameraDelegate setCallback:std::bind(&SENSiOSCamera::processNewFrame, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)];
}

SENSiOSCamera::~SENSiOSCamera()
{
}

/*
void SENSiOSCamera::start(SENSCameraConfig config)
{
    if(!_started)
    {
        _config = config;
        //_targetWdivH = (float)_config.targetWidth / (float)_config.targetHeight;

        //retrieve all camera characteristics
        if (_caputureProperties.size() == 0)
        {
            _caputureProperties = [_cameraDelegate retrieveCaptureProperties];
        }

        //find and set current camera characteristic depending on camera device id
        if (_characteristics.cameraId != config.deviceId)
        {
            for (const SENSCameraCharacteristics& c : _caputureProperties)
            {
                if (_config.deviceId == c.cameraId)
                {
                    _characteristics = c;
                    break;
                }
            }
        }

        if (_characteristics.cameraId.empty())
        {
            throw SENSException(SENSType::CAM, "Device id does not exist!", __LINE__, __FILE__);
        }
        
        //find best matching size in available stream configuration sizes
        SENSCameraStreamConfigs::Config bestConfig =
            _characteristics.streamConfig.findBestMatchingConfig({_config.targetWidth, _config.targetHeight});
        
        NSString* devId = [NSString stringWithUTF8String:_config.deviceId.c_str()];
        if([_cameraDelegate startCamera:devId withWidth:bestConfig.widthPix andHeight:bestConfig.heightPix])
            _started = true;
        else
            throw SENSException(SENSType::CAM, "Could not start camera!", __LINE__, __FILE__);
        
        _currStreamConfig = bestConfig;
         */
/*
    }
    else
        Utils::log("SENSiOSCamera", "Camera already started but start called!");
}
*/

void SENSiOSCamera::start(std::string deviceId, int width, int height, SENSCameraFocusMode focusMode)
{
    if(!_started)
    {
        _config.deviceId = deviceId;
        _config.targetWidth = width;
        _config.targetHeight = height;
        _config.focusMode = focusMode;
        
        //retrieve all camera characteristics
        if (_caputureProperties.size() == 0)
            _caputureProperties = [_cameraDelegate retrieveCaptureProperties];

        if(_caputureProperties.size() == 0)
            throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);
        
        
        auto best = _caputureProperties.findBestMatchingConfig(SENSCameraFacing::BACK, 65.f, _config.targetWidth, _config.targetHeight);
        
        
        
        //check that device id exists
        auto itChars = std::find_if(_caputureProperties.begin(), _caputureProperties.end(),
                                    [&]( const SENSCameraCharacteristics& cmp){ return cmp.deviceId() == _config.deviceId;});
        if(itChars == _caputureProperties.end())
            throw SENSException(SENSType::CAM, "Could not find device id!", __LINE__, __FILE__);
        
        _config.streamConfigIndex = itChars->findBestMatchingConfig({_config.targetWidth, _config.targetHeight});
        const SENSCameraCharacteristics::StreamConfig& bestConfig = itChars->streamConfigs().at(_config.streamConfigIndex);
        
        NSString* devId = [NSString stringWithUTF8String:_config.deviceId.c_str()];
        if([_cameraDelegate startCamera:devId withWidth:bestConfig.widthPix andHeight:bestConfig.heightPix])
            _started = true;
        else
            throw SENSException(SENSType::CAM, "Could not start camera!", __LINE__, __FILE__);
    }
    else
        Utils::log("SENSiOSCamera", "Camera already started but start called!");
}

void SENSiOSCamera::stop()
{
    if(_started)
    {
        if([_cameraDelegate stopCamera])
            _started = false;
    }
    else
        Utils::log("SENSiOSCamera", "Camera not started but stop called!");
}

const SENSCaptureProperties& SENSiOSCamera::getCaptureProperties()
{
    if(_caputureProperties.size() == 0)
        _caputureProperties = [_cameraDelegate retrieveCaptureProperties];
    
    return _caputureProperties;
}

SENSFramePtr SENSiOSCamera::getLatestFrame()
{
    std::lock_guard<std::mutex> lock(_processedFrameMutex);
    return _processedFrame;
}

void SENSiOSCamera::processNewFrame(unsigned char* data, int imgWidth, int imgHeight)
{
    //Utils::log("SENSiOSCamera", "processNewFrame: w %d w %d", imgWidth, imgHeight);
    cv::Mat rgba(imgHeight, imgWidth, CV_8UC4, (void*)data);
    cv::Mat rgbImg;
    cvtColor(rgba, rgbImg, cv::COLOR_RGBA2RGB, 3);
    SENSFramePtr sensFrame = postProcessNewFrame(rgbImg);
    
    //Utils::log("SENSiOSCamera", "next : w %d w %d", sensFrame->imgRGB.size().width, sensFrame->imgRGB.size().height);
    {
        std::lock_guard<std::mutex> lock(_processedFrameMutex);
        _processedFrame = std::move(sensFrame);
    }
}
