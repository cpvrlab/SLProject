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
        /*
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
        _config.targetHeight = 360;
        _config.focusMode = focusMode;
        
        //retrieve all camera characteristics
        if (_caputureProperties.size() == 0)
            _caputureProperties = [_cameraDelegate retrieveCaptureProperties];

        if(_caputureProperties.size() == 0)
            throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);
        
        //check that device id exists
        auto itChars = std::find_if(_caputureProperties.begin(), _caputureProperties.end(),
                                    [&]( const SENSCameraCharacteristics& cmp){ return cmp.deviceId() == _config.deviceId;});
        if(itChars == _caputureProperties.end())
            throw SENSException(SENSType::CAM, "Could not find device id!", __LINE__, __FILE__);
        
        const SENSCameraCharacteristics::StreamConfig& bestConfig = itChars->findBestMatchingConfig({_config.targetWidth, _config.targetHeight});

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
    Utils::log("SENSiOSCamera", "processNewFrame");
    
    cv::Mat rgba(imgHeight, imgWidth, CV_8UC4, (void*)data);
    cv::Mat rgbImg;
    cvtColor(rgba, rgbImg, cv::COLOR_RGBA2RGB, 3);
    
    /*
    cv::Size inputSize = rgbImg.size();

    // Crop Video image to required aspect ratio
    int cropW = 0, cropH = 0;
    SENS::cropImage(rgbImg,(float)_config.targetWidth / (float)_config.targetHeight, cropW, cropH);

    // Mirroring
    SENS::mirrorImage(rgbImg, _config.mirrorH, _config.mirrorV);

    // Create grayscale
    cv::Mat grayImg;
    if (_config.convertToGray)
    {
        cv::cvtColor(rgbImg, grayImg, cv::COLOR_BGR2GRAY);
    }
     */

    //SENSFramePtr sensFrame = std::make_shared<SENSFrame>(rgbImg, grayImg, inputSize.width, inputSize.height, cropW, cropH, _config.mirrorH, _config.mirrorV);
    
    SENSFramePtr sensFrame = postProcessNewFrame(rgbImg);
    {
        std::lock_guard<std::mutex> lock(_processedFrameMutex);
        _processedFrame = std::move(sensFrame);
    }
}
