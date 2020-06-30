#import "SENSiOSCamera.h"
#include <Utils.h>
#include <functional>
#include <sens/SENSUtils.h>
#include <memory>

SENSiOSCamera::SENSiOSCamera()
{
    _cameraDelegate = [[SENSiOSCameraDelegate alloc] init];
    [_cameraDelegate setCallback:std::bind(&SENSiOSCamera::processNewFrame, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)];
}

SENSiOSCamera::~SENSiOSCamera()
{
}

const SENSCameraConfig& SENSiOSCamera::start(std::string                   deviceId,
                                             const SENSCameraStreamConfig& streamConfig,
                                             cv::Size                      imgRGBSize,
                                             bool                          mirrorV,
                                             bool                          mirrorH,
                                             bool                          convToGrayToImgManip,
                                             cv::Size                      imgManipSize,
                                             bool                          provideIntrinsics,
                                             float                         fovDegFallbackGuess)
{
    if (_started)
    {
        Utils::warnMsg("SENSiOSCamera", "Call to start was ignored. Camera is currently running!", __LINE__, __FILE__);
        return _config;
    }

    _config.deviceId            = deviceId;
    _config.mirrorV             = mirrorV;
    _config.mirrorH             = mirrorH;
    _config.convertManipToGray  = convToGrayToImgManip;
    _config.manipWidth          = imgManipSize.width;
    _config.manipHeight         = imgManipSize.height;
    _config.provideIntrinsics   = provideIntrinsics;
    _config.fovDegFallbackGuess = fovDegFallbackGuess;

    if (imgRGBSize.width > 0 && imgRGBSize.height > 0)
    {
        _config.targetWidth  = imgRGBSize.width;
        _config.targetHeight = imgRGBSize.height;
    }
    else
    {
        _config.targetWidth  = streamConfig.widthPix;
        _config.targetHeight = streamConfig.heightPix;
    }

    //retrieve all camera characteristics
    if (_caputureProperties.size() == 0)
        _caputureProperties = [_cameraDelegate retrieveCaptureProperties];

    if (_caputureProperties.size() == 0)
        throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

    if (!_caputureProperties.containsDeviceId(deviceId))
        throw SENSException(SENSType::CAM, "DeviceId does not exist!", __LINE__, __FILE__);

    NSString* devId = [NSString stringWithUTF8String:_config.deviceId.c_str()];

    if ([_cameraDelegate startCamera:devId
                           withWidth:streamConfig.widthPix
                           andHeight:streamConfig.heightPix
                      autoFocusState:YES //alway on on ios because they provide dynamic intrinsics
             videoStabilizationState:_config.enableVideoStabilization])
    {
        _started = true;
    }
    else
        throw SENSException(SENSType::CAM, "Could not start camera!", __LINE__, __FILE__);

    return _config;
}

const SENSCameraConfig& SENSiOSCamera::start(SENSCameraFacing facing,
                                             float            approxHorizFov,
                                             cv::Size         imgRGBSize,
                                             bool             mirrorV,
                                             bool             mirrorH,
                                             bool             scaleImgRGB,
                                             bool             convToGrayToImgManip,
                                             cv::Size         imgManipSize,
                                             bool             provideIntrinsics,
                                             float            fovDegFallbackGuess)
{
    if (_started)
    {
        Utils::warnMsg("SENSiOSCamera", "Call to start was ignored. Camera is currently running!", __LINE__, __FILE__);
        return _config;
    }

    _config.mirrorV             = mirrorV;
    _config.mirrorH             = mirrorH;
    _config.convertManipToGray  = convToGrayToImgManip;
    _config.manipWidth          = imgManipSize.width;
    _config.manipHeight         = imgManipSize.height;
    _config.provideIntrinsics   = provideIntrinsics;
    _config.fovDegFallbackGuess = fovDegFallbackGuess;

    //retrieve all camera characteristics
    if (_caputureProperties.size() == 0)
        _caputureProperties = [_cameraDelegate retrieveCaptureProperties];

    if (_caputureProperties.size() == 0)
        throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

    auto bestConfig = _caputureProperties.findBestMatchingConfig(facing, approxHorizFov, imgRGBSize.width, imgRGBSize.height);
    if (bestConfig.first && bestConfig.second)
    {
        NSString*                     devId        = [NSString stringWithUTF8String:bestConfig.first->deviceId().c_str()];
        const SENSCameraStreamConfig* streamConfig = bestConfig.second;

        if ([_cameraDelegate startCamera:devId
                               withWidth:streamConfig->widthPix
                               andHeight:streamConfig->heightPix
                          autoFocusState:YES //alway on on ios because they provide dynamic intrinsics
                 videoStabilizationState:_config.enableVideoStabilization])
        {
            _config.deviceId     = bestConfig.first->deviceId();
            _config.streamConfig = streamConfig;
            _config.targetWidth  = streamConfig->widthPix;
            _config.targetHeight = streamConfig->heightPix;
            _started             = true;
        }
        else
            throw SENSException(SENSType::CAM, "Could not start camera!", __LINE__, __FILE__);
    }
    else
        throw SENSException(SENSType::CAM, "Could not start camera!", __LINE__, __FILE__);

    //todo: focal length has to fit to resolution!!!!!!

    //we make a calibration with full resolution and adjust it to the manipulated image size later if neccessary
    float horizFOVDev = SENS::calcFOVDegFromFocalLengthPix(_config.streamConfig->focalLengthPix, _config.streamConfig->widthPix);
    _calibration      = std::make_unique<SENSCalibration>(cv::Size(_config.streamConfig->widthPix, _config.streamConfig->heightPix),
                                                     horizFOVDev,
                                                     false,
                                                     false,
                                                     SENSCameraType::BACKFACING,
                                                     Utils::ComputerInfos().get());
    //adjust calibration
    if (_config.manipWidth > 0 && _config.manipHeight > 0 && _config.manipWidth != _config.streamConfig->widthPix && _config.manipHeight != _config.streamConfig->heightPix)
        _calibration->adaptForNewResolution({_config.manipWidth, _config.manipHeight}, false);
    else if (_config.targetWidth != _config.streamConfig->widthPix && _config.targetHeight != _config.streamConfig->heightPix)
        _calibration->adaptForNewResolution({_config.targetWidth, _config.targetHeight}, false);

    return _config;
}

/*
void SENSiOSCamera::start(std::string deviceId, int width, int height, SENSCameraFocusMode focusMode)
{
    if (!_started)
    {
        _config.deviceId     = deviceId;
        _config.targetWidth  = width;
        _config.targetHeight = height;
        _config.focusMode    = focusMode;

        //retrieve all camera characteristics
        if (_caputureProperties.size() == 0)
            _caputureProperties = [_cameraDelegate retrieveCaptureProperties];

        if (_caputureProperties.size() == 0)
            throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

        //_caputureProperties.findBestMatchingConfig(SENSCameraFacing::BACK, 65.f, 600, 450);
        //_caputureProperties.findBestMatchingConfig(SENSCameraFacing::BACK, 65.f, 1000, 500);
        //_caputureProperties.findBestMatchingConfig(SENSCameraFacing::BACK, 65.f, 800, 600);

        //check that device id exists
        auto itChars = std::find_if(_caputureProperties.begin(), _caputureProperties.end(), [&](const SENSCameraDeviceProperties& cmp) { return cmp.deviceId() == _config.deviceId; });
        if (itChars == _caputureProperties.end())
            throw SENSException(SENSType::CAM, "Could not find device id!", __LINE__, __FILE__);

        _config.streamConfigIndex                = itChars->findBestMatchingConfig({_config.targetWidth, _config.targetHeight});
        const SENSCameraStreamConfig& bestConfig = itChars->streamConfigs().at(_config.streamConfigIndex);

        NSString* devId           = [NSString stringWithUTF8String:_config.deviceId.c_str()];
        BOOL      enableAutoFocus = (_config.focusMode == SENSCameraFocusMode::CONTINIOUS_AUTO_FOCUS) ? YES : NO;
        if ([_cameraDelegate startCamera:devId
                               withWidth:bestConfig.widthPix
                               andHeight:bestConfig.heightPix
                          autoFocusState:enableAutoFocus
                 videoStabilizationState:_config.enableVideoStabilization])
        {
            _started = true;
        }
        else
            throw SENSException(SENSType::CAM, "Could not start camera!", __LINE__, __FILE__);
    }
    else
        Utils::log("SENSiOSCamera", "Camera already started but start called!");
}
*/

void SENSiOSCamera::stop()
{
    if (_started)
    {
        if ([_cameraDelegate stopCamera])
            _started = false;
    }
    else
        Utils::log("SENSiOSCamera", "Camera not started but stop called!");
}

const SENSCaptureProperties& SENSiOSCamera::captureProperties()
{
    if (_caputureProperties.size() == 0)
        _caputureProperties = [_cameraDelegate retrieveCaptureProperties];

    return _caputureProperties;
}

SENSFramePtr SENSiOSCamera::latestFrame()
{
    std::lock_guard<std::mutex> lock(_processedFrameMutex);
    return std::move(_processedFrame);
}

void SENSiOSCamera::processNewFrame(unsigned char* data, int imgWidth, int imgHeight, matrix_float3x3* camMat3x3)
{
    //Utils::log("SENSiOSCamera", "processNewFrame: w %d w %d", imgWidth, imgHeight);
    cv::Mat rgba(imgHeight, imgWidth, CV_8UC4, (void*)data);
    cv::Mat rgbImg;
    cvtColor(rgba, rgbImg, cv::COLOR_RGBA2RGB, 3);

    cv::Mat intrinsics;
    bool    intrinsicsChanged = false;
    if (camMat3x3)
    {
        intrinsicsChanged = true;
        intrinsics        = cv::Mat_<double>(3, 3);
        for (int i = 0; i < 3; ++i)
        {
            simd_float3 col            = camMat3x3->columns[i];
            intrinsics.at<double>(0, i) = (double)col[0];
            intrinsics.at<double>(1, i) = (double)col[1];
            intrinsics.at<double>(2, i) = (double)col[2];
        }
        //std::cout << "Mat cv" << std::endl;
        //std::cout << intrinsics << std::endl;
    }

    SENSFramePtr sensFrame = postProcessNewFrame(rgbImg, intrinsics, intrinsicsChanged);

    Utils::log("SENSiOSCamera", "next : w %d w %d", sensFrame->imgRGB.size().width, sensFrame->imgRGB.size().height);
    {
        std::lock_guard<std::mutex> lock(_processedFrameMutex);
        _processedFrame = std::move(sensFrame);
    }
}
