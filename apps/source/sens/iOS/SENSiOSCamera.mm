#import "SENSiOSCamera.h"
#include <Utils.h>
#include <functional>
#include <sens/SENSUtils.h>
#include <memory>

SENSiOSCamera::SENSiOSCamera()
{
    _cameraDelegate = [[SENSiOSCameraDelegate alloc] init];
    [_cameraDelegate setCallback:std::bind(&SENSiOSCamera::processNewFrame, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)];
    //todo: fixme
    _permissionGranted = true;
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

    _config.streamConfig        = &streamConfig;
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
             videoStabilizationState:_config.enableVideoStabilization
                     intrinsicsState:_config.provideIntrinsics])
    {
        //initialize guessed camera calibration
        initCalibration();
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

    //for ios to retrieve intrinsics we have to disable video stabilization
    if (provideIntrinsics)
        _config.enableVideoStabilization = false;

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
                 videoStabilizationState:_config.enableVideoStabilization
                         intrinsicsState:_config.provideIntrinsics])
        {
            //calculate crop for config
            int cropW, cropH, resW, resH;
            SENS::calcCrop({streamConfig->widthPix, streamConfig->heightPix},
                           (float)imgRGBSize.width / (float)imgRGBSize.height,
                           _config.cropWidth,
                           _config.cropHeight,
                           _config.targetWidth,
                           _config.targetHeight);

            _config.deviceId     = bestConfig.first->deviceId();
            _config.streamConfig = streamConfig;
            _started             = true;
        }
        else
            throw SENSException(SENSType::CAM, "Could not start camera!", __LINE__, __FILE__);
    }
    else
        throw SENSException(SENSType::CAM, "Could not start camera!", __LINE__, __FILE__);

    //initialize guessed camera calibration
    initCalibration();

    return _config;
}

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
    SENSFramePtr newFrame;
    {
        std::lock_guard<std::mutex> lock(_processedFrameMutex);
        newFrame = std::move(_processedFrame);
    }

    if (newFrame && !newFrame->intrinsics.empty())
    {
        _calibration = std::make_unique<SENSCalibration>(newFrame->intrinsics,
                                                         cv::Size(_config.streamConfig->widthPix, _config.streamConfig->heightPix),
                                                         _calibration->isMirroredH(),
                                                         _calibration->isMirroredV(),
                                                         _calibration->camType(),
                                                         _calibration->computerInfos());
        //adjust calibration
        if ((_config.manipWidth > 0 && _config.manipHeight > 0) || _config.manipWidth != _config.streamConfig->widthPix || _config.manipHeight != _config.streamConfig->heightPix)
            _calibration->adaptForNewResolution({_config.manipWidth, _config.manipHeight}, false);
        else if (_config.targetWidth != _config.streamConfig->widthPix || _config.targetHeight != _config.streamConfig->heightPix)
            _calibration->adaptForNewResolution({_config.targetWidth, _config.targetHeight}, false);
    }

    return newFrame;
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
            simd_float3 col             = camMat3x3->columns[i];
            intrinsics.at<double>(0, i) = (double)col[0];
            intrinsics.at<double>(1, i) = (double)col[1];
            intrinsics.at<double>(2, i) = (double)col[2];
        }
    }

    SENSFramePtr sensFrame = postProcessNewFrame(rgbImg, intrinsics, intrinsicsChanged);

    {
        std::lock_guard<std::mutex> lock(_processedFrameMutex);
        _processedFrame = std::move(sensFrame);
    }
}
