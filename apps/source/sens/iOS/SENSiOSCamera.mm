#import "SENSiOSCamera.h"
#include <Utils.h>
#include <functional>
#include <sens/SENSUtils.h>
#include <memory>

SENSiOSCamera::SENSiOSCamera()
{
    _cameraDelegate = [[SENSiOSCameraDelegate alloc] init];
    [_cameraDelegate setPermissionCB:std::bind(&SENSiOSCamera::updatePermission, this, std::placeholders::_1)];
    [_cameraDelegate setUpdateCB:std::bind(&SENSiOSCamera::processNewFrame, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)];
    //todo: fixme
    _permissionGranted = true;
}

SENSiOSCamera::~SENSiOSCamera()
{
}

const SENSCameraConfig& SENSiOSCamera::start(std::string                   deviceId,
                                             const SENSCameraStreamConfig& streamConfig,
                                             cv::Size                      imgBGRSize,
                                             bool                          mirrorV,
                                             bool                          mirrorH,
                                             bool                          convToGrayToImgManip,
                                             int                           imgManipWidth,
                                             bool                          provideIntrinsics,
                                             float                         fovDegFallbackGuess)
{
    if (_started)
    {
        Utils::warnMsg("SENSiOSCamera", "Call to start was ignored. Camera is currently running!", __LINE__, __FILE__);
        return _config;
    }

    cv::Size targetSize;

    if (imgBGRSize.width > 0 && imgBGRSize.height > 0)
    {
        targetSize.width  = imgBGRSize.width;
        targetSize.height = imgBGRSize.height;
    }
    else
    {
        targetSize.width  = streamConfig.widthPix;
        targetSize.height = streamConfig.heightPix;
    }

    cv::Size imgManipSize(imgManipWidth,
                          (int)((float)imgManipWidth * (float)targetSize.height / (float)targetSize.width));

    //retrieve all camera characteristics
    if (_captureProperties.size() == 0)
        _captureProperties = [_cameraDelegate retrieveCaptureProperties];

    if (_captureProperties.size() == 0)
        throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

    if (!_captureProperties.containsDeviceId(deviceId))
        throw SENSException(SENSType::CAM, "DeviceId does not exist!", __LINE__, __FILE__);

    NSString* devId = [NSString stringWithUTF8String:deviceId.c_str()];

    BOOL enableVideoStabilization = YES;
    if (provideIntrinsics)
        enableVideoStabilization = NO;

    if ([_cameraDelegate startCamera:devId
                           withWidth:streamConfig.widthPix
                           andHeight:streamConfig.heightPix
                      autoFocusState:YES //alway on on ios because they provide dynamic intrinsics
             videoStabilizationState:enableVideoStabilization
                     intrinsicsState:provideIntrinsics])
    {
        //init config here
        _config = SENSCameraConfig(deviceId,
                                   streamConfig,
                                   SENSCameraFocusMode::UNKNOWN,
                                   targetSize.width,
                                   targetSize.height,
                                   imgManipSize.width,
                                   imgManipSize.height,
                                   mirrorH,
                                   mirrorV,
                                   convToGrayToImgManip);
        //initialize guessed camera calibration
        initCalibration(fovDegFallbackGuess);
        _started = true;
    }
    else
    {
        throw SENSException(SENSType::CAM, "Could not start camera!", __LINE__, __FILE__);
    }

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
    if (_captureProperties.size() == 0)
        _captureProperties = [_cameraDelegate retrieveCaptureProperties];

    return _captureProperties;
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
                                                         cv::Size(_config.streamConfig.widthPix, _config.streamConfig.heightPix),
                                                         _calibration->isMirroredH(),
                                                         _calibration->isMirroredV(),
                                                         _calibration->camType(),
                                                         _calibration->computerInfos());
        //adjust calibration
        if (_config.targetWidth != _config.streamConfig.widthPix || _config.targetHeight != _config.streamConfig.heightPix)
            _calibration->adaptForNewResolution({_config.targetWidth, _config.targetHeight}, false);
    }

    return newFrame;
}

void SENSiOSCamera::processNewFrame(unsigned char* data, int imgWidth, int imgHeight, matrix_float3x3* camMat3x3)
{
    //Utils::log("SENSiOSCamera", "processNewFrame: w %d w %d", imgWidth, imgHeight);
    cv::Mat bgra(imgHeight, imgWidth, CV_8UC4, (void*)data);
    cv::Mat bgrImg;
    cvtColor(bgra, bgrImg, cv::COLOR_BGRA2BGR, 3);

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
    
    //todo: move to base class
    {
        SENSTimePt timePt = SENSClock::now();
        std::lock_guard<std::mutex> lock(_listenerMutex);
        for(SENSCameraListener* l : _listeners)
            l->onFrame(timePt, bgrImg.clone());
    }

    SENSFramePtr sensFrame = postProcessNewFrame(bgrImg, intrinsics, intrinsicsChanged);

    {
        std::lock_guard<std::mutex> lock(_processedFrameMutex);
        _processedFrame = std::move(sensFrame);
    }
}

void SENSiOSCamera::updatePermission(bool granted)
{
    _permissionGranted = granted;
}
