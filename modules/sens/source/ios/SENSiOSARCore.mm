#include "SENSiOSARCore.h"

SENSiOSARCore::SENSiOSARCore()
{
    _arcoreDelegate = [[SENSiOSARCoreDelegate alloc] init];
    _available      = [_arcoreDelegate isAvailable];
}

bool SENSiOSARCore::init()
{
    if (!_available)
        return false;

    [_arcoreDelegate initARKit];

    return true;
}

bool SENSiOSARCore::isReady()
{
    return _arcoreDelegate != nullptr;
}

bool SENSiOSARCore::resume()
{
    bool success = false;
    if (_arcoreDelegate)
        success = [_arcoreDelegate run];

    if (success)
    {
        _pause = false;
        _started = true; //for SENSCameraBase
    }

    return success;
}

void SENSiOSARCore::reset()
{
    if (_arcoreDelegate)
        [_arcoreDelegate reset];
}

void SENSiOSARCore::pause()
{
    if (_arcoreDelegate)
        [_arcoreDelegate pause];
    
    _pause = true;
    _started = false; //for SENSCameraBase
}

bool SENSiOSARCore::update(cv::Mat& pose)
{
    //retrieve the latest frame from arkit delegate
    cv::Mat intrinsic;
    cv::Mat imgBGR;
    bool    isTracking;
    [_arcoreDelegate latestFrame:&pose withImg:&imgBGR AndIntrinsic:&intrinsic AndImgWidth:&_inputFrameW AndImgHeight:&_inputFrameH IsTracking:&isTracking];

    if (!imgBGR.empty())
    {
        updateFrame(imgBGR, intrinsic, true);
    }
    else
        Utils::log("SENSiOSARCore", "frame is empty!");
    return isTracking;
}

void SENSiOSARCore::retrieveCaptureProperties()
{
    //the SENSCameraBase needs to have a valid frame, otherwise we cannot estimate the fov correctly
    if(!_frame)
    {
        resume();
        HighResTimer t;
        cv::Mat pose;
        do {
            update(pose);
        }
        while(!_frame && t.elapsedTimeInSec() < 5.f);
        
        pause();
    }
    
    if(_frame)
    {
        std::string      deviceId = "ARKit";
        SENSCameraFacing facing = SENSCameraFacing::BACK;

        float focalLengthPix = -1.f;
        if(!_frame->intrinsics.empty())
        {
            focalLengthPix = 0.5 * (_frame->intrinsics.at<double>(0, 0) + _frame->intrinsics.at<double>(1, 1));
        }
        SENSCameraDeviceProperties devProp(deviceId, facing);
        devProp.add(_frame->imgBGR.cols, _frame->imgBGR.rows, focalLengthPix);
        _captureProperties.push_back(devProp);
    }
    else
        Utils::warnMsg("SENSiOSARCore", "retrieveCaptureProperties: Could not retrieve a valid frame!", __LINE__, __FILE__);
}

const SENSCaptureProperties& SENSiOSARCore::captureProperties()
{
    if(_captureProperties.size() == 0)
        retrieveCaptureProperties();
    
    return _captureProperties;
}

//This function does not really start the camera as for the arcore iplementation, the frame gets only updated with a call to arcore::update.
//This function is needed to correctly use arcore as a camera in SENSCVCamera
const SENSCameraConfig& SENSiOSARCore::start(std::string    deviceId,
                              const SENSCameraStreamConfig& streamConfig,
                              bool                          provideIntrinsics)
{
    //define capture properties
    if(_captureProperties.size() == 0)
        retrieveCaptureProperties();
    
    if (_captureProperties.size() == 0)
        throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

    if (!_captureProperties.containsDeviceId(deviceId))
        throw SENSException(SENSType::CAM, "DeviceId does not exist!", __LINE__, __FILE__);
    
    SENSCameraFacing                  facing = SENSCameraFacing::UNKNOWN;
    const SENSCameraDeviceProperties* props  = _captureProperties.camPropsForDeviceId(deviceId);
    if (props)
        facing = props->facing();

    //init config here before processStart
    _config = SENSCameraConfig(deviceId,
                               streamConfig,
                               facing,
                               SENSCameraFocusMode::CONTINIOUS_AUTO_FOCUS);
    //inform camera listeners
    processStart();

    _started = true;
    return _config;
}
