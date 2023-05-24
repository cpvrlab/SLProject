#include "SENSiOSARKit.h"

SENSiOSARKit::SENSiOSARKit()
{
    _arcoreDelegate = [[SENSiOSARCoreDelegate alloc] init];
    _available      = [_arcoreDelegate isAvailable];
}

bool SENSiOSARKit::init(unsigned int textureId, bool retrieveCpuImg, int targetWidth)
{
    if (!_available)
        return false;

    [_arcoreDelegate initARKit];

    return true;
}

bool SENSiOSARKit::isReady()
{
    return _arcoreDelegate != nullptr;
}

bool SENSiOSARKit::resume()
{
    bool success = false;
    if (_arcoreDelegate)
        success = [_arcoreDelegate run];

    if (success)
    {
        _pause   = false;
        _started = true; //for SENSBaseCamera
    }

    return success;
}

void SENSiOSARKit::reset()
{
    if (_arcoreDelegate)
        [_arcoreDelegate reset];
}

void SENSiOSARKit::pause()
{
    if (_arcoreDelegate)
        [_arcoreDelegate pause];

    _pause   = true;
    _started = false; //for SENSBaseCamera
}

bool SENSiOSARKit::update(cv::Mat& pose)
{
    //retrieve the latest frame from arkit delegate
    cv::Mat intrinsic;
    cv::Mat imgBGR;
    bool    isTracking;

    if (_fetchPointCloud)
        [_arcoreDelegate latestFrame:&pose withImg:&imgBGR AndIntrinsic:&intrinsic AndImgWidth:&_inputFrameW AndImgHeight:&_inputFrameH IsTracking:&isTracking WithPointClout:&_pointCloud];
    else
        [_arcoreDelegate latestFrame:&pose withImg:&imgBGR AndIntrinsic:&intrinsic AndImgWidth:&_inputFrameW AndImgHeight:&_inputFrameH IsTracking:&isTracking WithPointClout:nullptr];

    if (!imgBGR.empty())
    {
        updateFrame(imgBGR, intrinsic, true, imgBGR.cols, imgBGR.rows);
    }
    else
        Utils::log("SENSiOSARKit", "frame is empty!");
    return isTracking;
}

void SENSiOSARKit::retrieveCaptureProperties()
{
    //the SENSBaseCamera needs to have a valid frame, otherwise we cannot estimate the fov correctly
    if (!_frame)
    {
        resume();
        HighResTimer t;
        cv::Mat      pose;
        do {
            update(pose);
        } while (!_frame && t.elapsedTimeInSec() < 5.f);

        pause();
    }

    if (_frame)
    {
        std::string      deviceId = "ARKit";
        SENSCameraFacing facing   = SENSCameraFacing::BACK;

        float focalLengthPix = -1.f;
        if (!_frame->intrinsics.empty())
        {
            focalLengthPix = 0.5 * (_frame->intrinsics.at<double>(0, 0) + _frame->intrinsics.at<double>(1, 1));
        }
        SENSCameraDeviceProps devProp(deviceId, facing);
        devProp.add(_frame->imgBGR.cols, _frame->imgBGR.rows, focalLengthPix);
        _captureProperties.push_back(devProp);
    }
    else
        Utils::warnMsg("SENSiOSARKit", "retrieveCaptureProperties: Could not retrieve a valid frame!", __LINE__, __FILE__);
}

const SENSCaptureProps& SENSiOSARKit::captureProperties()
{
    if (_captureProperties.size() == 0)
        retrieveCaptureProperties();

    return _captureProperties;
}

//This function does not really start the camera as for the arcore iplementation, the frame gets only updated with a call to arcore::update.
//This function is needed to correctly use arcore as a camera in SENSCVCamera
const SENSCameraConfig& SENSiOSARKit::start(std::string                   deviceId,
                                            const SENSCameraStreamConfig& streamConfig,
                                            bool                          provideIntrinsics)
{
    //define capture properties
    if (_captureProperties.size() == 0)
        retrieveCaptureProperties();

    if (_captureProperties.size() == 0)
        throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

    if (!_captureProperties.containsDeviceId(deviceId))
        throw SENSException(SENSType::CAM, "DeviceId does not exist!", __LINE__, __FILE__);

    SENSCameraFacing             facing = SENSCameraFacing::UNKNOWN;
    const SENSCameraDeviceProps* props  = _captureProperties.camPropsForDeviceId(deviceId);
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
