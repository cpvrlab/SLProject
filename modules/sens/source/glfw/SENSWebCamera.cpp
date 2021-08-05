#include "SENSWebCamera.h"
#include <SENSException.h>
#include <SENSUtils.h>
#include <Utils.h>

#define LOG_WEBCAM_WARN(...) Utils::log("SENSWebCamera", __VA_ARGS__);
//#define LOG_WEBCAM_INFO(...) Utils::log("SENSWebCamera", __VA_ARGS__);
//#define LOG_WEBCAM_DEBUG(...) Utils::log("SENSWebCamera", __VA_ARGS__);
//#define LOG_WEBCAM_WARN
#define LOG_WEBCAM_INFO
#define LOG_WEBCAM_DEBUG

const SENSCameraConfig& SENSWebCamera::start(std::string                   deviceId,
                                             const SENSCameraStreamConfig& streamConfig,
                                             bool                          provideIntrinsics)
{
    if (_started)
    {
        Utils::warnMsg("SENSWebCamera", "Call to start was ignored. Camera is currently running!", __LINE__, __FILE__);
        return _config;
    }

    //retrieve all camera characteristics
    if (_captureProperties.size() == 0)
        captureProperties();

    if (_captureProperties.size() == 0)
        throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

    if (!_captureProperties.containsDeviceId(deviceId))
        throw SENSException(SENSType::CAM, "DeviceId does not exist!", __LINE__, __FILE__);

    int id = std::stoi(deviceId);
    _videoCapture.open(id);

    if (!_videoCapture.isOpened())
        throw SENSException(SENSType::CAM, "Could not open camera with id: " + deviceId, __LINE__, __FILE__);

    _videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, streamConfig.widthPix);
    _videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, streamConfig.heightPix);
    _started           = true;
    _permissionGranted = true;

    //init config here
    _config = SENSCameraConfig(deviceId,
                               streamConfig,
                               SENSCameraFacing::UNKNOWN,
                               SENSCameraFocusMode::UNKNOWN);

    processStart();

    //start thread
    _cameraThread = std::thread(&SENSWebCamera::grab, this);

    return _config;
}

void SENSWebCamera::stop()
{
    _stop = true;
    if (_cameraThread.joinable())
        _cameraThread.join();
    _stop = false;

    if (_videoCapture.isOpened())
    {
        _videoCapture.release();

        _started = false;
    }
    else
    {
        LOG_WEBCAM_INFO("stop: ignored because camera is not open!");
    }
}

void SENSWebCamera::grab()
{
    while (!_stop)
    {
        cv::Mat bgrImg;
        if (_videoCapture.read(bgrImg))
        {
            if (bgrImg.cols == _config.streamConfig.widthPix &&
                bgrImg.rows == _config.streamConfig.heightPix)
                updateFrame(bgrImg, cv::Mat(), false, bgrImg.rows, bgrImg.cols);
            else
                LOG_WEBCAM_WARN("video capture delivers wrong resolution!");
        }
    }
}

const SENSCaptureProps& SENSWebCamera::captureProperties()
{
    if (!_captureProperties.size())
    {
        //definition of standard frame sizes that we want to test for support
        static std::vector<cv::Size> testSizes = {
          {640, 480},
          {640, 360},
          {960, 540},
          {1280, 960},
          {1280, 720},
          {1920, 1080}};

        //There is an invisible list of devices populated from your os and your webcams appear there in the order you plugged them in.
        //If you're e.g on a laptop with a builtin camera, that will be id 0, if you plug in an additional one, that's id 1.
        for (int i = 0; i < 10; ++i)
        {
            _videoCapture.open(i);

            if (_videoCapture.isOpened())
            {
                SENSCameraDeviceProps characteristics(std::to_string(i), SENSCameraFacing::UNKNOWN);

                //try some standard capture sizes
                for (auto s : testSizes)
                {
                    _videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, s.width);
                    _videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, s.height);
                    cv::Mat frame;
                    _videoCapture >> frame;
                    cv::Size newSize = frame.size();
                    if (!characteristics.contains(newSize) &&
                        newSize != cv::Size(0, 0))
                    {
                        //-1 means unknown focal length
                        characteristics.add(newSize.width, newSize.height, -1.f);
                    }
                }

                _captureProperties.push_back(characteristics);
                _videoCapture.release();
            }
        }

        //if still no caputure properties add a dummy
        if (_captureProperties.size() == 0)
        {
            SENSCameraDeviceProps dummyProps("0", SENSCameraFacing::UNKNOWN);
            dummyProps.add(640, 480, -1.f);
            _captureProperties.push_back(dummyProps);
        }
    }

    return _captureProperties;
}
