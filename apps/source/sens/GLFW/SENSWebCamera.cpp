#include "SENSWebCamera.h"
#include <SENSException.h>
#include <SENSUtils.h>
#include <Utils.h>

#define LOG_WEBCAM_WARN(...) Utils::log("SENSWebCamera", __VA_ARGS__);
#define LOG_WEBCAM_INFO(...) Utils::log("SENSWebCamera", __VA_ARGS__);
#define LOG_WEBCAM_DEBUG(...) Utils::log("SENSWebCamera", __VA_ARGS__);

void SENSWebCamera::start(const SENSCameraConfig config)
{
    if (!_videoCapture.isOpened())
    {
        _config      = config;
        _targetWdivH = (float)_config.targetWidth / (float)_config.targetHeight;

        int id = std::stoi(_config.deviceId);
        _videoCapture.open(id);

        if (!_videoCapture.isOpened())
            throw SENSException(SENSType::CAM, "Could not open camera with id: " + _config.deviceId, __LINE__, __FILE__);

        _videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, _config.targetWidth);
        _videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, _config.targetHeight);
        _started = true;
    }
    else
    {
        LOG_WEBCAM_WARN("start: ignored because camera is already open! Call stop first!");
    }
}

void SENSWebCamera::start(std::string id, int width, int height)
{
    SENSCameraConfig config;

    config.deviceId     = stoi(id);
    config.targetWidth  = width;
    config.targetHeight = height;

    start(config);
}

void SENSWebCamera::stop()
{
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

SENSFramePtr SENSWebCamera::getLatestFrame()
{
    SENSFramePtr sensFrame;

    if (!_started)
    {
        LOG_WEBCAM_WARN("getLatestFrame: Camera is not started!");
        return sensFrame;
    }

    if (!_videoCapture.isOpened())
        throw SENSException(SENSType::CAM, "Capture device is not open although camera is started!", __LINE__, __FILE__);

    cv::Mat rgbImg;
    if (_videoCapture.read(rgbImg))
    {
        //do image adjustments
        int cropW = 0, cropH = 0;
        SENS::cropImage(rgbImg, _targetWdivH, cropW, cropH);
        SENS::mirrorImage(rgbImg, _config.mirrorH, _config.mirrorV);

        cv::Mat grayImg;
        if (_config.convertToGray)
        {
            cv::cvtColor(rgbImg, grayImg, cv::COLOR_BGR2GRAY);
        }

        sensFrame = std::make_shared<SENSFrame>(
          rgbImg,
          grayImg,
          rgbImg.size().width,
          rgbImg.size().height,
          cropW,
          cropH,
          _config.mirrorH,
          _config.mirrorV);
    }
    return sensFrame;
}

std::vector<SENSCameraCharacteristics> SENSWebCamera::getAllCameraCharacteristics()
{
    //definition of standard frame sizes that we want to test for support
    static std::vector<cv::Size> testSizes = {
      {640, 360},
      {640, 480},
      {960, 540},
      {1280, 960},
      {1280, 720},
      {1920, 1080}};

    //stop running capturing
    bool deviceWasOpen = false;
    if (_videoCapture.isOpened())
    {
        stop();
        deviceWasOpen = true;
    }

    std::vector<SENSCameraCharacteristics> allCharacteristics;

    //There is an invisible list of devices populated from your os and your webcams appear there in the order you plugged them in.
    //If you're e.g on a laptop with a builtin camera, that will be id 0, if you plug in an additional one, that's id 1.
    for (int i = 0; i < 10; ++i)
    {
        _videoCapture.open(i);

        if (_videoCapture.isOpened())
        {
            SENSCameraCharacteristics characteristics;
            characteristics.cameraId = std::to_string(i);
            characteristics.provided = false;
            //try some standard capture sizes
            for (auto s : testSizes)
            {
                _videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, s.width);
                _videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, s.height);
                cv::Mat frame;
                _videoCapture >> frame;
                cv::Size newSize = frame.size();
                if (!characteristics.streamConfig.contains(newSize))
                {
                    characteristics.streamConfig.add(newSize);
                }
            }

            allCharacteristics.push_back(characteristics);
            _videoCapture.release();
        }
    }

    //dummy data to debug second camera
    //{
    //    SENSCameraCharacteristics characteristics;
    //    characteristics.cameraId = "1";
    //    characteristics.provided = false;
    //    characteristics.streamConfig.add(cv::Size(640, 654));
    //    allCharacteristics.push_back(characteristics);
    //}

    //start again with old config
    if (deviceWasOpen)
        start(_config);

    return allCharacteristics;
}
