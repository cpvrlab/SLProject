#include "SENSWebCamera.h"
#include <SENSException.h>
#include <SENSUtils.h>
#include <Utils.h>

#define LOG_WEBCAM_WARN(...) Utils::log("SENSWebCamera", __VA_ARGS__);
#define LOG_WEBCAM_INFO(...) Utils::log("SENSWebCamera", __VA_ARGS__);
#define LOG_WEBCAM_DEBUG(...) Utils::log("SENSWebCamera", __VA_ARGS__);

SENSWebCamera::~SENSWebCamera()
{
    if (_thread.joinable())
        _thread.join();
}

void SENSWebCamera::init(SENSCamera::Facing facing)
{
    LOG_WEBCAM_INFO("init: called but is has no effect in SENSWebCamera");
    //_state   = State::INITIALIZED;
    _started = false;
}

void SENSWebCamera::start(const Config config)
{
    //todo: wait for camera to be granted
    _config      = config;
    _targetWdivH = (float)_config.targetWidth / (float)_config.targetHeight;

    if (_thread.joinable())
        _thread.join();
    _thread = std::thread(&SENSWebCamera::openCamera, this);

    //SENSWebCamera::openCamera();
}

void SENSWebCamera::start(int width, int height)
{
    Config config;

    config.targetWidth  = width;
    config.targetHeight = height;

    start(config);
}

void SENSWebCamera::stop()
{
    if (_videoCapture.isOpened())
        _videoCapture.release();

    _started = false;
}

SENSFramePtr SENSWebCamera::getLatestFrame()
{
    SENSFramePtr sensFrame;

    if (!_started)
        return sensFrame;

    if (!_videoCapture.isOpened())
        throw SENSException(SENSType::CAM, "Capture device is not open!", __LINE__, __FILE__);

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

void SENSWebCamera::openCamera()
{
    _videoCapture.open(0);
    //if (!_videoCapture.isOpened())
    //throw SENSException(SENSType::CAM, "Could not open camera with id: " + std::to_string(0), __LINE__, __FILE__);

    if (_videoCapture.isOpened())
    {
        _videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        _videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        int w = (int)_videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)_videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    }

    _started = true;
}
