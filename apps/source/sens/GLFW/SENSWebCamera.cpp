#include "SENSWebCamera.h"
#include <SENSException.h>
#include <SENSUtils.h>
#include <Utils.h>

#define LOG_WEBCAM_WARN(...) Utils::log("SENSWebCamera", __VA_ARGS__);
#define LOG_WEBCAM_INFO(...) Utils::log("SENSWebCamera", __VA_ARGS__);
#define LOG_WEBCAM_DEBUG(...) Utils::log("SENSWebCamera", __VA_ARGS__);

SENSWebCamera::SENSWebCamera()
{
}

SENSWebCamera::~SENSWebCamera()
{
    if (_thread.joinable())
        _thread.join();
}

//void SENSWebCamera::init(SENSCameraFacing facing)
//{
//    LOG_WEBCAM_INFO("init: called but is has no effect in SENSWebCamera");
//    //_state   = State::INITIALIZED;
//    _started = false;
//}

void SENSWebCamera::start(const Config config)
{
    _config      = config;
    _targetWdivH = (float)_config.targetWidth / (float)_config.targetHeight;

    if (!_started && !_isStarting)
    {
        if (_thread.joinable())
            _thread.join();

        _isStarting = true;
        _thread     = std::thread(&SENSWebCamera::openCamera, this);
    }
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
        int addW = 0, addH = 0;
        //float targetScreenWdivH = 640.f / 360.f;
        //SENS::extendWithBars(rgbImg, targetScreenWdivH, cv::BORDER_CONSTANT /*cv::BORDER_REPLICATE*/, addW, addH);

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
    if (!_videoCapture.isOpened())
    {
        int id = std::stoi(_config.deviceId);
        _videoCapture.open(id);
    }

    //exception not possible as we are in a thread
    //if (!_videoCapture.isOpened())
    //throw SENSException(SENSType::CAM, "Could not open camera with id: " + std::to_string(0), __LINE__, __FILE__);

    if (_videoCapture.isOpened())
    {
        _videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, _config.targetWidth);
        _videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, _config.targetHeight);
        //int w = (int)_videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
        //int h = (int)_videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    }

    _started    = true;
    _isStarting = false;
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
        if (_videoCapture.open(i))
        {
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
    }

    //start again with old config
    if (deviceWasOpen)
        start(_config);

    return allCharacteristics;
}
