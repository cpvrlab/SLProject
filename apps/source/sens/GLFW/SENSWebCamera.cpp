#include "SENSWebCamera.h"
#include <SENSException.h>
#include <SENSUtils.h>
#include <Utils.h>

//#define LOG_WEBCAM_WARN(...) Utils::log("SENSWebCamera", __VA_ARGS__);
//#define LOG_WEBCAM_INFO(...) Utils::log("SENSWebCamera", __VA_ARGS__);
//#define LOG_WEBCAM_DEBUG(...) Utils::log("SENSWebCamera", __VA_ARGS__);
#define LOG_WEBCAM_WARN
#define LOG_WEBCAM_INFO
#define LOG_WEBCAM_DEBUG

const SENSCameraConfig& SENSWebCamera::start(std::string                   deviceId,
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
        Utils::warnMsg("SENSWebCamera", "Call to start was ignored. Camera is currently running!", __LINE__, __FILE__);
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

    cv::Size imgManipSize;
    if (_config.manipWidth > 0 && _config.manipHeight > 0)
        imgManipSize = {imgManipWidth, (int)((float)imgManipWidth * (float)targetSize.height / (float)targetSize.width)};
    else
        imgManipSize = targetSize;

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

    _videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, targetSize.width);
    _videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, targetSize.height);
    _started           = true;
    _permissionGranted = true;

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

    initCalibration(fovDegFallbackGuess);
    
    //start thread
    _cameraThread = std::thread(&SENSWebCamera::grab, this);

    return _config;
}

void SENSWebCamera::stop()
{
    _stop = true;
    if(_cameraThread.joinable())
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
    while(!_stop)
    {
        cv::Mat bgrImg;
        if (_videoCapture.read(bgrImg))
        {
            //todo: move to base class
            //inform listeners
            {
                std::unique_lock<std::mutex> lock(_listenerMutex);
                if(_listeners.size())
                {
                    lock.unlock();
                    SENSTimePt timePt = SENSClock::now();
                    cv::Mat bgrImg;
                    lock.lock();
                    //if the video writer is slower than the video feed, we have to react, otherwise there
                    //will be a buffer overflow
                    for (SENSCameraListener *l : _listeners)
                        l->onFrame(timePt, bgrImg.clone());
                }
            }
            
            SENSFramePtr sensFrame = postProcessNewFrame(bgrImg, cv::Mat(), false);
            {
                std::lock_guard<std::mutex> lock(_frameMutex);
                _sensFrame = sensFrame;
            }
        }
    }
}

SENSFramePtr SENSWebCamera::latestFrame()
{
    SENSFramePtr sensFrame;

    if (!_started)
    {
        LOG_WEBCAM_WARN("getLatestFrame: Camera is not started!");
        return sensFrame;
    }

    if (!_videoCapture.isOpened())
        throw SENSException(SENSType::CAM, "Capture device is not open although camera is started!", __LINE__, __FILE__);

    /*
    cv::Mat bgrImg;
    if (_videoCapture.read(bgrImg))
    {
        //todo: move to base class
        {
            SENSTimePt timePt = SENSClock::now();
            std::lock_guard<std::mutex> lock(_listenerMutex);
            for(SENSCameraListener* l : _listeners)
                l->onFrame(timePt, bgrImg.clone());
        }
        
        sensFrame = postProcessNewFrame(bgrImg, cv::Mat(), false);
    }
     */
    
    {
        std::lock_guard<std::mutex> lock(_frameMutex);
        sensFrame = _sensFrame;
    }
    return sensFrame;
}

const SENSCaptureProperties& SENSWebCamera::captureProperties()
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
                SENSCameraDeviceProperties characteristics(std::to_string(i), SENSCameraFacing::UNKNOWN);

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
            SENSCameraDeviceProperties dummyProps("0", SENSCameraFacing::UNKNOWN);
            dummyProps.add(640, 480, -1.f);
            _captureProperties.push_back(dummyProps);
        }
    }

    return _captureProperties;
}
