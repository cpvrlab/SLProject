#ifndef SENS_CAMERA_H
#define SENS_CAMERA_H

#include <opencv2/core.hpp>
#include <SENSFrame.h>
#include <SENSException.h>
#include <atomic>

class SENSCameraStreamConfigs
{
public:
    void add(cv::Size size)
    {
        _streamSizes.push_back(size);
    }

    const std::vector<cv::Size>& getStreamSizes() const
    {
        return _streamSizes;
    }

    void clear()
    {
        _streamSizes.clear();
    }

    //searches for best matching size and returns it
    cv::Size findBestMatchingSize(cv::Size requiredSize);

private:
    std::vector<cv::Size> _streamSizes;
};

class SENSCamera
{
public:
    SENSCamera()
    {
        _started           = false;
        _permissionGranted = false;
    }

    enum class Facing
    {
        FRONT = 0,
        BACK
    };

    enum class Type
    {
        NORMAL = 0,
        MACRO,
        TELE
    };

    enum class State
    {
        IDLE,
        INITIALIZED,     //!init() was called
        START_REQUESTED, //!start() and camera is asynchronously starting up
        STARTED          //!camera is giving images in requested size
    };

    enum class FocusMode
    {
        CONTINIOUS_AUTO_FOCUS = 0,
        FIXED_INFINITY_FOCUS
    };

    struct Config
    {
        FocusMode focusMode = FocusMode::CONTINIOUS_AUTO_FOCUS;
        //! largest target image width (only RGB)
        int targetWidth = 0;
        //! largest target image width (only RGB)
        int targetHeight = 0;
        //! width of smaller image version (e.g. for tracking)
        int smallWidth = 0;
        //! height of smaller image version (e.g. for tracking)
        int smallHeight = 0;
        //! mirror image horizontally after capturing
        bool mirrorH = false;
        //! mirror image vertically after capturing
        bool mirrorV = false;
        //! provide scaled (smaller) version with size (smallWidth, smallHeight)
        bool provideScaledImage = false;
        //! provide gray version of small image
        bool convertToGray = false;
        //! adjust image in asynchronous thread
        bool adjustAsynchronously = false;
    };

    virtual void init(SENSCamera::Facing facing) = 0;
    virtual void start(const Config config)      = 0;
    virtual void start(int width, int height)    = 0;
    virtual void stop()                          = 0;

    virtual SENSFramePtr getLatestFrame() = 0;

    bool     started() const { return _started; }
    bool     permissionGranted() const { return _permissionGranted; }
    void     setPermissionGranted() { _permissionGranted = true; }
    cv::Size getFrameSize() { return cv::Size(_config.targetWidth, _config.targetHeight); }

    //! returns true if you can retrieve meta-data for the camera on this device (e.g. available stream frames sizes, sensor size and focal lengths)
    bool                         isCamInfoProvided() { return _camInfoProvided; }
    const std::vector<cv::Size>& getCamInfoStreamSizes() const { return _camInfoAvailableStreamConfig.getStreamSizes(); }
    const cv::Size2f&            getCamInfoPhysicalSensorSizeMM() { return _camInfoPhysicalSensorSizeMM; }

protected:
    float             _targetWdivH = -1.0f;
    Config            _config;
    std::atomic<bool> _started;
    std::atomic<bool> _permissionGranted;

    SENSCamera::Facing _facing = SENSCamera::Facing::BACK;

    bool                    _camInfoProvided = false;
    SENSCameraStreamConfigs _camInfoAvailableStreamConfig;
    std::vector<float>      _camInfoFocalLenghts;
    cv::Size2f              _camInfoPhysicalSensorSizeMM;

    //State _state = State::IDLE;
};

#endif //SENS_CAMERA_H
