#ifndef SENS_CAMERA_H
#define SENS_CAMERA_H

#include <opencv2/core.hpp>
#include <SENSFrame.h>
#include <SENSException.h>
#include <SENSCalibration.h>
#include <atomic>
#include <map>

enum class SENSCameraFacing
{
    FRONT = 0,
    BACK,
    EXTERNAL,
    UNKNOWN
};

static std::string getPrintableFacing(SENSCameraFacing facing)
{
    switch (facing)
    {
        case SENSCameraFacing::FRONT: return "FRONT";
        case SENSCameraFacing::BACK: return "BACK";
        case SENSCameraFacing::EXTERNAL: return "EXTERNAL";
        default: return "UNKNOWN";
    }
}

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

    bool contains(cv::Size toFind)
    {
        return std::find_if(
                 _streamSizes.begin(),
                 _streamSizes.end(),
                 [&](const cv::Size& cmp) -> bool { return cmp == toFind; }) != _streamSizes.end();
    }

    //searches for best matching size and returns it
    cv::Size findBestMatchingSize(cv::Size requiredSize);

private:
    std::vector<cv::Size> _streamSizes;
};

struct SENSCameraCharacteristics
{
    std::string             cameraId;
    SENSCameraStreamConfigs streamConfig;
    //flags if following properties are valid (they are not available for every device)
    bool               provided = false;
    std::vector<float> focalLenghts;
    cv::Size2f         physicalSensorSizeMM;
    SENSCameraFacing   facing = SENSCameraFacing::UNKNOWN;
};

class SENSCamera
{
    friend class SENSCameraManager;

public:
    SENSCamera()
    {
        _started = false;
    }

    /*
    enum class Type
    {
        NORMAL = 0,
        MACRO,
        TELE
    };
     */

    enum class State
    {
        CLOSED,
        INITIALIZED,
        STARTING,
        STARTED,
        REPEATING_REQUEST,
        CLOSING
    };

    enum class FocusMode
    {
        CONTINIOUS_AUTO_FOCUS = 0,
        FIXED_INFINITY_FOCUS
    };

    //define a config to start a capture session on a camera device
    struct Config
    {
        std::string deviceId  = "0";
        FocusMode   focusMode = FocusMode::CONTINIOUS_AUTO_FOCUS;
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

    //virtual void init(SENSCameraFacing facing) = 0;
    virtual void                                   start(const Config config)    = 0;
    virtual void                                   start(int width, int height)  = 0;
    virtual void                                   stop()                        = 0;
    virtual std::vector<SENSCameraCharacteristics> getAllCameraCharacteristics() = 0;

    virtual SENSFramePtr getLatestFrame() = 0;

    bool     started() const { return _started; }
    cv::Size getFrameSize() { return cv::Size(_config.targetWidth, _config.targetHeight); }

    //! returns true if you can retrieve meta-data for the camera on this device (e.g. available stream frames sizes, sensor size and focal lengths)
    bool                         isCharacteristicsProvided() { return _characteristics.provided; }
    const std::vector<cv::Size>& getCharacteristicsStreamSizes() const { return _characteristics.streamConfig.getStreamSizes(); }
    const cv::Size2f&            getCharacteristicsPhysicalSensorSizeMM() { return _characteristics.physicalSensorSizeMM; }
    const std::vector<float>&    getCharacteristicsFocalLengthsMM() { return _characteristics.focalLenghts; }

    bool permissionGranted() const { return _permissionGranted; }
    void setPermissionGranted() { _permissionGranted = true; }

protected:
    float             _targetWdivH = -1.0f;
    Config            _config;
    std::atomic<bool> _started;

    SENSCameraCharacteristics _characteristics;

    std::atomic<bool> _permissionGranted{false};
};

#endif //SENS_CAMERA_H
