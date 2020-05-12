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
    std::vector<float> focalLenghtsMM;
    cv::Size2f         physicalSensorSizeMM;
    SENSCameraFacing   facing = SENSCameraFacing::UNKNOWN;
};

enum class SENSCameraFocusMode
{
    CONTINIOUS_AUTO_FOCUS = 0,
    FIXED_INFINITY_FOCUS
};

//define a config to start a capture session on a camera device
struct SENSCameraConfig
{
    std::string         deviceId  = "0";
    SENSCameraFocusMode focusMode = SENSCameraFocusMode::CONTINIOUS_AUTO_FOCUS;
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

class SENSCamera
{
public:
    virtual ~SENSCamera() {}
    virtual void                                   start(SENSCameraConfig config)               = 0;
    virtual void                                   start(std::string id, int width, int height) = 0;
    virtual void                                   stop()                                       = 0;
    virtual std::vector<SENSCameraCharacteristics> getAllCameraCharacteristics()                = 0;
    virtual SENSFramePtr                           getLatestFrame()                             = 0;

    virtual const SENSCameraCharacteristics& characteristics() const = 0;
    virtual const SENSCameraConfig&          config() const          = 0;
    virtual bool                             started() const         = 0;

    virtual bool permissionGranted() const = 0;
    virtual void setPermissionGranted()    = 0;
};

class SENSCameraBase : public SENSCamera
{
public:
    const SENSCameraCharacteristics& characteristics() const override { return _characteristics; }
    const SENSCameraConfig&          config() const override { return _config; };
    bool                             started() const override { return _started; }

    bool permissionGranted() const override { return _permissionGranted; }
    void setPermissionGranted() override { _permissionGranted = true; }

protected:
    float             _targetWdivH = -1.0f;
    SENSCameraConfig  _config;
    std::atomic<bool> _started{false};

    SENSCameraCharacteristics              _characteristics;
    std::vector<SENSCameraCharacteristics> _allCharacteristics;

    std::atomic<bool> _permissionGranted{false};
};

#include <thread>
/*!
The SENSCameraBase implementations may only be called from a single thread. Start and Stop will block and on the
other hand if called from different threads, calling e.g. stop while starting will lead to 
problems. 
The SENSCameraAsync adds an additional state machine layer that handles events and makes sure
that the possible states are corretly handled.
By using an additional layer, we can separate the already complex camera implementations from
the additonally complex statemachine.
The SENSCameraAsync wraps a unique pointer of SENSCameraBase. In this way we may use the same implementation
for all SENSCameraBase types.
*/
class SENSCameraAsync : public SENSCamera
{
public:
    //The SENSCameraAsync takes ownership of the camera, thats why one has to provide a unique pointer
    explicit SENSCameraAsync(std::unique_ptr<SENSCameraBase> camera)
    {
        _camera = std::move(camera);
        if (!_camera)
            throw SENSException(SENSType::CAM, "SENSCameraAsync: initialized with invalid SENSCameraBase object!", __LINE__, __FILE__);
    }

    void start(const SENSCameraConfig config) override
    {
        _camera->start(config);
    }
    void start(std::string id, int width, int height) override
    {
    }
    void stop() override
    {
        _camera->stop();
    }
    std::vector<SENSCameraCharacteristics> getAllCameraCharacteristics() override
    {
        if (_camera)
            return _camera->getAllCameraCharacteristics();
        else
            return std::vector<SENSCameraCharacteristics>();
    }

    SENSFramePtr getLatestFrame() override
    {
        SENSFramePtr frame;

        if (_camera)
        {
            frame = _camera->getLatestFrame();
        }
        else
        {
            //warn
        }
        return frame;
    }

    const SENSCameraCharacteristics& characteristics() const override
    {
        return _camera->characteristics();
    }

    const SENSCameraConfig& config() const override
    {
        return _camera->config();
    }

    bool started() const override
    {
        return _camera->started();
    }

    bool permissionGranted() const override
    {
        return _camera->permissionGranted();
    }
    void setPermissionGranted() override
    {
        _camera->setPermissionGranted();
    }

private:
    //wrapped SENSCameraBase instance
    std::unique_ptr<SENSCameraBase> _camera;

    //processing thread
    std::thread _thread;
};

#endif //SENS_CAMERA_H
