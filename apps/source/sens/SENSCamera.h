#ifndef SENS_CAMERA_H
#define SENS_CAMERA_H

#include <opencv2/core.hpp>
#include <SENSFrame.h>
#include <SENSException.h>
#include <SENSCalibration.h>
#include <atomic>
#include <map>
#include <thread>
#include <SENSUtils.h>
//---------------------------------------------------------------------------
//Common defininitions:

//! Definition of camera facing
enum class SENSCameraFacing
{
    FRONT = 0,
    BACK,
    EXTERNAL,
    UNKNOWN
};

//! mapping of SENSCameraFacing to a readable string
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

//! Definition of autofocus mode
enum class SENSCameraFocusMode
{
    CONTINIOUS_AUTO_FOCUS = 0,
    FIXED_INFINITY_FOCUS
};

//! mapping of SENSCameraFocusMode to a readable string
static std::string getPrintableFocusMode(SENSCameraFocusMode focusMode)
{
    switch (focusMode)
    {
        case SENSCameraFocusMode::CONTINIOUS_AUTO_FOCUS: return "CONTINIOUS_AUTO_FOCUS";
        case SENSCameraFocusMode::FIXED_INFINITY_FOCUS: return "FIXED_INFINITY_FOCUS";
        default: return "UNKNOWN";
    }
}

//---------------------------------------------------------------------------
//Stream configuration (this is what the camera device is capable to do)

//!Available stream configurations
class SENSCameraStreamConfigs
{
public:
    struct Config
    {
        int widthPix = 0;
        int heightPix = 0;
        //focal length in pixel (-1 means unknown)
        float focalLengthPix = -1.f;
        //todo: min max frame rate
    };
       
    void add(int widthPix, int heightPix, float focalLengthPix)
    {
        Config config;
        config.widthPix = widthPix;
        config.heightPix = heightPix;
        config.focalLengthPix = focalLengthPix;
        _streamConfigs.push_back(config);
    }

    const std::vector<Config>& getStreamConfigs() const
    {
        return _streamConfigs;
    }

    void clear()
    {
        _streamConfigs.clear();
    }

    bool contains(cv::Size toFind)
    {
        return std::find_if(
                 _streamConfigs.begin(),
                 _streamConfigs.end(),
                 [&](const Config& cmp) -> bool {return cmp.widthPix == toFind.width && cmp.heightPix == toFind.height; }
                            ) != _streamConfigs.end();
    }

    //searches for best matching size and returns it
    const SENSCameraStreamConfigs::Config& findBestMatchingConfig(cv::Size requiredSize) const;
    
private:
    std::vector<Config> _streamConfigs;
};

/*
struct SENSCameraStreamConfig
{
    SENSCameraStreamConfig(int width, int height, float horizFovDeg)
    {
    }
    const int width;
    const int height;
    const float horizFovDeg;
    const float focalLengthPix;
    //const float minFrameDuraton;
    //const float maxFrameDuration;
};

struct SENSCameraProperties
{
    std::string deviceId;
    bool supportsAutofocus;
    bool supportsVideoStabilization;
    SENSCameraFacing facing;
};

//Class that defines all properties of available cameras on a device and
//functions to find the required configuration
class SENSCaptureProperties
{
public:
    bool containsPropertiesForId(const std::string& deviceId)
    {
        return _camProperties.find(deviceId) != _camProperties.end();
    }
    SENSCameraProperties getPropertiesForId(const std::string& deviceId)
    {
        return _camProperties[deviceId];
    }
    
    bool contains(const std::string& deviceId, int width, int height)
    {
        auto it = _streamConfigs.find(deviceId);
        if(it != _streamConfigs.end())
        {
            const std::vector<SENSCameraStreamConfig>& scs = it->second;
            return std::find_if(
                     scs.begin(),
                     scs.end(),
                     [&](const SENSCameraStreamConfig& cmp) -> bool {return cmp.width == width && cmp.height == height; }
                                ) != scs.end();
        }
        else
            return false;
    }
    
private:
    //cameras available stream configurations mapped by
    std::map<std::string, std::vector<SENSCameraStreamConfig>> _streamConfigs;
    //camera properties mapped by camera device id
    std::map<std::string, SENSCameraProperties> _camProperties;
};
 */

struct SENSCameraCharacteristics
{
    std::string             cameraId;
    SENSCameraStreamConfigs streamConfig;
    //flags if following properties are valid (they are not available for every device)
    //bool               provided = false;
    //std::vector<float> focalLenghtsMM;
    //cv::Size2f         physicalSensorSizeMM;
    SENSCameraFacing   facing = SENSCameraFacing::UNKNOWN;
};

class SENSCaptureProperties : public std::vector<SENSCameraCharacteristics>
{
public:
    /*
    struct Device
    {
        
    };
    struct StreamConfig
    {
        
    };
     */
    
private:
};
 

//---------------------------------------------------------------------------
//SENSCameraConfig (this is what the user would like to have)

//define a config to start a capture session on a camera device
struct SENSCameraConfig
{
    std::string deviceId;
    //! currently selected stream config index (use it to look up original capture size)
    int streamConfigIndex = 0;
    //! autofocus mode
    SENSCameraFocusMode focusMode = SENSCameraFocusMode::CONTINIOUS_AUTO_FOCUS;
    //! largest target image width (only RGB)
    int targetWidth = 0;
    //! largest target image width (only RGB)
    int targetHeight = 0;
    //! current horizontal field of view (-1 if unknown)
    float horizFovDeg = -1.f;
    
    //! width of smaller image version (e.g. for tracking)
    int manipWidth = 0;
    //! height of smaller image version (e.g. for tracking)
    int manipHeight = 0;
    //! mirror image horizontally after capturing
    bool mirrorH = false;
    //! mirror image vertically after capturing
    bool mirrorV = false;
    //! provide scaled (smaller) version with size (smallWidth, smallHeight)
    bool provideScaledImage = false;
    //! provide gray version of small image
    bool convertManipToGray = false;
    
    //! adjust image in asynchronous thread
    bool adjustAsynchronously = false;
    
    //! enable video stabilization if available
    //bool enableVideoStabilization = false;
};

//! Pure abstract camera class
class SENSCamera
{
public:
    virtual ~SENSCamera() {}
    
    //! enable mirror manipulation of input image (vertically, horizontally) (you should call this function before start)
    virtual void mirrorImage(bool mirrorV, bool mirrorH) = 0;
    //! provide scaled image with given size (will result in a resized gray image) (you should call this function before start)
    virtual void provideScaledImage(int width, int height) = 0;
    //! provide gray image (maybe scaled, if provideScaledImage was called before) (you should call this function before start)
    virtual void provideGrayImage(bool convertToGray) = 0;
    
    //virtual void            start(SENSCameraConfig config)               = 0;
    //! Start camera with a known device id. The camera will select the closest available frame size and crop it to width and height
    virtual void start(std::string id, int width, int height, SENSCameraFocusMode focusMode=SENSCameraFocusMode::FIXED_INFINITY_FOCUS) = 0;
    //virtual void start(std::string id, SENSCameraStreamConfigs streamConfig, int width, int height, SENSCameraFocusMode focusMode=SENSCameraFocusMode::FIXED_INFINITY_FOCUS) = 0;
    //! Stop a started camera device
    virtual void stop()                                       = 0;
    //! Get the latest captured frame. If no frame was captured the frame will be empty (null).
    virtual SENSFramePtr getLatestFrame()                             = 0;
    //! Get SENSCaptureProperties which contains necessary information about all available camera devices and their capabilities
    virtual const SENSCaptureProperties& getCaptureProperties() = 0;

    //! defines what the currently selected camera is cabable to do (including all available camera devices)
    //virtual const SENSCameraCharacteristics& characteristics() const = 0;
    //! defines how the camera was configured during start
    virtual const SENSCameraConfig& config() const = 0;
    //!  currently selected stream configuration
    //virtual const SENSCameraStreamConfig& currSteamConfig() const = 0;
    //virtual const SENSCameraProperties& currCameraProperties() const = 0;
    
    virtual bool started() const         = 0;

    virtual bool permissionGranted() const = 0;
    virtual void setPermissionGranted()    = 0;
};

//! Implementation of common functionality and members
class SENSCameraBase : public SENSCamera
{
public:
    //const SENSCameraCharacteristics& characteristics() const override { return _characteristics; }
    const SENSCameraConfig& config() const override { return _config; };
    //const SENSCameraStreamConfig& currSteamConfig() const override { return _currStreamConfig; }
    //const SENSCameraProperties& currCameraProperties() const override { return _currCamProps; }
    
    bool started() const override { return _started; }

    bool permissionGranted() const override { return _permissionGranted; }
    void setPermissionGranted() override { _permissionGranted = true; }
  

    //! enable mirror manipulation of input image (vertically, horizontally)
    void mirrorImage(bool mirrorV, bool mirrorH) override
    {
        _config.mirrorV = mirrorV;
        _config.mirrorH = mirrorH;
    }
    //! provide scaled image with given size (will result in a resized gray image)
    void provideScaledImage(int width, int height) override
    {
        _config.manipWidth = width;
        _config.manipHeight = height;
        _config.provideScaledImage = true;
    }
    //! provide gray image (maybe scaled, if provideScaledImage was called before)
    void provideGrayImage(bool convertToGray) override
    {
        _config.convertManipToGray = convertToGray;
    }
protected:
    SENSCaptureProperties _caputureProperties;
    
    SENSFramePtr postProcessNewFrame(cv::Mat& rgbImg);
    //! current camera device id (look up in _captureProperties)
    //std::string         _currDeviceId;
    //! current index in stream configuration vector respecting camera device id (look up in _captureProperties)
    //int                 _currStreamConfigIndex = -1;
    //! currently adjusted focus mode
    //SENSCameraFocusMode _currFocusMode;
    
    //float             _targetWdivH = -1.0f;
    
    //! flags if camera was started
    std::atomic<bool> _started{false};

    SENSCameraConfig _config;
    //SENSCameraProperties _currCamProps;
    //SENSCameraStreamConfig _currStreamConfig;
    
    //SENSCameraCharacteristics              _characteristics;
    //! stores all camera characteristics of all devices that are available.
    //std::vector<SENSCameraCharacteristics> _allCharacteristics;
    //! current stream configuration
    //SENSCameraStreamConfigs::Config _currStreamConfig;
    
    std::atomic<bool> _permissionGranted{false};
};

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
/*
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

    const SENSCameraConfig& config() const override
    {
        return _camera->config();
    }
    
    const SENSCameraStreamConfig& currSteamConfig() const override
    {
        return _camera->currSteamConfig();
    }
    
    const SENSCameraProperties& currCameraProperties() const override
    {
        return _camera->currCameraProperties();
        
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
 */

#endif //SENS_CAMERA_H
