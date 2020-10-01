#ifndef SENS_CAMERA_H
#define SENS_CAMERA_H

#include <opencv2/core.hpp>
#include <SENSFrame.h>
#include <SENSException.h>
#include <SENSCalibration.h>
#include <atomic>
#include <map>
#include <thread>
#include <algorithm>
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
    FIXED_INFINITY_FOCUS,
    UNKNOWN
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
struct SENSCameraStreamConfig
{
    int widthPix  = 0;
    int heightPix = 0;
    //focal length in pixel (-1 means unknown)
    float focalLengthPix = -1.f;
    //todo: min max frame rate
    //bool intrinsicsProvided = false;
    //float minFrameRate = 0.f;
    //float maxFrameRate = 0.f;
};

class SENSCameraDeviceProperties
{
public:
    SENSCameraDeviceProperties()
    {
    }

    SENSCameraDeviceProperties(const std::string& deviceId, SENSCameraFacing facing)
      : _deviceId(deviceId),
        _facing(facing)
    {
    }

    int                                        findBestMatchingConfig(cv::Size requiredSize) const;
    const std::vector<SENSCameraStreamConfig>& streamConfigs() const { return _streamConfigs; }
    const std::string&                         deviceId() const { return _deviceId; }
    const SENSCameraFacing&                    facing() const { return _facing; }

    bool contains(cv::Size toFind)
    {
        return std::find_if(
                 _streamConfigs.begin(),
                 _streamConfigs.end(),
                 [&](const SENSCameraStreamConfig& cmp) -> bool { return cmp.widthPix == toFind.width && cmp.heightPix == toFind.height; }) != _streamConfigs.end();
    }

    void add(int widthPix, int heightPix, float focalLengthPix)
    {
        SENSCameraStreamConfig config;
        config.widthPix       = widthPix;
        config.heightPix      = heightPix;
        config.focalLengthPix = focalLengthPix;
        _streamConfigs.push_back(config);
    }

private:
    std::string                         _deviceId;
    SENSCameraFacing                    _facing = SENSCameraFacing::UNKNOWN;
    std::vector<SENSCameraStreamConfig> _streamConfigs;
};

//---------------------------------------------------------------------------
//SENSCameraConfig
//define a config to start a capture session on a camera device
struct SENSCameraConfig
{
    //this constructor forces the user to always define a complete parameter set. In this way no parameter is forgotten..
    SENSCameraConfig(std::string                   deviceId,
                     const SENSCameraStreamConfig& streamConfig,
                     SENSCameraFocusMode           focusMode          = SENSCameraFocusMode::CONTINIOUS_AUTO_FOCUS,
                     int                           targetWidth        = 0,
                     int                           targetHeight       = 0,
                     int                           manipWidth         = 0,
                     int                           manipHeight        = 0,
                     bool                          mirrorH            = false,
                     bool                          mirrorV            = false,
                     bool                          convertManipToGray = false)
      : deviceId(deviceId),
        streamConfig(streamConfig),
        focusMode(focusMode),
        targetWidth(targetWidth),
        targetHeight(targetHeight),
        manipWidth(manipWidth),
        manipHeight(manipHeight),
        mirrorH(mirrorH),
        mirrorV(mirrorV),
        convertManipToGray(convertManipToGray)
    {
    }
    SENSCameraConfig() = default;

    //SENSCameraConfig& operator=(const SENSCameraConfig& other) = default; // Copy assignment operator
    //SENSCameraConfig& operator=(SENSCameraConfig&&) = default;            // Move assignment operator
    //SENSCameraConfig(const SENSCameraConfig&) = default; // Copy constructor
    //SENSCameraConfig(SENSCameraConfig&&)      = default; // Move constructor
    //virtual ~SENSCameraConfig() {}                       // Destructor

    std::string deviceId;
    //! currently selected stream config index (use it to look up original capture size)
    //int streamConfigIndex = -1;
    SENSCameraStreamConfig streamConfig;
    //! autofocus mode
    SENSCameraFocusMode focusMode;
    //! largest target image width (only BGR)
    int targetWidth;
    //! largest target image width (only BGR)
    int targetHeight;
    //! width of smaller image version (e.g. for tracking)
    int manipWidth;
    //! height of smaller image version (e.g. for tracking)
    int manipHeight;
    //! mirror image horizontally after capturing
    bool mirrorH;
    //! mirror image vertically after capturing
    bool mirrorV;
    //! provide gray version of small image
    bool convertManipToGray;
};

class SENSCaptureProperties : public std::vector<SENSCameraDeviceProperties>
{
public:
    bool                              containsDeviceId(const std::string& deviceId) const;
    const SENSCameraDeviceProperties* camPropsForDeviceId(const std::string& deviceId) const;
    //returned pointer is null if nothing was found
    std::pair<const SENSCameraDeviceProperties* const, const SENSCameraStreamConfig* const> findBestMatchingConfig(SENSCameraFacing facing, const float horizFov, const int width, const int height) const;
};

class SENSCameraListener
{
public:
    virtual ~SENSCameraListener() {}
    virtual void onFrame(const SENSTimePt& timePt, cv::Mat frame)         = 0;
    virtual void onCalibrationChanged(const SENSCalibration& calibration) = 0;
};

//! Pure abstract camera class
class SENSCamera
{
public:
    virtual ~SENSCamera() {}

    /*!/
    Call configure functions to configure the camera before you call start. It will try to configure the camera to your needs or tries to find the best possible solution. It will return the currently found best configuration (SENSCameraConfig)
    SENSCamera is always configured (if possible). One can retrieve the current configuration calling SENSCamera::config().
    Configure functions can only be called when the camera is stopped.
    Configure functions will throw an exception if something goes wrong, e.g. if no camera device was found.
    */

    /*!
    Call this function to configure the camera if you exactly know, what device you want to open and with which stream configuration.
    You can find out these properties by using SENSCaptureProperties object (see getCaptureProperties())
    @param deviceId camera device id to start
    @param streamConfig SENSCameraStreamConfig from camera device. This defines the size of the image. The image is converted to BGR and assigned to SENSFrame::imgBGR.
    @param imgBGRSize specifies the size of retrieved camera image. The input img is cropped and scaled to fit to imgBGRSize. If values are (0,0) this parameter is ignored and original size is used. The result is assigned to SENSFrame::imgBGR.
    @param mirrorV  enable mirror manipulation of input image vertically
    @param mirrorV  enable mirror manipulation of input image horizontally
    @param convToGrayToImgManip specifies if a gray converted version of SENSFrame::imgBGR should be calculated and assigned to SENSFrame::imgManip.
    @param imgManipWidth specifies the width of SENSFrame::imgManip. If convToGrayToImgManip is true, the gray image is resized and cropped to fit to imgManipWidth and the aspect ratio of imgBGRSize. Otherwise a scaled and cropped version of imgBGR is calculated and assigned to SENSFrame::imgManip.
    @param provideIntrinsics specifies if intrinsics estimation should be enabled. The estimated intrinsics are transferred with every SENSFrame as they may be different for every frame (e.g. on iOS). This value has different effects on different architectures. On android it will fix the autofocus to infinity and the intrinsics will be calculated using the focal length and the sensor size. Luckily on iOS intrinsics are provided even with autofocus. On desktop we will provide a guess using a manually defined fov guess.
    @param fovDegFallbackGuess fallback field of view in degree guess if no camera intrinsics can be made via camera api values.
    @returns the found configuration that is adjusted and used when SENSCamera::start() is called.
    */
    virtual const SENSCameraConfig& start(std::string                   deviceId,
                                          const SENSCameraStreamConfig& streamConfig,
                                          cv::Size                      imgBGRSize           = cv::Size(),
                                          bool                          mirrorV              = false,
                                          bool                          mirrorH              = false,
                                          bool                          convToGrayToImgManip = false,
                                          int                           imgManipWidth        = -1,
                                          bool                          provideIntrinsics    = true,
                                          float                         fovDegFallbackGuess  = 65.f) = 0;

    //! Stop a started camera device
    virtual void stop() = 0;
    //! Get the latest captured frame. If no frame was captured the frame will be empty (null).
    virtual SENSFramePtr latestFrame() = 0;
    //! Get SENSCaptureProperties which contains necessary information about all available camera devices and their capabilities
    virtual const SENSCaptureProperties& captureProperties() = 0;
    //! defines what the currently selected camera is cabable to do (including all available camera devices)
    //virtual const SENSCameraDeviceProperties& characteristics() const = 0;
    //! defines how the camera was configured during start
    virtual const SENSCameraConfig& config() const = 0;
    //! returns  SENSCalibration if it was started (maybe a guessed one from a fov guess). Else returns nullptr. The calibration is used for computer vision applications. So, if a manipulated image is requested (see imgManipSize in SENSCamera::start(...), SENSFrame::imgManip and SENSCameraConfig) this calibration is adjusted to fit to this image, else to the original sized image (see SENSFrame::imgBGR)
    virtual const SENSCalibration* const calibration() const = 0;
    //! Set calibration and adapt it to current image size. Camera has to be started, before this function is called.
    virtual void setCalibration(SENSCalibration calibration, bool buildUndistortionMaps) = 0;

    virtual void registerListener(SENSCameraListener* listener)   = 0;
    virtual void unregisterListener(SENSCameraListener* listener) = 0;

    virtual bool started() const = 0;

    virtual bool permissionGranted() const = 0;
    virtual void setPermissionGranted()    = 0;
};

//! Implementation of common functionality and members
class SENSCameraBase : public SENSCamera
{
public:
    const SENSCameraConfig& config() const override { return _config; };

    bool started() const override { return _started; }

    bool permissionGranted() const override { return _permissionGranted; }
    void setPermissionGranted() override { _permissionGranted = true; }

    const SENSCalibration* const calibration() const override
    {
        return _calibration.get();
    }

    void setCalibration(SENSCalibration calibration, bool buildUndistortionMaps) override;

    void registerListener(SENSCameraListener* listener) override;
    void unregisterListener(SENSCameraListener* listener) override;

protected:
    void         initCalibration(float fovDegFallbackGuess);
    SENSFramePtr postProcessNewFrame(cv::Mat& bgrImg, cv::Mat intrinsics, bool intrinsicsChanged);

    SENSCaptureProperties _captureProperties;

    //! flags if camera was started
    std::atomic<bool> _started{false};
    SENSCameraConfig  _config;
    std::atomic<bool> _permissionGranted{false};
    //! The calibration is used for computer vision applications. This calibration is adjusted to fit to the original sized image (see SENSFrame::imgBGR and SENSCameraConfig::targetWidth, targetHeight)
    std::unique_ptr<SENSCalibration> _calibration;

    std::vector<SENSCameraListener*> _listeners;
    std::mutex                       _listenerMutex;
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
