//#############################################################################
//  File:      SENSCamera.h
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SENS_CAMERA_H
#define SENS_CAMERA_H

#include <opencv2/core.hpp>
#include "SENSFrame.h"
#include "SENSException.h"
#include "SENSCalibration.h"
#include <atomic>
#include <map>
#include <thread>
#include <algorithm>
#include "SENSUtils.h"
#include <HighResTimer.h>
//-----------------------------------------------------------------------------
// Common definitions:
//-----------------------------------------------------------------------------
//! Definition of camera facing
enum class SENSCameraFacing
{
    FRONT = 0,
    BACK,
    EXTERNAL,
    UNKNOWN
};
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
//! Definition of autofocus mode
enum class SENSCameraFocusMode
{
    CONTINIOUS_AUTO_FOCUS = 0,
    FIXED_INFINITY_FOCUS,
    UNKNOWN
};
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
struct SENSCameraStreamConfig
{
    int   widthPix       = 0;
    int   heightPix      = 0;
    float focalLengthPix = -1.f; //< focal length in pixel (-1 means unknown)
};
//-----------------------------------------------------------------------------
class SENSCameraDeviceProps
{
public:
    SENSCameraDeviceProps() {}
    SENSCameraDeviceProps(const std::string& deviceId, SENSCameraFacing facing)
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
                 [&](const SENSCameraStreamConfig& cmp) -> bool
                 { return cmp.widthPix == toFind.width && cmp.heightPix == toFind.height; }) != _streamConfigs.end();
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
//-----------------------------------------------------------------------------
/*! SENSCameraConfig
    This class defines the current configuration of a camera
 */
struct SENSCameraConfig
{
    // this constructor forces the user to always define a complete parameter set. In this way no parameter is forgotten..
    SENSCameraConfig(std::string                   deviceId,
                     const SENSCameraStreamConfig& streamConfig,
                     SENSCameraFacing              facing,
                     SENSCameraFocusMode           focusMode = SENSCameraFocusMode::CONTINIOUS_AUTO_FOCUS)
      : deviceId(deviceId),
        streamConfig(streamConfig),
        facing(facing),
        focusMode(focusMode)
    {
    }
    SENSCameraConfig() = default;

    std::string            deviceId;
    SENSCameraStreamConfig streamConfig; //!< currently selected stream config index (use it to look up original capture size)
    SENSCameraFocusMode    focusMode;    //!< autofocus mode
    SENSCameraFacing       facing;       //!< camera facing
};
//-----------------------------------------------------------------------------
class SENSCaptureProps : public std::vector<SENSCameraDeviceProps>
{
public:
    bool                         containsDeviceId(const std::string& deviceId) const;
    bool                         supportsCameraFacing(const SENSCameraFacing& facing) const;
    const SENSCameraDeviceProps* camPropsForDeviceId(const std::string& deviceId) const;
    // returned pointer is null if nothing was found
    std::pair<const SENSCameraDeviceProps* const, const SENSCameraStreamConfig* const>
    findBestMatchingConfig(SENSCameraFacing facing,
                           const float      horizFov,
                           const int        width,
                           const int        height) const;
};
//-----------------------------------------------------------------------------
class SENSCameraListener
{
public:
    virtual ~SENSCameraListener() {}
    virtual void onFrame(const SENSTimePt& timePt, cv::Mat frame)      = 0;
    virtual void onCameraConfigChanged(const SENSCameraConfig& config) = 0;
};
//-----------------------------------------------------------------------------
//! Pure abstract camera class
class SENSCamera
{
public:
    virtual ~SENSCamera() {}

    /*!
    Call this function to configure the camera if you exactly know, what device you want to open and with which stream configuration.
    You can find out these properties by using SENSCaptureProps object (see getCaptureProperties())
    @param deviceId camera device id to start
    @param streamConfig SENSCameraStreamConfig from camera device. This defines the size of the image. The image is converted to BGR and assigned to SENSFrame::imgBGR.
    @param provideIntrinsics specifies if intrinsics estimation should be enabled. The estimated intrinsics are transferred with every SENSFrame as they may be different for every frame (e.g. on iOS). This value has different effects on different architectures.
    @returns the found configuration that is adjusted and used when SENSCamera::start() is called.
    */
    virtual const SENSCameraConfig& start(std::string                   deviceId,
                                          const SENSCameraStreamConfig& streamConfig,
                                          bool                          provideIntrinsics = true) = 0;

    //! Stop a started camera device
    virtual void stop() = 0;

    //! Get SENSCaptureProps which contains necessary information about all available camera devices and their capabilities
    virtual const SENSCaptureProps& captureProperties() = 0;

    //! Get the latest captured frame. If no frame was captured the frame will be empty (null).
    virtual SENSFrameBasePtr latestFrame() = 0;

    //! defines how the camera was configured during start
    virtual const SENSCameraConfig& config() const = 0;

    virtual void registerListener(SENSCameraListener* listener)   = 0;
    virtual void unregisterListener(SENSCameraListener* listener) = 0;
    virtual bool started() const                                  = 0;
    virtual bool permissionGranted() const                        = 0;
    virtual void setPermissionGranted()                           = 0;
};
//-----------------------------------------------------------------------------
//! Implementation of common functionality and members
class SENSBaseCamera : public SENSCamera
{
public:
    SENSFrameBasePtr        latestFrame() override;
    const SENSCameraConfig& config() const override { return _config; };
    void                    registerListener(SENSCameraListener* listener) override;
    void                    unregisterListener(SENSCameraListener* listener) override;
    bool                    started() const override { return _started; }
    bool                    permissionGranted() const override { return _permissionGranted; }
    void                    setPermissionGranted() override { _permissionGranted = true; }

protected:
    void updateFrame(cv::Mat bgrImg,
                     cv::Mat intrinsics,
                     int     width,
                     int     height,
                     bool    intrinsicsChanged);

    void processStart(); //!< call from start function to do startup preprocessing

    SENSCaptureProps                 _captureProperties;
    std::atomic<bool>                _started{false}; //!< flags if camera was started
    SENSCameraConfig                 _config;         //!< indicates what is currently running
    std::atomic<bool>                _permissionGranted{false};
    std::vector<SENSCameraListener*> _listeners;
    std::mutex                       _listenerMutex;
    SENSFrameBasePtr                 _frame; //!< current frame
    std::mutex                       _frameMutex;
};
//-----------------------------------------------------------------------------
#endif // SENS_CAMERA_H
