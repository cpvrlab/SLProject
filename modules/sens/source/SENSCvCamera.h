#ifndef SENS_CV_CAMERA_H
#define SENS_CV_CAMERA_H

#include <opencv2/core.hpp>
#include <SENSFrame.h>
#include <SENSException.h>
#include <SENSCalibration.h>
#include <atomic>
#include <map>
#include <thread>
#include <algorithm>
#include <SENSUtils.h>
#include <HighResTimer.h>
#include "SENSCamera.h"

//---------------------------------------------------------------------------
//SENSCameraConfig
//define a config to start a capture session on a camera device
struct SENSCvCameraConfig
{
    //this constructor forces the user to always define a complete parameter set. In this way no parameter is forgotten..
    SENSCvCameraConfig(const SENSCameraDeviceProps*  deviceProps,
                       const SENSCameraStreamConfig* streamConfig,
                       int                           targetWidth,
                       int                           targetHeight,
                       int                           manipWidth,
                       int                           manipHeight,
                       bool                          mirrorH,
                       bool                          mirrorV,
                       bool                          convertManipToGray)
      : deviceProps(deviceProps),
        streamConfig(streamConfig),
        targetWidth(targetWidth),
        targetHeight(targetHeight),
        manipWidth(manipWidth),
        manipHeight(manipHeight),
        mirrorH(mirrorH),
        mirrorV(mirrorV),
        convertManipToGray(convertManipToGray)
    {
    }

    const SENSCameraDeviceProps*  deviceProps;
    const SENSCameraStreamConfig* streamConfig;
    //! largest target image width (only BGR)
    const int targetWidth;
    //! largest target image width (only BGR)
    const int targetHeight;
    //! width of smaller image version (e.g. for tracking)
    const int manipWidth;
    //! height of smaller image version (e.g. for tracking)
    const int manipHeight;
    //! mirror image horizontally after capturing
    const bool mirrorH;
    //! mirror image vertically after capturing
    const bool mirrorV;
    //! provide gray version of small image
    const bool convertManipToGray;
};

/*!
 Camera wrapper for computer vision applications that adds convenience functions and that makes post processing steps.
 */
class SENSCvCamera
{
public:
    //!error code definition, will be returned by configure in case of failure. See method advice() for explanation.
    enum ConfigReturnCode
    {
        SUCCESS = 0,
        WARN_TARGET_SIZE_NOT_FOUND,
        ERROR_CAMERA_INVALID,
        ERROR_CAMERA_IS_RUNNING,
        ERROR_FACING_NOT_AVAILABLE,
        ERROR_UNKNOWN
    };

    //! returns advice to an error code
    static std::string advice(const ConfigReturnCode returnCode)
    {
        switch (returnCode)
        {
            case SUCCESS: return "Camera was configured successfully";
            case WARN_TARGET_SIZE_NOT_FOUND: return "Only a smaller resolution was found (you can still start the camera)";
            case ERROR_CAMERA_INVALID: return "Camera is null";
            case ERROR_CAMERA_IS_RUNNING: return "Camera is running, you have to stop it first";
            case ERROR_FACING_NOT_AVAILABLE: return "Camera facing is not available as configured";
            case ERROR_UNKNOWN: return "Unknown error";
            default: return "Unknown return code";
        }
    }

    SENSCvCamera(SENSCamera* camera);

    /*!
     configure camera: the function checks with the wrapped camera if it could successfully configure the camera.
     */
    ConfigReturnCode configure(SENSCameraFacing facing,
                               int              targetWidth,
                               int              targetHeight,
                               int              manipWidth,
                               int              manipHeight,
                               bool             mirrorH,
                               bool             mirrorV,
                               bool             convertManipToGray);

    bool started();
    bool start();
    void stop();

    SENSFramePtr latestFrame();

    bool supportsFacing(SENSCameraFacing facing);
    //get camera pointer reference
    SENSCamera*&              cameraRef() { return _camera; }
    bool                      isConfigured() { return _config != nullptr; }
    const SENSCvCameraConfig* config() { return _config.get(); }

    //! returns  SENSCalibration if it was started (maybe a guessed one from a fovV guess). Else returns nullptr. The calibration is used for computer vision applications. So, if a manipulated image is requested (see imgManipSize in SENSCamera::start(...), SENSFrame::imgManip and SENSCameraConfig) this calibration is adjusted to fit to this image, else to the original sized image (see SENSFrame::imgBGR)
    const SENSCalibration* const calibration() const { return _calibration.get(); }
    const SENSCalibration* const calibrationManip() const { return _calibrationManip.get(); }

    //guess a calibration from what we know and update all derived calibration
    //(fallback is used if camera api defines no fovV value)
    void guessAndSetCalibration(float fovDegFallbackGuess);

    cv::Mat scaledCameraMat();

    //! Set calibration and adapt it to current image size. Camera has to be configured, before this function is called.
    void setCalibration(const SENSCalibration& calibration, bool buildUndistortionMaps);
    //! clear calibration set from outsilde with setCalibration. Only then an automatic guess will be possible.
    void clearCalibration() { _calibrationOverwrite.reset(); }

private:
    SENSFramePtr processNewFrame(const SENSTimePt& timePt, cv::Mat bgrImg, cv::Mat intrinsics, bool intrinsicsChanged);

    SENSCamera*                         _camera;
    std::unique_ptr<SENSCvCameraConfig> _config;

    //! The calibration is used for computer vision applications. This calibration is adjusted to fit to the original sized image (see SENSFrame::imgBGR and SENSCameraConfig::targetWidth, targetHeight)
    std::unique_ptr<SENSCalibration> _calibration;
    //calibration that fits to (targetWidth,targetHeight)
    std::unique_ptr<SENSCalibration> _calibrationManip;

    //this is a calibration that is set from outside. if it is valid, no guesses will be made
    std::unique_ptr<SENSCalibration> _calibrationOverwrite;
};

#endif //SENS_CV_CAMERA_H
