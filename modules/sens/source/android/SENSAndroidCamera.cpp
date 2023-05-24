//#############################################################################
//  File:      SENSAndroidCamera.cpp
//  Authors:   Michael Goettlicher, Luc Girod, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <iostream>
#include <string>
#include <utility>
#include <algorithm>
#include "SENSAndroidCamera.h"
#include "SENSException.h"

#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <Utils.h>
#include <HighResTimer.h>
#include "SENSAndroidCameraUtils.h"
#include "SENSUtils.h"

#define LOG_NDKCAM_WARN(...) Utils::log("SENSAndroidCamera", __VA_ARGS__);
#define LOG_NDKCAM_INFO(...) Utils::log("SENSAndroidCamera", __VA_ARGS__);
#define LOG_NDKCAM_DEBUG(...) Utils::log("SENSAndroidCamera", __VA_ARGS__);
/*
 * Camera Manager Listener object
 */
void onCameraAvailable(void* ctx, const char* id)
{
    reinterpret_cast<SENSAndroidCamera*>(ctx)->onCameraStatusChanged(id, true);
}
void onCameraUnavailable(void* ctx, const char* id)
{
    reinterpret_cast<SENSAndroidCamera*>(ctx)->onCameraStatusChanged(id, false);
}

/*
 * CameraDevice callbacks
 */
void onDeviceDisconnected(void* ctx, ACameraDevice* dev)
{
    reinterpret_cast<SENSAndroidCamera*>(ctx)->onDeviceDisconnected(dev);
}

void onDeviceErrorChanges(void* ctx, ACameraDevice* dev, int err)
{
    reinterpret_cast<SENSAndroidCamera*>(ctx)->onDeviceError(dev, err);
}

// CaptureSession state callbacks
void onSessionClosed(void* ctx, ACameraCaptureSession* ses)
{
    LOG_NDKCAM_WARN("onSessionClosed: CaptureSession state: session %p closed", ses);
    reinterpret_cast<SENSAndroidCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::CLOSED);
}

void onSessionReady(void* ctx, ACameraCaptureSession* ses)
{
    LOG_NDKCAM_WARN("onSessionReady: CaptureSession state: session %p ready", ses);
    reinterpret_cast<SENSAndroidCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::READY);
}

void onSessionActive(void* ctx, ACameraCaptureSession* ses)
{
    LOG_NDKCAM_WARN("onSessionActive: CaptureSession state: session %p active", ses);
    reinterpret_cast<SENSAndroidCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::ACTIVE);
}

SENSAndroidCamera::SENSAndroidCamera()
  : _cameraDeviceOpened(false)
{
    LOG_NDKCAM_INFO("Camera instantiated");
}

SENSAndroidCamera::~SENSAndroidCamera()
{
    //stop();
    LOG_NDKCAM_INFO("~SENSAndroidCamera: Camera destructor finished");
}

/**
 * ImageReader listener: called by AImageReader for every frame captured
 * We pass the event to ImageReader class, so it could do some housekeeping
 * about
 * the loaded queue. For example, we could keep a counter to track how many
 * buffers are full and idle in the queue. If camera almost has no buffer to
 * capture
 * we could release ( skip ) some frames by AImageReader_getNextImage() and
 * AImageReader_delete().
 */
void onImageCallback(void* ctx, AImageReader* reader)
{
    reinterpret_cast<SENSAndroidCamera*>(ctx)->imageCallback(reader);
}

//start camera selected in initOptimalCamera as soon as it is available
void SENSAndroidCamera::openCamera()
{
    //init camera manager
    if (!_cameraManager)
    {
        //init availability
        for (const SENSCameraDeviceProps& c : _captureProperties)
        {
            _cameraAvailability[c.deviceId()] = false;
        }

        LOG_NDKCAM_DEBUG("openCamera: Creating camera manager ...");
        _cameraManager = ACameraManager_create();
        if (!_cameraManager)
            throw SENSException(SENSType::CAM, "Could not instantiate camera manager!", __LINE__, __FILE__);

        //register callbacks
        _cameraManagerAvailabilityCallbacks = {
          .context             = this,
          .onCameraAvailable   = onCameraAvailable,
          .onCameraUnavailable = onCameraUnavailable,
        };
        ACameraManager_registerAvailabilityCallback(_cameraManager,
                                                    &_cameraManagerAvailabilityCallbacks);

        //Attention: if we never access the _cameraManager the onCameraStatusChanged never comes (seems to be an android bug)
        {
            ACameraIdList*  cameraIds = nullptr;
            camera_status_t status    = ACameraManager_getCameraIdList(_cameraManager, &cameraIds);
            ACameraManager_deleteCameraIdList(cameraIds);
            //PrintCameras(_cameraManager);
        }
        LOG_NDKCAM_DEBUG("openCamera: Camera manager created!");
    }

    //find current SENSCameraDeviceProps
    const SENSCameraDeviceProps* camProps = _captureProperties.camPropsForDeviceId(_config.deviceId);

    if (!_cameraDeviceOpened)
    {
        LOG_NDKCAM_DEBUG("openCamera: Camera device not open");
        auto condition = [&]
        {
            LOG_NDKCAM_DEBUG("openCamera: checking condition");
            return (_cameraAvailability[camProps->deviceId()]);
        };
        std::unique_lock<std::mutex> lock(_cameraAvailabilityMutex);
        //wait here before opening the required camera device until it is available
        _openCameraCV.wait(lock, condition);

        LOG_NDKCAM_DEBUG("openCamera: Opening camera ...");
        //open the so found camera with _characteristics.cameraId
        ACameraDevice_stateCallbacks cameraDeviceListener = {
          .context        = this,
          .onDisconnected = ::onDeviceDisconnected,
          .onError        = ::onDeviceErrorChanges,
        };

        camera_status_t cameraState;
        int             n    = 0;
        int             nMax = 10;
        while (n < nMax)
        {
            cameraState = ACameraManager_openCamera(_cameraManager,
                                                    camProps->deviceId().c_str(),
                                                    &cameraDeviceListener,
                                                    &_cameraDevice);

            if (cameraState == ACAMERA_OK)
                break;
            n++;
        }

        if (cameraState != ACAMERA_OK)
        {
            throw SENSException(SENSType::CAM, "Could not camera camera!", __LINE__, __FILE__);
        }
        else
        {
            _cameraDeviceOpened = true;
            LOG_NDKCAM_DEBUG("openCamera: Camera opened!");
        }
    }
    else
    {
        LOG_NDKCAM_DEBUG("openCamera: Camera device is already open");
    }

    const auto& streamConfig = _config.streamConfig;
    LOG_NDKCAM_INFO("openCamera: CaptureSize (%d, %d)", streamConfig.widthPix, streamConfig.heightPix);

    if (_imageReader && _captureSize != cv::Size(streamConfig.widthPix, streamConfig.heightPix))
    {
        LOG_NDKCAM_INFO("openCamera: ImageReader valid and captureSize does not fit");
        //stop repeating request and wait for stopped state
        if (_captureSession)
        {
            LOG_NDKCAM_DEBUG("openCamera: Stopping repeating request...");
            //if (_captureSessionState == CaptureSessionState::ACTIVE)
            //{
            ACameraCaptureSession_stopRepeating(_captureSession);

            auto condition = [&]
            {
                return (_captureSessionState != CaptureSessionState::ACTIVE);
            };
            std::unique_lock<std::mutex> lock(_captureSessionStateMutex);
            //wait here until capture session is stopped
            _captureSessionStateCV.wait(lock, condition);
            //}
            //else
            //    LOG_NDKCAM_WARN("CaptureSessionState NOT ACTIVE");
            LOG_NDKCAM_DEBUG("openCamera: Repeating request stopped!");

            //LOG_NDKCAM_DEBUG("stop: closing capture session...");
            //todo: it is recommended not to close before creating a new session
            //ACameraCaptureSession_close(_captureSession);
            //_captureSession = nullptr;
        }

        LOG_NDKCAM_DEBUG("openCamera: Free request stuff...");
        if (_captureRequest)
        {
            ACaptureRequest_removeTarget(_captureRequest, _cameraOutputTarget);
            ACaptureRequest_free(_captureRequest);
            _captureRequest = nullptr;
        }

        if (_captureSessionOutput)
        {
            ACaptureSessionOutputContainer_remove(_captureSessionOutputContainer,
                                                  _captureSessionOutput);
            ACaptureSessionOutput_free(_captureSessionOutput);
            _captureSessionOutput = nullptr;
        }

        if (_surface)
        {
            ANativeWindow_release(_surface);
            _surface = nullptr;
        }

        if (_captureSessionOutputContainer)
        {
            ACaptureSessionOutputContainer_free(_captureSessionOutputContainer);
            _captureSessionOutputContainer = nullptr;
        }

        if (_imageReader)
        {
            LOG_NDKCAM_DEBUG("openCamera: Deleting image reader...");
            AImageReader_delete(_imageReader);
            _imageReader = nullptr;
        }
    }

    if (_cameraDeviceOpened)
    {
        if (!_imageReader)
        {
            LOG_NDKCAM_INFO("openCamera: Creating image reader...");

            _captureSize = cv::Size(streamConfig.widthPix, streamConfig.heightPix);

            //create image reader with 2 surfaces (a surface is the like a ring buffer for images)
            if (AImageReader_new(streamConfig.widthPix, streamConfig.heightPix, AIMAGE_FORMAT_YUV_420_888, 2, &_imageReader) != AMEDIA_OK)
                throw SENSException(SENSType::CAM, "Could not create image reader!", __LINE__, __FILE__);

            //register onImageAvailable listener
            AImageReader_ImageListener listener{
              .context          = this,
              .onImageAvailable = onImageCallback,
            };
            AImageReader_setImageListener(_imageReader, &listener);

            createCaptureSession();
        }
    }
    else
    {
        //todo: throw something
    }
}

const SENSCameraConfig& SENSAndroidCamera::start(std::string                   deviceId,
                                                 const SENSCameraStreamConfig& streamConfig,
                                                 bool                          provideIntrinsics)
{
    if (_started)
    {
        Utils::warnMsg("SENSWebCamera", "Call to start was ignored. Camera is currently running!", __LINE__, __FILE__);
        return _config;
    }

    //retrieve all camera characteristics
    if (_captureProperties.size() == 0)
        captureProperties();

    if (_captureProperties.size() == 0)
        throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

    if (!_captureProperties.containsDeviceId(deviceId))
        throw SENSException(SENSType::CAM, "DeviceId does not exist!", __LINE__, __FILE__);

    SENSCameraFacing             facing = SENSCameraFacing::UNKNOWN;
    const SENSCameraDeviceProps* props  = _captureProperties.camPropsForDeviceId(deviceId);
    if (props)
        facing = props->facing();

    //init config here
    _config = SENSCameraConfig(deviceId,
                               streamConfig,
                               facing,
                               SENSCameraFocusMode::UNKNOWN);
    processStart();

    openCamera();

    _started = true;
    return _config;
}

void SENSAndroidCamera::createCaptureSession()
{
    //Get the pointer to a surface from the image reader (Surface from java is like nativeWindow in ndk)
    AImageReader_getWindow(_imageReader, &_surface);

    // Avoid surface to be deleted
    ANativeWindow_acquire(_surface);
    //create a capture session and provide the surfaces to it
    ACaptureSessionOutput_create(_surface, &_captureSessionOutput);
    //create an output container for capture session and add it to the session
    ACaptureSessionOutputContainer_create(&_captureSessionOutputContainer);
    ACaptureSessionOutputContainer_add(_captureSessionOutputContainer, _captureSessionOutput);

    ACameraOutputTarget_create(_surface, &_cameraOutputTarget);
    ACameraDevice_createCaptureRequest(_cameraDevice, TEMPLATE_PREVIEW, &_captureRequest);

    ACaptureRequest_addTarget(_captureRequest, _cameraOutputTarget);

    _captureSessionState = CaptureSessionState::READY;

    ACameraCaptureSession_stateCallbacks captureSessionStateCallbacks = {
      .context  = this,
      .onActive = ::onSessionActive,
      .onReady  = ::onSessionReady,
      .onClosed = ::onSessionClosed};
    camera_status_t captureSessionStatus = ACameraDevice_createCaptureSession(_cameraDevice,
                                                                              _captureSessionOutputContainer,
                                                                              &captureSessionStateCallbacks,
                                                                              &_captureSession);
    if (captureSessionStatus != AMEDIA_OK)
    {
        LOG_NDKCAM_WARN("Creating capture session failed!");
    }
    //throw SENSException(SENSType::CAM, "Could not create capture session!", __LINE__, __FILE__);

    //adjust capture request properties:

    //auto focus mode
    if (_config.focusMode == SENSCameraFocusMode::FIXED_INFINITY_FOCUS)
    {
        uint8_t afMode = ACAMERA_CONTROL_AF_MODE_OFF;
        ACaptureRequest_setEntry_u8(_captureRequest, ACAMERA_CONTROL_AF_MODE, 1, &afMode);
        float focusDistance = 0.0f;
        ACaptureRequest_setEntry_float(_captureRequest, ACAMERA_LENS_FOCUS_DISTANCE, 1, &focusDistance);
    }
    else
    {
        uint8_t afMode = ACAMERA_CONTROL_AF_MODE_CONTINUOUS_VIDEO;
        ACaptureRequest_setEntry_u8(_captureRequest, ACAMERA_CONTROL_AF_MODE, 1, &afMode);
    }

    //digital video stabilization (software) -> turn off by default (for now)
    {
        uint8_t mode = ACAMERA_CONTROL_VIDEO_STABILIZATION_MODE_OFF;
        ACaptureRequest_setEntry_u8(_captureRequest, ACAMERA_CONTROL_VIDEO_STABILIZATION_MODE, 1, &mode);
    }
    //optical video stabilization (hardware)
    {
        uint8_t mode = ACAMERA_LENS_OPTICAL_STABILIZATION_MODE_OFF;
        ACaptureRequest_setEntry_u8(_captureRequest, ACAMERA_LENS_OPTICAL_STABILIZATION_MODE, 1, &mode);
    }

    //install repeating request
    ACameraCaptureSession_setRepeatingRequest(_captureSession, nullptr, 1, &_captureRequest, nullptr);
}

void SENSAndroidCamera::stop()
{
    if (_started)
    {
        if (_captureSession)
        {
            LOG_NDKCAM_DEBUG("stop: stopping repeating request...");
            if (_captureSessionState == CaptureSessionState::ACTIVE)
            {
                ACameraCaptureSession_stopRepeating(_captureSession);
            }
            else
                LOG_NDKCAM_WARN("stop: CaptureSessionState NOT ACTIVE");

            LOG_NDKCAM_DEBUG("stop: closing capture session...");
            //todo: it is recommended not to close before creating a new session
            ACameraCaptureSession_close(_captureSession);
            _captureSession = nullptr;
        }

        if (_captureRequest)
        {
            LOG_NDKCAM_DEBUG("stop: free request stuff...");
            ACaptureRequest_removeTarget(_captureRequest, _cameraOutputTarget);
            ACaptureRequest_free(_captureRequest);
            _captureRequest = nullptr;
        }

        if (_captureSessionOutput)
        {
            ACaptureSessionOutputContainer_remove(_captureSessionOutputContainer,
                                                  _captureSessionOutput);
            ACaptureSessionOutput_free(_captureSessionOutput);
            _captureSessionOutput = nullptr;
        }

        if (_surface)
        {
            ANativeWindow_release(_surface);
            _surface = nullptr;
        }

        if (_captureSessionOutputContainer)
        {
            ACaptureSessionOutputContainer_free(_captureSessionOutputContainer);
            _captureSessionOutputContainer = nullptr;
        }

        if (_cameraDevice)
        {
            LOG_NDKCAM_DEBUG("stop: closing camera...");
            ACameraDevice_close(_cameraDevice);
            _cameraDevice       = nullptr;
            _cameraDeviceOpened = false;
        }

        if (_cameraManager)
        {
            LOG_NDKCAM_DEBUG("stop: deleting camera manager...");
            ACameraManager_unregisterAvailabilityCallback(_cameraManager,
                                                          &_cameraManagerAvailabilityCallbacks);
            ACameraManager_delete(_cameraManager);
            _cameraManager = nullptr;
        }

        if (_imageReader)
        {
            LOG_NDKCAM_DEBUG("stop: free image reader...");
            AImageReader_delete(_imageReader);
            _imageReader = nullptr;
        }
    }
}

cv::Mat SENSAndroidCamera::convertToYuv(AImage* image)
{
    int32_t height, width, rowStrideY;
    AImage_getHeight(image, &height);
    AImage_getWidth(image, &width);
    AImage_getPlaneRowStride(image, 0, &rowStrideY);

    //pointers to yuv data planes and length of yuv data planes in byte
    uint8_t *yPixel, /* *uPixel,*/ *vPixel;
    int32_t  yLen, /*uLen,*/ vLen;
    AImage_getPlaneData(image, 0, &yPixel, &yLen);
    //AImage_getPlaneData(image, 1, &uPixel, &uLen);
    AImage_getPlaneData(image, 2, &vPixel, &vLen);

    //Attention: There may be additional padding at the end of every line, in this case width is not equal to rowStrideY.
    //As this padding is not contained at the end of the Y-block, yLen can be calculated as follows:
    // yLen = rowStrideY * height - (rowStrideY - width)
    //But when copying the UV-block we have to "insert" the additional padding at the end of the Y-block
    //in the new yuv image!
    //(https://stackoverflow.com/questions/40030533/android-camera2-preview-output-sizes)
    //(https://stackoverflow.com/questions/52726002/camera2-captured-picture-conversion-from-yuv-420-888-to-nv21/52740776#52740776)

    //copy image data to yuv image: we use the rowStrideY to define the maximum data block width including potential padding
    cv::Mat yuv(height + (height / 2), rowStrideY, CV_8UC1);
    memcpy(yuv.data, yPixel, yLen);
    //The interleaved uv data starts with v pixels, you can inspect this by comparing uPixel and vPixel adresses,
    //which is one byte lower. So the order is V/U: NV12: YYYYUV NV21: YYYYVU
    // This is also described like this in wikipedia in section https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV420sp_(NV21)_to_RGB_conversion_(Android)
    // U follows V in the interleaved block (in contradiction to what is shown in the drawing explaining yuv in wikipedia).
    // As both planes have the same length, but one starts one byte lower, we have to copy one
    //additional byte to get all the data (see vLen+1).
    //We do not have to additionally copy the v plane. The u plane contains the interleaved u and v data!
    memcpy(yuv.data + yLen + (rowStrideY - width), vPixel, vLen + 1);

    //If there is line padding we get rid of it now by defining a sub region of interest in the target image size
    if (rowStrideY > width)
    {
        cv::Rect roi(0, 0, width, yuv.rows);
        cv::Mat  roiYuv = yuv(roi);
        return roiYuv;
    }
    else
        return yuv;
}

void SENSAndroidCamera::imageCallback(AImageReader* reader)
{
    AImage*        image  = nullptr;
    media_status_t status = AImageReader_acquireLatestImage(reader, &image);
    if (status == AMEDIA_OK && image)
    {
        cv::Mat yuv = convertToYuv(image);

        AImage_delete(image);

        cv::Mat      bgr;
        HighResTimer t;
        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21, 3);
        SENS_DEBUG("SENSAndroidCamera: time for yuv conversion: %f ms", t.elapsedTimeInMilliSec());

        updateFrame(bgr, cv::Mat(), false, bgr.cols, bgr.rows);
    }
}

/**
 * Handle Camera DeviceStateChanges msg, notify device is disconnected
 * simply close the camera
 */
void SENSAndroidCamera::onDeviceDisconnected(ACameraDevice* dev)
{
    if (dev == _cameraDevice)
    {
        std::string id(ACameraDevice_getId(dev));
        LOG_NDKCAM_WARN("device %s is disconnected", id.c_str());

        {
            std::lock_guard<std::mutex> lock(_cameraAvailabilityMutex);
            _cameraAvailability[id] = false;
        }

        _cameraDeviceOpened = false;
        ACameraDevice_close(_cameraDevice);
        _cameraDevice = nullptr;
    }
}
/**
 * Handles Camera's deviceErrorChanges message, no action;
 * mainly debugging purpose
 *
 *
 */
void SENSAndroidCamera::onDeviceError(ACameraDevice* dev, int err)
{
    if (dev == _cameraDevice)
    {
        std::string errStr;
        switch (err)
        {
            case ERROR_CAMERA_IN_USE:
                errStr = "ERROR_CAMERA_IN_USE";
                break;
            case ERROR_CAMERA_SERVICE:
                errStr = "ERROR_CAMERA_SERVICE";
                break;
            case ERROR_CAMERA_DEVICE:
                errStr = "ERROR_CAMERA_DEVICE";
                break;
            case ERROR_CAMERA_DISABLED:
                errStr = "ERROR_CAMERA_DISABLED";
                break;
            case ERROR_MAX_CAMERAS_IN_USE:
                errStr = "ERROR_MAX_CAMERAS_IN_USE";
                break;
            default:
                errStr = "Unknown Error";
        }

        std::string id(ACameraDevice_getId(dev));
        {
            std::lock_guard<std::mutex> lock(_cameraAvailabilityMutex);
            _cameraAvailability[id] = false;
        }
        _cameraDeviceOpened = false;

        LOG_NDKCAM_INFO("CameraDevice %s is in error %s", id.c_str(), errStr.c_str());
    }
}

/**
 * OnCameraStatusChanged()
 *  handles Callback from ACameraManager
 */
void SENSAndroidCamera::onCameraStatusChanged(const char* id, bool available)
{
    LOG_NDKCAM_INFO("onCameraStatusChanged: id: %s available: %s ", id, available ? "true" : "false");
    {
        std::lock_guard<std::mutex> lock(_cameraAvailabilityMutex);
        _cameraAvailability[std::string(id)] = available;
    }
    _openCameraCV.notify_one();
}

std::string getPrintableState(CaptureSessionState state)
{
    if (state == CaptureSessionState::READY) // session is ready
        return "READY";
    else if (state == CaptureSessionState::ACTIVE)
        return "ACTIVE";
    else if (state == CaptureSessionState::CLOSED)
        return "CLOSED";
    else if (state == CaptureSessionState::MAX_STATE)
        return "MAX_STATE";
    else
        return "UNKNOWN";
}
/**
 * Handles capture session state changes.
 *   Update into internal session state.
 */
void SENSAndroidCamera::onSessionState(ACameraCaptureSession* ses,
                                       CaptureSessionState    state)
{
    if (!_captureSession)
        LOG_NDKCAM_WARN("onSessionState: CaptureSession is NULL");

    if (state >= CaptureSessionState::MAX_STATE)
    {
        throw SENSException(SENSType::CAM, "Wrong state " + std::to_string((int)state), __LINE__, __FILE__);
    }

    LOG_NDKCAM_WARN("onSessionState: CaptureSession state: %s", getPrintableState(state).c_str());

    {
        std::lock_guard<std::mutex> lock(_captureSessionStateMutex);
        _captureSessionState = state;

        if (_captureSessionState == CaptureSessionState::ACTIVE)
        {
            _started = true;
        }
        else
        {
            _started = false;
        }
    }
    _captureSessionStateCV.notify_one();
}

const SENSCaptureProps& SENSAndroidCamera::captureProperties()
{
    if (_captureProperties.size() == 0)
    {
        ACameraManager* cameraManager = ACameraManager_create();
        if (!cameraManager)
            throw SENSException(SENSType::CAM, "Could not instantiate camera manager!", __LINE__, __FILE__);

        ACameraIdList* cameraIds = nullptr;
        if (ACameraManager_getCameraIdList(cameraManager, &cameraIds) != ACAMERA_OK)
            throw SENSException(SENSType::CAM, "Could not retrieve camera list!", __LINE__, __FILE__);

        for (int i = 0; i < cameraIds->numCameras; ++i)
        {
            std::string cameraId = cameraIds->cameraIds[i];

            ACameraMetadata* camCharacteristics;
            ACameraManager_getCameraCharacteristics(cameraManager, cameraId.c_str(), &camCharacteristics);

            int32_t         numEntries = 0; //will be filled by getAllTags with number of entries
            const uint32_t* tags       = nullptr;
            ACameraMetadata_getAllTags(camCharacteristics, &numEntries, &tags);

            std::vector<float> focalLengthsMM;
            cv::Size2f         physicalSensorSizeMM;
            SENSCameraFacing   facing = SENSCameraFacing::UNKNOWN;

            //make a first loop to estimate physical sensor parameters
            for (int tagIdx = 0; tagIdx < numEntries; ++tagIdx)
            {
                ACameraMetadata_const_entry lensInfo = {0};
                //first check that ACAMERA_LENS_FACING is contained at all
                if (tags[tagIdx] == ACAMERA_LENS_FACING)
                {
                    ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo);
                    acamera_metadata_enum_android_lens_facing_t androidFacing = static_cast<acamera_metadata_enum_android_lens_facing_t>(lensInfo.data.u8[0]);
                    if (androidFacing == ACAMERA_LENS_FACING_BACK)
                        facing = SENSCameraFacing::BACK;
                    else if (androidFacing == ACAMERA_LENS_FACING_FRONT)
                        facing = SENSCameraFacing::FRONT;
                    else //if (androidFacing == ACAMERA_LENS_FACING_EXTERNAL)
                        facing = SENSCameraFacing::EXTERNAL;
                }
                else if (tags[tagIdx] == ACAMERA_LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                {
                    if (ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo) ==
                        ACAMERA_OK)
                    {
                        for (int i = 0; i < lensInfo.count; ++i)
                        {
                            //characteristics.focalLenghtsMM.push_back(lensInfo.data.f[i]);
                            focalLengthsMM.push_back(lensInfo.data.f[i]);
                        }
                    }
                }
                else if (tags[tagIdx] == ACAMERA_SENSOR_INFO_PHYSICAL_SIZE)
                {
                    if (ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo) == ACAMERA_OK)
                    {
                        //characteristics.physicalSensorSizeMM.width = lensInfo.data.f[0];
                        //characteristics.physicalSensorSizeMM.height = lensInfo.data.f[1];

                        physicalSensorSizeMM.width  = lensInfo.data.f[0];
                        physicalSensorSizeMM.height = lensInfo.data.f[1];
                    }
                }
            }

            //todo: if we have more than one focal length, what do we do?
            //if we have more than one focal length, we select the first one..

            SENSCameraDeviceProps characteristics(cameraId, facing);

            //in the second loop we use the physical sensor parameters to specify a focal length in pixel for every stream config
            for (int tagIdx = 0; tagIdx < numEntries; ++tagIdx)
            {
                if (tags[tagIdx] == ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS)
                {
                    ACameraMetadata_const_entry lensInfo = {0};
                    if (ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo) == ACAMERA_OK)
                    {
                        if (lensInfo.count & 0x3)
                            throw SENSException(SENSType::CAM,
                                                "STREAM_CONFIGURATION (%d) should multiple of 4",
                                                __LINE__,
                                                __FILE__);

                        if (lensInfo.type != ACAMERA_TYPE_INT32)
                            throw SENSException(SENSType::CAM,
                                                "STREAM_CONFIGURATION TYPE(%d) is not ACAMERA_TYPE_INT32(1)",
                                                __LINE__,
                                                __FILE__);

                        int width = 0, height = 0;
                        for (uint32_t i = 0; i < lensInfo.count; i += 4)
                        {
                            //example for content interpretation:
                            //std::string direction         = lensInfo.data.i32[i + 3] ? "INPUT" : "OUTPUT";
                            //std::string format            = GetFormatStr(lensInfo.data.i32[i]);

                            //OUTPUT format and AIMAGE_FORMAT_YUV_420_888 image format
                            if (!lensInfo.data.i32[i + 3] && lensInfo.data.i32[i] == AIMAGE_FORMAT_YUV_420_888)
                            {
                                width  = lensInfo.data.i32[i + 1];
                                height = lensInfo.data.i32[i + 2];

                                float focalLengthPix = -1.f;
                                if (focalLengthsMM.size() && physicalSensorSizeMM.width > 0 && physicalSensorSizeMM.height > 0)
                                {
                                    //we assume the image is cropped at one side only. we compare the sensor aspect ratio
                                    //with the image aspect ratio and use the uncropped length to estimate a focal length in pixel that fits to this stream configuration size
                                    if ((float)physicalSensorSizeMM.width / (float)physicalSensorSizeMM.height > (float)width / (float)height)
                                        focalLengthPix = focalLengthsMM.front() / physicalSensorSizeMM.height * (float)height;
                                    else
                                        focalLengthPix = focalLengthsMM.front() / physicalSensorSizeMM.width * (float)width;
                                }

                                if (!characteristics.contains({width, height}))
                                    characteristics.add(width, height, focalLengthPix);
                            }
                        }
                    }
                }
            }
            ACameraMetadata_free(camCharacteristics);
            _captureProperties.push_back(characteristics);
        }

        ACameraManager_deleteCameraIdList(cameraIds);
        ACameraManager_delete(cameraManager);
    }

    return _captureProperties;
}
