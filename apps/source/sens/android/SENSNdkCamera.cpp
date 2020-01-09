#include <iostream>
#include <string>
#include "SENSNdkCamera.h"
#include "SENSException.h"

#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <Utils.h>

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

SENSNdkCamera::SENSNdkCamera(SENSCamera::Facing facing)
  : SENSCamera(facing)
{
    _valid = false;
    //init camera manager
    _cameraManager = ACameraManager_create();
    if (!_cameraManager)
        throw SENSException(SENSType::CAM, "Could not instantiate camera manager!", __LINE__, __FILE__);

    //find camera device that fits our needs and retrieve required camera charakteristics
    //todo: depending on facing... retrieve all resolutions and sensor size and fov
    initOptimalCamera(facing);

    //open the camera
    camera_status_t openResult = ACameraManager_openCamera(_cameraManager, _activeCameraId.c_str(), getDeviceListener(), &_cameras[_activeCameraId]._device);
    if (openResult != ACAMERA_OK)
    {
        throw SENSException(SENSType::CAM, "Could not open camera! Check the camera permissions!", __LINE__, __FILE__);
    }
    //register callbacks
    ACameraManager_registerAvailabilityCallback(_cameraManager, getManagerListener());

    _valid = true;
}

SENSNdkCamera::~SENSNdkCamera()
{
    _valid = false;
    // stop session if it is on:
    if (_captureSessionState == CaptureSessionState::ACTIVE)
    {
        ACameraCaptureSession_stopRepeating(_captureSession);
    }
    ACameraCaptureSession_close(_captureSession);

    //free request stuff
    ACaptureRequest_removeTarget(_captureRequest, _cameraOutputTarget);
    ACaptureRequest_free(_captureRequest);
    ACaptureSessionOutputContainer_remove(_captureSessionOutputContainer, _captureSessionOutput);
    ACaptureSessionOutput_free(_captureSessionOutput);
    ANativeWindow_release(_surface);
    ACaptureSessionOutputContainer_free(_captureSessionOutputContainer);

    //close camera
    ACameraDevice_close(_cameras[_activeCameraId]._device);

    //delete camera manager
    if (_cameraManager)
    {
        ACameraManager_unregisterAvailabilityCallback(_cameraManager, getManagerListener());
        ACameraManager_delete(_cameraManager);
        _cameraManager = nullptr;
    }

    //free image reader
    if (_imageReader)
    {
        AImageReader_delete(_imageReader);
    }
}

//todo: add callback for image available and/or completely started
void SENSNdkCamera::start(int width, int height, FocusMode focusMode)
{
    _targetWidth  = width;
    _targetHeight = height;
    _targetWdivH  = (float)width / (float)height;
    _mirrorH = false;
    _mirrorV = false;
    _convertToGray = true;

    //create request with necessary parameters

    //create image reader with 2 surfaces (a surface is the like a ring buffer for images)
    if (AImageReader_new(_targetWidth, _targetHeight, AIMAGE_FORMAT_YUV_420_888, 2, &_imageReader) != AMEDIA_OK)
        throw SENSException(SENSType::CAM, "Could not create image reader!", __LINE__, __FILE__);
    //todo: register onImageAvailable listener

    //create session
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
        ACameraDevice_createCaptureRequest(_cameras[_activeCameraId]._device, TEMPLATE_PREVIEW, &_captureRequest);
        //todo change focus

        ACaptureRequest_addTarget(_captureRequest, _cameraOutputTarget);

        if (ACameraDevice_createCaptureSession(_cameras[_activeCameraId]._device,
                                               _captureSessionOutputContainer,
                                               getSessionListener(),
                                               &_captureSession) != AMEDIA_OK)
            throw SENSException(SENSType::CAM, "Could not create capture session!", __LINE__, __FILE__);
    }

    //install repeating request
    ACameraCaptureSession_setRepeatingRequest(_captureSession, nullptr, 1, &_captureRequest, nullptr);

    //todo
    _imageBuffer = (uint8_t*)malloc(3 * width * height);
}

void SENSNdkCamera::stop()
{
}

//-----------------------------------------------------------------------------
//! Does all adjustments needed for the videoTexture
/*! CVCapture::adjustForSL processes the following adjustments for all input
images no matter with what they where captured:
\n
1) Crops the input image if it doesn't match the screens aspect ratio. The
input image mostly does't fit the aspect of the output screen aspect. If the
input image is too high we crop it on top and bottom, if it is too wide we
crop it on the sides.
If viewportWdivH is negative the viewport aspect will be adapted to the video
aspect ratio. No cropping will be applied.
\n
2) Some cameras toward a face mirror the image and some do not. If a input
image should be mirrored or not is stored in CVCalibration::_isMirroredH
(H for horizontal) and CVCalibration::_isMirroredV (V for vertical).
\n
3) Many of the further processing steps are faster done on grayscale images.
We therefore create a copy that is grayscale converted.
*/

SENSFramePtr SENSNdkCamera::adjust(cv::Mat frame)
{
    cv::Size inputSize = frame.size();
    //////////////////////////////////////////////////////////////////
    // Crop Video image to the aspect ratio of OpenGL background //
    //////////////////////////////////////////////////////////////////

    // Cropping is done almost always.
    // So this is Android image copy loop #2

    float inWdivH = (float)frame.cols / (float)frame.rows;
    // viewportWdivH is negative the viewport aspect will be the same
    float outWdivH = _targetWdivH < 0.0f ? inWdivH : _targetWdivH;

    int cropH  = 0; // crop height in pixels of the source image
    int cropW  = 0; // crop width in pixels of the source image
    if (Utils::abs(inWdivH - outWdivH) > 0.01f)
    {
        int width  = 0; // width in pixels of the destination image
        int height = 0; // height in pixels of the destination image
        int wModulo4;
        int hModulo4;

        if (inWdivH > outWdivH) // crop input image left & right
        {
            width  = (int)((float)frame.rows * outWdivH);
            height = frame.rows;
            cropW  = (int)((float)(frame.cols - width) * 0.5f);

            // Width must be devidable by 4
            wModulo4 = width % 4;
            if (wModulo4 == 1) width--;
            if (wModulo4 == 2)
            {
                cropW++;
                width -= 2;
            }
            if (wModulo4 == 3) width++;
        }
        else // crop input image at top & bottom
        {
            width  = frame.cols;
            height = (int)((float)frame.cols / outWdivH);
            cropH  = (int)((float)(frame.rows - height) * 0.5f);

            // Height must be devidable by 4
            hModulo4 = height % 4;
            if (hModulo4 == 1) height--;
            if (hModulo4 == 2)
            {
                cropH++;
                height -= 2;
            }
            if (hModulo4 == 3) height++;
        }

        frame(cv::Rect(cropW, cropH, width, height)).copyTo(frame);
        //imwrite("AfterCropping.bmp", lastFrame);
    }

    //////////////////
    // Mirroring //
    //////////////////

    // Mirroring is done for most selfie cameras.
    // So this is Android image copy loop #3
    if (_mirrorH)
    {
        cv::Mat mirrored;
        if (_mirrorH)
            cv::flip(frame, mirrored, -1);
        else
            cv::flip(frame, mirrored, 1);
        frame = mirrored;
    }
    else if (_mirrorV)
    {
        cv::Mat mirrored;
        if (_mirrorH)
            cv::flip(frame, mirrored, -1);
        else
            cv::flip(frame, mirrored, 0);
        frame = mirrored;
    }

    /////////////////////////
    // Create grayscale //
    /////////////////////////

    // Creating a grayscale version from an YUV input source is stupid.
    // We just could take the Y channel.
    // Android image copy loop #4
    cv::Mat grayImg;
    if(_convertToGray)
    {
        cv::cvtColor(frame, grayImg, cv::COLOR_BGR2GRAY);

        // Reset calibrated image size
        //if (frame.size() != activeCamera->calibration.imageSize()) {
        //    activeCamera->calibration.adaptForNewResolution(lastFrame.size());
        //}
    }

    SENSFramePtr sensFrame = std::make_shared<SENSFrame>(frame, grayImg, inputSize.width, inputSize.height, cropW, cropH, _mirrorH, _mirrorV );
    return sensFrame;
}

SENSFramePtr SENSNdkCamera::getLatestFrame()
{
    SENSFramePtr sensFrame;

    cv::Mat        frame;
    AImage*        image;
    media_status_t status = AImageReader_acquireLatestImage(_imageReader, &image);
    if (status == AMEDIA_OK) //status may be unequal to media_ok if there is no new frame available, what is ok if we are very fast
    {
        int32_t format;
        int32_t height;
        int32_t width;
        AImage_getFormat(image, &format);
        AImage_getHeight(image, &height);
        AImage_getWidth(image, &width);

        //copy image data to yuv image
        //pointers to yuv data planes
        uint8_t *yPixel, *uPixel, *vPixel;
        //length of yuv data planes in byte
        int32_t yLen, uLen, vLen;
        AImage_getPlaneData(image, 0, &yPixel, &yLen);
        AImage_getPlaneData(image, 1, &uPixel, &uLen);
        AImage_getPlaneData(image, 2, &vPixel, &vLen);

        unsigned char* imageBuffer = (uint8_t*)malloc(yLen + uLen + vLen);
        memcpy(imageBuffer, yPixel, yLen);
        memcpy(imageBuffer + yLen, uPixel, uLen);
        memcpy(imageBuffer + yLen + uLen, vPixel, vLen);

        //now that the data is copied we have to delete the image
        AImage_delete(image);

        //todo: measure time for yuv conversion and adjustment and move to other thread
        //make yuv conversion
        cv::Mat yuv(height + height / 2, width, CV_8UC1, (void*)imageBuffer);
        cv::cvtColor(yuv, frame, cv::COLOR_YUV2RGB_NV21, 3);

        //make cropping, scaling and mirroring
        sensFrame = adjust(frame);
    }

    return sensFrame;
}

void SENSNdkCamera::initOptimalCamera(SENSCamera::Facing facing)
{
    if (_cameraManager == nullptr)
        throw SENSException(SENSType::CAM, "Camera manager is invalid!", __LINE__, __FILE__);

    ACameraIdList* cameraIds = nullptr;
    if (ACameraManager_getCameraIdList(_cameraManager, &cameraIds) != ACAMERA_OK)
        throw SENSException(SENSType::CAM, "Could not retrieve camera list!", __LINE__, __FILE__);

    for (int i = 0; i < cameraIds->numCameras; ++i)
    {
        const char* id = cameraIds->cameraIds[i];

        ACameraMetadata* metadataObj;
        ACameraManager_getCameraCharacteristics(_cameraManager, id, &metadataObj);

        int32_t         count = 0;
        const uint32_t* tags  = nullptr;
        ACameraMetadata_getAllTags(metadataObj, &count, &tags);
        for (int tagIdx = 0; tagIdx < count; ++tagIdx)
        {
            if (ACAMERA_LENS_FACING == tags[tagIdx])
            {
                ACameraMetadata_const_entry lensInfo = {
                  0,
                };
                ACameraMetadata_getConstEntry(metadataObj, tags[tagIdx], &lensInfo);
                CameraId cam(id);
                cam.facing_ = static_cast<acamera_metadata_enum_android_lens_facing_t>(
                  lensInfo.data.u8[0]);
                cam.owner_        = false;
                cam._device       = nullptr;
                _cameras[cam._id] = cam;
                if (cam.facing_ == ACAMERA_LENS_FACING_BACK)
                {
                    _activeCameraId = cam._id;
                }
                break;
            }
        }
        ACameraMetadata_free(metadataObj);
    }

    if (_cameras.size() == 0)
        throw SENSException(SENSType::CAM, "No Camera Available on the device", __LINE__, __FILE__);

    if (_activeCameraId.length() == 0)
    {
        // if no back facing camera found, pick up the first one to use...
        _activeCameraId = _cameras.begin()->second._id;
    }
    ACameraManager_deleteCameraIdList(cameraIds);

    //find cameras with this facing

    //select the one with standard focal length (no macro or fishy lens)

    //retrieve camera charakteristics
    //focal length: ACAMERA_LENS_INFO_AVAILABLE_FOCAL_LENGTHS
    //physical sensor size: ACAMERA_SENSOR_INFO_PHYSICAL_SIZE
    //
}

void SENSNdkCamera::getBackFacingCameraList()
{
    if (_cameraManager == nullptr)
        throw SENSException(SENSType::CAM, "Camera manager is invalid!", __LINE__, __FILE__);

    ACameraIdList* cameraIds = nullptr;
    if (ACameraManager_getCameraIdList(_cameraManager, &cameraIds) != ACAMERA_OK)
        throw SENSException(SENSType::CAM, "Could not retrieve camera list!", __LINE__, __FILE__);

    for (int i = 0; i < cameraIds->numCameras; ++i)
    {
        const char* id = cameraIds->cameraIds[i];

        ACameraMetadata* metadataObj;
        ACameraManager_getCameraCharacteristics(_cameraManager, id, &metadataObj);

        int32_t         count = 0;
        const uint32_t* tags  = nullptr;
        ACameraMetadata_getAllTags(metadataObj, &count, &tags);
        for (int tagIdx = 0; tagIdx < count; ++tagIdx)
        {
            if (ACAMERA_LENS_FACING == tags[tagIdx])
            {
                ACameraMetadata_const_entry lensInfo = {
                  0,
                };
                ACameraMetadata_getConstEntry(metadataObj, tags[tagIdx], &lensInfo);
                CameraId cam(id);
                cam.facing_ = static_cast<acamera_metadata_enum_android_lens_facing_t>(
                  lensInfo.data.u8[0]);
                cam.owner_        = false;
                cam._device       = nullptr;
                _cameras[cam._id] = cam;
                if (cam.facing_ == ACAMERA_LENS_FACING_BACK)
                {
                    _activeCameraId = cam._id;
                }
                break;
            }
        }
        ACameraMetadata_free(metadataObj);
    }

    if (_cameras.size() == 0)
        throw SENSException(SENSType::CAM, "No Camera Available on the device", __LINE__, __FILE__);

    if (_activeCameraId.length() == 0)
    {
        // if no back facing camera found, pick up the first one to use...
        _activeCameraId = _cameras.begin()->second._id;
    }
    ACameraManager_deleteCameraIdList(cameraIds);
}

/*
 * CameraDevice callbacks
 */
void onDeviceStateChanges(void* ctx, ACameraDevice* dev)
{
    reinterpret_cast<SENSNdkCamera*>(ctx)->onDeviceState(dev);
}

void onDeviceErrorChanges(void* ctx, ACameraDevice* dev, int err)
{
    reinterpret_cast<SENSNdkCamera*>(ctx)->onDeviceError(dev, err);
}

ACameraDevice_stateCallbacks* SENSNdkCamera::getDeviceListener()
{
    static ACameraDevice_stateCallbacks cameraDeviceListener = {
      .context        = this,
      .onDisconnected = ::onDeviceStateChanges,
      .onError        = ::onDeviceErrorChanges,
    };
    return &cameraDeviceListener;
}

/**
 * Handle Camera DeviceStateChanges msg, notify device is disconnected
 * simply close the camera
 */
void SENSNdkCamera::onDeviceState(ACameraDevice* dev)
{
    std::string id(ACameraDevice_getId(dev));
    LOGW("device %s is disconnected", id.c_str());

    _cameras[id].available_ = false;
    ACameraDevice_close(_cameras[id]._device);
    _cameras.erase(id);
}
/**
 * Handles Camera's deviceErrorChanges message, no action;
 * mainly debugging purpose
 *
 *
 */
void SENSNdkCamera::onDeviceError(ACameraDevice* dev, int err)
{
    std::string id(ACameraDevice_getId(dev));

    LOGI("CameraDevice %s is in error %#x", id.c_str(), err);
    //PrintCameraDeviceError(err);

    CameraId& cam = _cameras[id];

    switch (err)
    {
        case ERROR_CAMERA_IN_USE:
            cam.available_ = false;
            cam.owner_     = false;
            break;
        case ERROR_CAMERA_SERVICE:
        case ERROR_CAMERA_DEVICE:
        case ERROR_CAMERA_DISABLED:
        case ERROR_MAX_CAMERAS_IN_USE:
            cam.available_ = false;
            cam.owner_     = false;
            break;
        default:
            std::cout << "Unknown Camera Device Error: %#x" << std::endl;
    }
}

/*
 * Camera Manager Listener object
 */
void OnCameraAvailable(void* ctx, const char* id)
{
    reinterpret_cast<SENSNdkCamera*>(ctx)->onCameraStatusChanged(id, true);
}
void OnCameraUnavailable(void* ctx, const char* id)
{
    reinterpret_cast<SENSNdkCamera*>(ctx)->onCameraStatusChanged(id, false);
}

/**
 * Construct a camera manager listener on the fly and return to caller
 *
 * @return ACameraManager_AvailabilityCallback
 */
ACameraManager_AvailabilityCallbacks* SENSNdkCamera::getManagerListener()
{
    static ACameraManager_AvailabilityCallbacks cameraMgrListener = {
      .context             = this,
      .onCameraAvailable   = OnCameraAvailable,
      .onCameraUnavailable = OnCameraUnavailable,
    };
    return &cameraMgrListener;
}

/**
 * OnCameraStatusChanged()
 *  handles Callback from ACameraManager
 */
void SENSNdkCamera::onCameraStatusChanged(const char* id, bool available)
{
    if (_valid)
    {
        _cameras[std::string(id)].available_ = available ? true : false;
    }
}

// CaptureSession state callbacks
void onSessionClosed(void* ctx, ACameraCaptureSession* ses)
{
    LOGW("session %p closed", ses);
    reinterpret_cast<SENSNdkCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::CLOSED);
}
void onSessionReady(void* ctx, ACameraCaptureSession* ses)
{
    LOGW("session %p ready", ses);
    reinterpret_cast<SENSNdkCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::READY);
}
void onSessionActive(void* ctx, ACameraCaptureSession* ses)
{
    LOGW("session %p active", ses);
    reinterpret_cast<SENSNdkCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::ACTIVE);
}

ACameraCaptureSession_stateCallbacks* SENSNdkCamera::getSessionListener()
{
    static ACameraCaptureSession_stateCallbacks sessionListener = {
      .context  = this,
      .onActive = ::onSessionActive,
      .onReady  = ::onSessionReady,
      .onClosed = ::onSessionClosed,
    };
    return &sessionListener;
}

/**
 * Handles capture session state changes.
 *   Update into internal session state.
 */
void SENSNdkCamera::onSessionState(ACameraCaptureSession* ses,
                                   CaptureSessionState    state)
{
    if (!ses || ses != _captureSession)
    {
        LOGW("CaptureSession is %s", (ses ? "NOT our session" : "NULL"));
        return;
    }

    if (state >= CaptureSessionState::MAX_STATE)
    {
        throw SENSException(SENSType::CAM, "Wrong state " + std::to_string((int)state), __LINE__, __FILE__);
    }

    _captureSessionState = state;
}