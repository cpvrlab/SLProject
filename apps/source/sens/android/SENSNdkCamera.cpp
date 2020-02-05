#include <iostream>
#include <string>
#include <utility>
#include <algorithm>
#include "SENSNdkCamera.h"
#include "SENSException.h"

#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <Utils.h>
#include "SENSNdkCameraUtils.h"
#include "SENSUtils.h"

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

//searches for best machting size and returns it
cv::Size AvailableStreamConfigs::findBestMatchingSize(cv::Size requiredSize)
{
    if (_streamSizes.size() == 0)
        throw SENSException(SENSType::CAM, "No stream configuration available!", __LINE__, __FILE__);

    std::vector<std::pair<float, int>> matchingSizes;

    //calculate score for
    for (int i = 0; i < _streamSizes.size(); ++i)
    {
        //stream size has to have minimum the size of the required size
        if (_streamSizes[i].width >= requiredSize.width && _streamSizes[i].height >= requiredSize.height)
        {
            float crop = 0.f;
            //calculate necessary crop to adjust stream image to required size
            if (((float)requiredSize.width / (float)requiredSize.height) > ((float)_streamSizes[i].width / (float)_streamSizes[i].height))
            {
                float scaleFactor  = (float)requiredSize.width / (float)_streamSizes[i].width;
                float heightScaled = _streamSizes[i].height * scaleFactor;
                crop += heightScaled - requiredSize.height;
                crop += (float)_streamSizes[i].width - (float)requiredSize.width;
            }
            else
            {
                float scaleFactor = (float)requiredSize.height / (float)_streamSizes[i].height;
                float widthScaled = _streamSizes[i].width * scaleFactor;
                crop += widthScaled - requiredSize.width;
                crop += (float)_streamSizes[i].height - (float)requiredSize.height;
            }

            float cropScaleScore = crop;
            matchingSizes.push_back(std::make_pair(cropScaleScore, i));
        }
    }
    //sort by crop
    std::sort(matchingSizes.begin(), matchingSizes.end());

    return _streamSizes[matchingSizes.front().second];
}

SENSNdkCamera::SENSNdkCamera(SENSCamera::Facing facing)
  : SENSCamera(facing),
    _threadException("SENSNdkCamera: empty default exception")
{
    _valid = false;
    //init camera manager
    _cameraManager = ACameraManager_create();
    if (!_cameraManager)
        throw SENSException(SENSType::CAM, "Could not instantiate camera manager!", __LINE__, __FILE__);

    //PrintCameras(_cameraManager);

    //find camera device that fits our needs and retrieve required camera charakteristics
    initOptimalCamera(facing);

    //open the camera
    camera_status_t openResult = ACameraManager_openCamera(_cameraManager, _cameraId.c_str(), getDeviceListener(), &_cameraDevice);
    if (openResult != ACAMERA_OK)
    {
        throw SENSException(SENSType::CAM, "Could not open camera! Already opened?", __LINE__, __FILE__);
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
    ACameraDevice_close(_cameraDevice);
    _cameraDevice = nullptr;

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

    //terminate the thread
    if (_thread)
    {
        _stopThread = true;
        _waitCondition.notify_all();
        if (_thread->joinable())
            _thread->join();
        _thread.release();
    }
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
    reinterpret_cast<SENSNdkCamera*>(ctx)->imageCallback(reader);
}

void SENSNdkCamera::start(const SENSCamera::Config config)
{
    _config      = config;
    _targetWdivH = (float)_config.targetWidth / (float)_config.targetHeight;

    //todo: find best fitting image format in _availableStreamConfig
    cv::Size captureSize = _availableStreamConfig.findBestMatchingSize({_config.targetWidth, _config.targetHeight});

    //create request with necessary parameters

    //create image reader with 2 surfaces (a surface is the like a ring buffer for images)
    if (AImageReader_new(captureSize.width, captureSize.height, AIMAGE_FORMAT_YUV_420_888, 2, &_imageReader) != AMEDIA_OK)
        throw SENSException(SENSType::CAM, "Could not create image reader!", __LINE__, __FILE__);

    //make the adjustments in an asynchronous thread
    if (_config.adjustAsynchronously)
    {
        //register onImageAvailable listener
        AImageReader_ImageListener listener{
          .context          = this,
          .onImageAvailable = onImageCallback,
        };
        AImageReader_setImageListener(_imageReader, &listener);

        //start the thread
        _thread = std::make_unique<std::thread>(&SENSNdkCamera::run, this);
    }

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
        ACameraDevice_createCaptureRequest(_cameraDevice, TEMPLATE_PREVIEW, &_captureRequest);

        ACaptureRequest_addTarget(_captureRequest, _cameraOutputTarget);

        if (ACameraDevice_createCaptureSession(_cameraDevice,
                                               _captureSessionOutputContainer,
                                               getSessionListener(),
                                               &_captureSession) != AMEDIA_OK)
            throw SENSException(SENSType::CAM, "Could not create capture session!", __LINE__, __FILE__);

        //adjust capture request properties
        if (_config.focusMode == FocusMode::FIXED_INFINITY_FOCUS)
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
    }

    //install repeating request
    ACameraCaptureSession_setRepeatingRequest(_captureSession, nullptr, 1, &_captureRequest, nullptr);
}

//todo: add callback for image available and/or completely started
void SENSNdkCamera::start(int width, int height)
{
    Config config;
    config.targetWidth  = width;
    config.targetHeight = height;
    start(config);
}

void SENSNdkCamera::stop()
{
    //todo
}

//-----------------------------------------------------------------------------
//! Does all adjustments needed
SENSFramePtr SENSNdkCamera::processNewYuvImg(cv::Mat yuvImg)
{
    //concert yuv to rgb
    cv::Mat rgbImg;
    cv::cvtColor(yuvImg, rgbImg, cv::COLOR_YUV2RGB_NV21, 3);

    cv::Size inputSize = rgbImg.size();
    //////////////////////////////////////////////////////////////////
    // Crop Video image to required aspect ratio //
    //////////////////////////////////////////////////////////////////

    // Cropping is done almost always.
    // So this is Android image copy loop #2
    int cropW = 0, cropH = 0;
    SENS::cropImage(rgbImg, _targetWdivH, cropW, cropH);

    //////////////////
    // Mirroring //
    //////////////////

    // Mirroring is done for most selfie cameras.
    // So this is Android image copy loop #3
    SENS::mirrorImage(rgbImg, _config.mirrorH, _config.mirrorV);

    /////////////////////////
    // Create grayscale //
    /////////////////////////

    // Creating a grayscale version from an YUV input source is stupid.
    // We just could take the Y channel.
    // Android image copy loop #4
    cv::Mat grayImg;
    if (_config.convertToGray)
    {
        cv::cvtColor(rgbImg, grayImg, cv::COLOR_BGR2GRAY);

        // Reset calibrated image size
        //if (frame.size() != activeCamera->calibration.imageSize()) {
        //    activeCamera->calibration.adaptForNewResolution(lastFrame.size());
        //}
    }

    SENSFramePtr sensFrame = std::make_shared<SENSFrame>(rgbImg, grayImg, inputSize.width, inputSize.height, cropW, cropH, _config.mirrorH, _config.mirrorV);
    return std::move(sensFrame);
}

cv::Mat SENSNdkCamera::convertToYuv(AImage* image)
{
    //int32_t format;
    int32_t height;
    int32_t width;
    //AImage_getFormat(image, &format);
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

    cv::Mat yuv(height + (height / 2), width, CV_8UC1);
    size_t  yubBytes  = yuv.total();
    size_t  origBytes = yLen + uLen + vLen;
    //LOGI("yubBytes %d origBytes %d", yubBytes, origBytes);
    memcpy(yuv.data, yPixel, yLen);
    memcpy(yuv.data + yLen, uPixel, uLen);
    //We do not have to copy the v plane. The u plane contains the interleaved u and v data!
    //memcpy(yuv.data + yLen + uLen, vPixel, vLen);

    return yuv;
}

SENSFramePtr SENSNdkCamera::getLatestFrame()
{
    SENSFramePtr sensFrame;

    if (_config.adjustAsynchronously)
    {
        std::unique_lock<std::mutex> lock(_threadOutputMutex);
        if (_processedFrame)
        {
            //move: its faster because shared_ptr has atomic reference counting and we are the only ones using the object
            sensFrame = std::move(_processedFrame);
        }
        if (_threadHasException)
        {
            throw _threadException;
        }
    }
    else
    {
        AImage*        image;
        media_status_t status = AImageReader_acquireLatestImage(_imageReader, &image);
        if (status == AMEDIA_OK) //status may be unequal to media_ok if there is no new frame available, what is ok if we are very fast
        {
            static int newImage = 0;
            //LOGI("new image %d", newImage++);
            cv::Mat yuv = convertToYuv(image);
            //now that the data is copied we have to delete the image
            AImage_delete(image);

            //make cropping, scaling and mirroring
            sensFrame = processNewYuvImg(yuv);
        }
    }

    return std::move(sensFrame);
}

void SENSNdkCamera::run()
{
    try
    {
        while (!_stopThread)
        {
            cv::Mat yuv;
            {
                std::unique_lock<std::mutex> lock(_threadInputMutex);
                //wait until _yuvImgToProcess is valid or stop thread is required
                auto condition = [&] { return (!_yuvImgToProcess.empty() || _stopThread); };
                _waitCondition.wait(lock, condition);
                //stop processing
                if (_stopThread)
                    return;

                yuv = _yuvImgToProcess;
            }

            //do the processing
            SENSFramePtr sensFrame;
            //make yuv to rgb conversion, cropping, scaling, mirroring, gray conversion
            sensFrame = processNewYuvImg(yuv);

            //move processing result to worker thread output
            {
                std::unique_lock<std::mutex> lock(_threadOutputMutex);
                _processedFrame = std::move(sensFrame);
            }
        }
    }
    catch (std::exception& e)
    {
        std::unique_lock<std::mutex> lock(_threadOutputMutex);
        _threadException    = std::runtime_error(e.what());
        _threadHasException = true;
    }
    catch (...)
    {
        std::unique_lock<std::mutex> lock(_threadOutputMutex);
        _threadException    = SENSException(SENSType::CAM, "Exception in worker thread!", __LINE__, __FILE__);
        _threadHasException = true;
    }
}

void SENSNdkCamera::imageCallback(AImageReader* reader)
{
    AImage*        image  = nullptr;
    media_status_t status = AImageReader_acquireLatestImage(reader, &image);
    if (status == AMEDIA_OK && image)
    {
        static int newImage = 0;
        //LOGI("new image %d", newImage++);

        cv::Mat yuv = convertToYuv(image);

        AImage_delete(image);

        //move yuv image to worker thread input
        {
            std::unique_lock<std::mutex> lock(_threadInputMutex);
            _yuvImgToProcess = yuv;
        }
        _waitCondition.notify_all();
    }
}

void SENSNdkCamera::initOptimalCamera(SENSCamera::Facing facing)
{
    if (_cameraManager == nullptr)
        throw SENSException(SENSType::CAM, "Camera manager is invalid!", __LINE__, __FILE__);

    ACameraIdList* cameraIds = nullptr;
    if (ACameraManager_getCameraIdList(_cameraManager, &cameraIds) != ACAMERA_OK)
        throw SENSException(SENSType::CAM, "Could not retrieve camera list!", __LINE__, __FILE__);

    //find correctly facing cameras
    std::vector<std::string> cameras;

    for (int i = 0; i < cameraIds->numCameras; ++i)
    {
        const char* id = cameraIds->cameraIds[i];

        ACameraMetadata* camCharacteristics;
        ACameraManager_getCameraCharacteristics(_cameraManager, id, &camCharacteristics);

        int32_t         numEntries = 0; //will be filled by getAllTags with number of entries
        const uint32_t* tags       = nullptr;
        ACameraMetadata_getAllTags(camCharacteristics, &numEntries, &tags);
        for (int tagIdx = 0; tagIdx < numEntries; ++tagIdx)
        {
            //first check that ACAMERA_LENS_FACING is contained at all
            if (ACAMERA_LENS_FACING == tags[tagIdx])
            {
                ACameraMetadata_const_entry lensInfo = {0};
                ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo);
                acamera_metadata_enum_android_lens_facing_t androidFacing = static_cast<acamera_metadata_enum_android_lens_facing_t>(lensInfo.data.u8[0]);
                if (facing == SENSCamera::Facing::BACK && androidFacing == ACAMERA_LENS_FACING_BACK ||
                    facing == SENSCamera::Facing::FRONT && androidFacing == ACAMERA_LENS_FACING_FRONT)
                {
                    cameras.push_back(id);
                }

                break;
            }
        }
        ACameraMetadata_free(camCharacteristics);
    }

    if (cameras.size() == 0)
    {
        throw SENSException(SENSType::CAM, "No Camera Available on the device", __LINE__, __FILE__);
    }
    else if (cameras.size() == 1)
    {
        _cameraId = cameras[0];
    }
    else
    {
        //todo: select best fitting camera and assign cameraId. Select the one with standard focal length (no macro or fishy lens).
        throw SENSException(SENSType::CAM,
                            "Multiple camera devices with the same facing available! Implement selection logic!",
                            __LINE__,
                            __FILE__);
    }

    ACameraManager_deleteCameraIdList(cameraIds);

    //retrieve camera characteristics
    ACameraMetadata* camCharacteristics;
    ACameraManager_getCameraCharacteristics(_cameraManager, _cameraId.c_str(), &camCharacteristics);

    int32_t         numEntries = 0; //will be filled by getAllTags with number of entries
    const uint32_t* tags       = nullptr;
    ACameraMetadata_getAllTags(camCharacteristics, &numEntries, &tags);
    for (int tagIdx = 0; tagIdx < numEntries; ++tagIdx)
    {
        ACameraMetadata_const_entry lensInfo = {0};

        if (tags[tagIdx] == ACAMERA_LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        {
            if (ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo) == ACAMERA_OK)
            {
                for (int i = 0; i < lensInfo.count; ++i)
                {
                    _focalLenghts.push_back(lensInfo.data.f[i]);
                }
            }
        }
        else if (tags[tagIdx] == ACAMERA_SENSOR_INFO_PHYSICAL_SIZE)
        {
            if (ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo) == ACAMERA_OK)
            {
                _physicalSensorSizeMM.width  = lensInfo.data.f[0];
                _physicalSensorSizeMM.height = lensInfo.data.f[1];
            }
        }
        else if (tags[tagIdx] == ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS)
        {
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
                    //std::string format direction = lensInfo.data.i32[i + 3] ? "INPUT" : "OUTPUT";
                    //std::string format = GetFormatStr(lensInfo.data.i32[i]);

                    //OUTPUT format and AIMAGE_FORMAT_YUV_420_888 image format
                    if (!lensInfo.data.i32[i + 3] && lensInfo.data.i32[i] == AIMAGE_FORMAT_YUV_420_888)
                    {
                        width  = lensInfo.data.i32[i + 1];
                        height = lensInfo.data.i32[i + 2];
                        _availableStreamConfig.add({width, height});
                    }
                }
            }
        }
    }
    ACameraMetadata_free(camCharacteristics);
}
/*
 * CameraDevice callbacks
 */
void onDeviceDisconnected(void* ctx, ACameraDevice* dev)
{
    reinterpret_cast<SENSNdkCamera*>(ctx)->onDeviceDisconnected(dev);
}

void onDeviceErrorChanges(void* ctx, ACameraDevice* dev, int err)
{
    reinterpret_cast<SENSNdkCamera*>(ctx)->onDeviceError(dev, err);
}

ACameraDevice_stateCallbacks* SENSNdkCamera::getDeviceListener()
{
    static ACameraDevice_stateCallbacks cameraDeviceListener = {
      .context        = this,
      .onDisconnected = ::onDeviceDisconnected,
      .onError        = ::onDeviceErrorChanges,
    };
    return &cameraDeviceListener;
}

/**
 * Handle Camera DeviceStateChanges msg, notify device is disconnected
 * simply close the camera
 */
void SENSNdkCamera::onDeviceDisconnected(ACameraDevice* dev)
{
    LOGW("device %s is disconnected");

    _cameraAvailable = false;
    ACameraDevice_close(_cameraDevice);
    _cameraDevice = nullptr;
}
/**
 * Handles Camera's deviceErrorChanges message, no action;
 * mainly debugging purpose
 *
 *
 */
void SENSNdkCamera::onDeviceError(ACameraDevice* dev, int err)
{
    if (dev == _cameraDevice)
    {
        std::string id(ACameraDevice_getId(dev));
        LOGI("CameraDevice %s is in error %#x", id.c_str(), err);
        switch (err)
        {
            case ERROR_CAMERA_IN_USE:
                _cameraAvailable = false;
                break;
            case ERROR_CAMERA_SERVICE:
            case ERROR_CAMERA_DEVICE:
            case ERROR_CAMERA_DISABLED:
            case ERROR_MAX_CAMERAS_IN_USE:
                _cameraAvailable = false;
                break;
            default:
                std::cout << "Unknown Camera Device Error: %#x" << std::endl;
        }
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
    if (_valid && std::string(id) == _cameraId)
    {
        _cameraAvailable = available ? true : false;
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
