#include <iostream>
#include <string>
#include <utility>
#include <algorithm>
#include "SENSNdkCamera.h"
#include "SENSException.h"

#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <Utils.h>
#include <HighResTimer.h>
#include "SENSNdkCameraUtils.h"
#include "SENSUtils.h"

#define LOG_NDKCAM_WARN(...) Utils::log("SENSNdkCamera", __VA_ARGS__);
#define LOG_NDKCAM_INFO(...) Utils::log("SENSNdkCamera", __VA_ARGS__);
#define LOG_NDKCAM_DEBUG(...) Utils::log("SENSNdkCamera", __VA_ARGS__);
/*
 * Camera Manager Listener object
 */
void onCameraAvailable(void* ctx, const char* id)
{
    reinterpret_cast<SENSNdkCamera*>(ctx)->onCameraStatusChanged(id, true);
}
void onCameraUnavailable(void* ctx, const char* id)
{
    reinterpret_cast<SENSNdkCamera*>(ctx)->onCameraStatusChanged(id, false);
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

// CaptureSession state callbacks
void onSessionClosed(void* ctx, ACameraCaptureSession* ses)
{
    LOG_NDKCAM_WARN("CaptureSession state: session %p closed", ses);
    reinterpret_cast<SENSNdkCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::CLOSED);
}

void onSessionReady(void* ctx, ACameraCaptureSession* ses)
{
    LOG_NDKCAM_WARN("CaptureSession state: session %p ready", ses);
    reinterpret_cast<SENSNdkCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::READY);
}

void onSessionActive(void* ctx, ACameraCaptureSession* ses)
{
    LOG_NDKCAM_WARN("CaptureSession state: session %p active", ses);
    reinterpret_cast<SENSNdkCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::ACTIVE);
}

SENSNdkCamera::SENSNdkCamera(SENSCameraCharacteristics characteristics)
  : _threadException("SENSNdkCamera: empty default exception"),
    _initialized(false),
    _cameraDeviceOpened(false)
{
    _characteristics = characteristics;
    LOG_NDKCAM_INFO("Camera instantiated");
}

SENSNdkCamera::~SENSNdkCamera()
{
    stop();
    LOG_NDKCAM_INFO("~SENSNdkCamera: Camera destructor finished");
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

/*
void SENSNdkCamera::init(SENSCameraFacing facing)
{
    //stop old camera device
    stop();

    //init camera manager
    _cameraManager = ACameraManager_create();
    if (!_cameraManager)
        throw SENSException(SENSType::CAM, "Could not instantiate camera manager!", __LINE__, __FILE__);

    //register callbacks
    _cameraManagerAvailabilityCallbacks = {
      .context             = this,
      .onCameraAvailable   = onCameraAvailable,
      .onCameraUnavailable = onCameraUnavailable,
    };
    ACameraManager_registerAvailabilityCallback(_cameraManager, &_cameraManagerAvailabilityCallbacks);

    //PrintCameras(_cameraManager);

    //find camera device that fits our needs and retrieve required camera charakteristics
    //initOptimalCamera(Facing::);
    _initialized = true;
    throw SENSException(SENSType::CAM, "Do not use this function anymore", __LINE__, __FILE__);

    //open the camera asynchronously to make sure that it is available
    _openCameraThread = std::thread(&SENSNdkCamera::openCamera, this);
}
*/
//start camera selected in initOptimalCamera as soon as it is available
void SENSNdkCamera::openCamera()
{
    auto condition = [&] {
        return (_cameraAvailability[_characteristics.cameraId]);
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
    _cameraDeviceOpenResult = ACameraManager_openCamera(_cameraManager, _characteristics.cameraId.c_str(), &cameraDeviceListener, &_cameraDevice);
    {
        std::lock_guard<std::mutex> lock(_cameraDeviceOpenedMutex);
        _cameraDeviceOpened = true;
    }
    _cameraDeviceOpenedCV.notify_one();
}

void SENSNdkCamera::start(const SENSCamera::Config config)
{
    //if (!_initialized)
    //    throw SENSException(SENSType::CAM, "Camera is not initialized (call init function)!", __LINE__, __FILE__);

    //init camera manager
    if (!_cameraManager)
    {
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
    }

    //Attention: if we never access the _cameraManager the onCameraStatusChanged never comes (seems to be an android bug)
    {
        ACameraIdList*  cameraIds = nullptr;
        camera_status_t status    = ACameraManager_getCameraIdList(_cameraManager, &cameraIds);
        ACameraManager_deleteCameraIdList(cameraIds);
        //PrintCameras(_cameraManager);
    }

    //open the camera asynchronously to make sure that it is available
    _openCameraThread = std::thread(&SENSNdkCamera::openCamera, this);

    //todo: how do we track if a camera is started twice?

    _config      = config;
    _targetWdivH = (float)_config.targetWidth / (float)_config.targetHeight;

    //todo: find best fitting image format in _availableStreamConfig
    cv::Size captureSize = _characteristics.streamConfig.findBestMatchingSize({_config.targetWidth, _config.targetHeight});

    LOG_NDKCAM_INFO("start: captureSize (%d, %d)", captureSize.width, captureSize.height);
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
        _stopThread = false;
        _thread     = std::make_unique<std::thread>(&SENSNdkCamera::run, this);
    }

    auto condition = [&] {
        return _cameraDeviceOpened;
    };
    std::unique_lock<std::mutex> lock(_cameraDeviceOpenedMutex);
    //wait until camera is opened
    _cameraDeviceOpenedCV.wait(lock, condition);

    if (_cameraDeviceOpenResult != ACAMERA_OK)
    {
        throw SENSException(SENSType::CAM, "Could not open camera! Already opened?", __LINE__, __FILE__);
    }

    createCaptureSession();
}

void SENSNdkCamera::createCaptureSession()
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
    if (ACameraDevice_createCaptureSession(_cameraDevice,
                                           _captureSessionOutputContainer,
                                           &captureSessionStateCallbacks,
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
    //if (_initialized)
    {
        //todo: when to know if we have to detatch? starting and started? or state?
        _openCameraThread.detach();
        _initialized = false;

        //todo: no need to initialize, no need to clear!
        //_camInfoAvailableStreamConfig.clear();
        //_camInfoPhysicalSensorSizeMM = cv::Size2f(0.f, 0.f);
        //_camInfoFocalLenghts.clear();
    }

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
        ACameraManager_unregisterAvailabilityCallback(_cameraManager, &_cameraManagerAvailabilityCallbacks);
        ACameraManager_delete(_cameraManager);
        _cameraManager = nullptr;
    }

    if (_imageReader)
    {
        LOG_NDKCAM_DEBUG("stop: free image reader...");
        AImageReader_delete(_imageReader);
        _imageReader = nullptr;
    }

    if (_thread)
    {
        LOG_NDKCAM_DEBUG("stop: terminate the thread...");
        _stopThread = true;
        _waitCondition.notify_one();
        if (_thread->joinable())
        {
            _thread->join();
            LOG_NDKCAM_DEBUG("stop: thread joined");
        }
        _thread.release();
        _stopThread = false;
    }
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
            //static int getFrameN = 0;
            //LOG_NDKCAM_INFO("getLatestFrame: getFrameN %d", getFrameN++);
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
            //static int getFrameN = 0;
            //LOG_NDKCAM_INFO("getLatestFrame: getFrameN %d", getFrameN++);

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
        LOG_NDKCAM_INFO("run: thread started");
        while (!_stopThread)
        {
            cv::Mat yuv;
            {
                std::unique_lock<std::mutex> lock(_threadInputMutex);
                //wait until _yuvImgToProcess is valid or stop thread is required
                auto condition = [&] {
                    return (!_yuvImgToProcess.empty() || _stopThread);
                };
                _waitCondition.wait(lock, condition);

                if (_stopThread)
                {
                    return;
                }

                if (!_yuvImgToProcess.empty())
                    yuv = _yuvImgToProcess;
            }

            //static int imageConsumed = 0;
            //LOG_NDKCAM_INFO("run: imageConsumed %d", imageConsumed++)

            //make yuv to rgb conversion, cropping, scaling, mirroring, gray conversion
            SENSFramePtr sensFrame = processNewYuvImg(yuv);

            //move processing result to worker thread output
            {
                std::unique_lock<std::mutex> lock(_threadOutputMutex);
                _processedFrame = std::move(sensFrame);
            }
        }
    }
    catch (std::exception& e)
    {
        LOG_NDKCAM_INFO("run: exception");
        std::unique_lock<std::mutex> lock(_threadOutputMutex);
        _threadException    = std::runtime_error(e.what());
        _threadHasException = true;
    }
    catch (...)
    {
        LOG_NDKCAM_INFO("run: exception");
        std::unique_lock<std::mutex> lock(_threadOutputMutex);
        _threadException    = SENSException(SENSType::CAM, "Exception in worker thread!", __LINE__, __FILE__);
        _threadHasException = true;
    }

    if (_stopThread)
    {
        LOG_NDKCAM_INFO("run: stopped thread");
    }
}

void SENSNdkCamera::imageCallback(AImageReader* reader)
{
    AImage*        image  = nullptr;
    media_status_t status = AImageReader_acquireLatestImage(reader, &image);
    if (status == AMEDIA_OK && image)
    {
        cv::Mat yuv = convertToYuv(image);

        AImage_delete(image);

        //move yuv image to worker thread input
        {
            static int newImage = 0;
            //LOG_NDKCAM_INFO("imageCallback: new image %d", newImage++);
            std::lock_guard<std::mutex> lock(_threadInputMutex);
            _yuvImgToProcess = yuv;
        }
        _waitCondition.notify_one();
    }
}

/*
void SENSNdkCamera::initOptimalCamera(SENSCameraFacing facing)
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
                if (facing == SENSCameraFacing::BACK && androidFacing == ACAMERA_LENS_FACING_BACK ||
                    facing == SENSCameraFacing::FRONT && androidFacing == ACAMERA_LENS_FACING_FRONT)
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
                    _camInfoFocalLenghts.push_back(lensInfo.data.f[i]);
                }
            }
        }
        else if (tags[tagIdx] == ACAMERA_SENSOR_INFO_PHYSICAL_SIZE)
        {
            if (ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo) == ACAMERA_OK)
            {
                _camInfoPhysicalSensorSizeMM.width  = lensInfo.data.f[0];
                _camInfoPhysicalSensorSizeMM.height = lensInfo.data.f[1];
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
                        _camInfoAvailableStreamConfig.add({width, height});
                    }
                }
            }
        }
    }
    ACameraMetadata_free(camCharacteristics);
}
*/

/**
 * Handle Camera DeviceStateChanges msg, notify device is disconnected
 * simply close the camera
 */
void SENSNdkCamera::onDeviceDisconnected(ACameraDevice* dev)
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
void SENSNdkCamera::onDeviceError(ACameraDevice* dev, int err)
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
void SENSNdkCamera::onCameraStatusChanged(const char* id, bool available)
{
    LOG_NDKCAM_INFO("onCameraStatusChanged: id: %s available: %s ", id, available ? "true" : "false");
    {
        std::lock_guard<std::mutex> lock(_cameraAvailabilityMutex);
        _cameraAvailability[std::string(id)] = available;
        _openCameraCV.notify_one();
    }
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
void SENSNdkCamera::onSessionState(ACameraCaptureSession* ses,
                                   CaptureSessionState    state)
{
    LOG_NDKCAM_DEBUG("CaptureSession state: entered");

    if (!_captureSession)
        LOG_NDKCAM_WARN("CaptureSession state: CaptureSession is NULL");

    if (state >= CaptureSessionState::MAX_STATE)
    {
        throw SENSException(SENSType::CAM, "Wrong state " + std::to_string((int)state), __LINE__, __FILE__);
    }

    LOG_NDKCAM_WARN("CaptureSession state: %s", getPrintableState(state).c_str());
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

void SENSNdkCameraManager::updateCameraCharacteristics()
{
    ACameraManager* cameraManager = ACameraManager_create();
    if (!cameraManager)
        throw SENSException(SENSType::CAM, "Could not instantiate camera manager!", __LINE__, __FILE__);

    ACameraIdList* cameraIds = nullptr;
    if (ACameraManager_getCameraIdList(cameraManager, &cameraIds) != ACAMERA_OK)
        throw SENSException(SENSType::CAM, "Could not retrieve camera list!", __LINE__, __FILE__);

    for (int i = 0; i < cameraIds->numCameras; ++i)
    {
        SENSCameraCharacteristics characteristics;
        characteristics.cameraId = cameraIds->cameraIds[i];

        ACameraMetadata* camCharacteristics;
        ACameraManager_getCameraCharacteristics(cameraManager, characteristics.cameraId.c_str(), &camCharacteristics);

        int32_t         numEntries = 0; //will be filled by getAllTags with number of entries
        const uint32_t* tags       = nullptr;
        ACameraMetadata_getAllTags(camCharacteristics, &numEntries, &tags);
        for (int tagIdx = 0; tagIdx < numEntries; ++tagIdx)
        {
            ACameraMetadata_const_entry lensInfo = {0};
            //first check that ACAMERA_LENS_FACING is contained at all
            if (tags[tagIdx] == ACAMERA_LENS_FACING)
            {
                ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo);
                acamera_metadata_enum_android_lens_facing_t androidFacing = static_cast<acamera_metadata_enum_android_lens_facing_t>(lensInfo.data.u8[0]);
                if (androidFacing == ACAMERA_LENS_FACING_BACK)
                    characteristics.facing = SENSCameraFacing::BACK;
                else if (androidFacing == ACAMERA_LENS_FACING_FRONT)
                    characteristics.facing = SENSCameraFacing::FRONT;
                else //if (androidFacing == ACAMERA_LENS_FACING_EXTERNAL)
                    characteristics.facing = SENSCameraFacing::EXTERNAL;
            }
            else if (tags[tagIdx] == ACAMERA_LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            {
                if (ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo) == ACAMERA_OK)
                {
                    for (int i = 0; i < lensInfo.count; ++i)
                    {
                        characteristics.focalLenghts.push_back(lensInfo.data.f[i]);
                    }
                }
            }
            else if (tags[tagIdx] == ACAMERA_SENSOR_INFO_PHYSICAL_SIZE)
            {
                if (ACameraMetadata_getConstEntry(camCharacteristics, tags[tagIdx], &lensInfo) == ACAMERA_OK)
                {
                    characteristics.physicalSensorSizeMM.width  = lensInfo.data.f[0];
                    characteristics.physicalSensorSizeMM.height = lensInfo.data.f[1];
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
                            characteristics.streamConfig.add({width, height});
                        }
                    }
                }
            }
        }
        ACameraMetadata_free(camCharacteristics);
        _characteristics.push_back(characteristics);
    }

    ACameraManager_deleteCameraIdList(cameraIds);
    ACameraManager_delete(cameraManager);
}

SENSCameraPtr SENSNdkCameraManager::getOptimalCamera(SENSCameraFacing facing)
{
    SENSCameraPtr camera;
    for (const SENSCameraCharacteristics& c : _characteristics)
    {
        if (c.facing == facing)
        {
            camera = std::unique_ptr<SENSNdkCamera>(new SENSNdkCamera(c));
        }
    }
    return std::move(camera);
}

SENSCameraPtr SENSNdkCameraManager::getCameraForId(std::string id)
{
    SENSCameraPtr camera;
    for (const SENSCameraCharacteristics& c : _characteristics)
    {
        if (c.cameraId == id)
        {
            camera = std::unique_ptr<SENSNdkCamera>(new SENSNdkCamera(c));
        }
    }
    return std::move(camera);
}
