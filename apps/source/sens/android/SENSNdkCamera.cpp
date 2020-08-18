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
    LOG_NDKCAM_WARN("onSessionClosed: CaptureSession state: session %p closed", ses);
    reinterpret_cast<SENSNdkCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::CLOSED);
}

void onSessionReady(void* ctx, ACameraCaptureSession* ses)
{
    LOG_NDKCAM_WARN("onSessionReady: CaptureSession state: session %p ready", ses);
    reinterpret_cast<SENSNdkCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::READY);
}

void onSessionActive(void* ctx, ACameraCaptureSession* ses)
{
    LOG_NDKCAM_WARN("onSessionActive: CaptureSession state: session %p active", ses);
    reinterpret_cast<SENSNdkCamera*>(ctx)
      ->onSessionState(ses, CaptureSessionState::ACTIVE);
}

SENSNdkCamera::SENSNdkCamera()
  : _threadException("SENSNdkCamera: empty default exception"),
    _cameraDeviceOpened(false)
{
    LOG_NDKCAM_INFO("Camera instantiated");
}

SENSNdkCamera::~SENSNdkCamera()
{
    //stop();
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

//start camera selected in initOptimalCamera as soon as it is available
void SENSNdkCamera::openCamera()
{
    //init camera manager
    if (!_cameraManager)
    {
        //init availability
        for (const SENSCameraDeviceProperties& c : _captureProperties)
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

    //find current SENSCameraDeviceProperties
    const SENSCameraDeviceProperties* camProps = _captureProperties.camPropsForDeviceId(_config.deviceId);

    if (!_cameraDeviceOpened)
    {
        LOG_NDKCAM_DEBUG("openCamera: Camera device not open");
        auto condition = [&] {
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

            auto condition = [&] {
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

        if (_thread)
        {
            LOG_NDKCAM_DEBUG("openCamera: Terminate the thread...");
            _stopThread = true;
            _waitCondition.notify_one();
            if (_thread->joinable())
            {
                _thread->join();
                LOG_NDKCAM_DEBUG("openCamera: Thread joined");
            }
            _thread.release();
            _stopThread = false;
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

            //make the adjustments in an asynchronous thread
            if (_adjustAsynchronously)
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

            createCaptureSession();
        }
    }
    else
    {
        //todo: throw something
    }
}

const SENSCameraConfig& SENSNdkCamera::start(std::string                   deviceId,
                                             const SENSCameraStreamConfig& streamConfig,
                                             cv::Size                      imgRGBSize,
                                             bool                          mirrorV,
                                             bool                          mirrorH,
                                             bool                          convToGrayToImgManip,
                                             int                           imgManipWidth,
                                             bool                          provideIntrinsics,
                                             float                         fovDegFallbackGuess)
{
    if (_started)
    {
        Utils::warnMsg("SENSWebCamera", "Call to start was ignored. Camera is currently running!", __LINE__, __FILE__);
        return _config;
    }

    cv::Size targetSize;
    if (imgRGBSize.width > 0 && imgRGBSize.height > 0)
    {
        targetSize.width  = imgRGBSize.width;
        targetSize.height = imgRGBSize.height;
    }
    else
    {
        targetSize.width  = streamConfig.widthPix;
        targetSize.height = streamConfig.heightPix;
    }

    cv::Size imgManipSize(imgManipWidth,
                          (int)((float)imgManipWidth * (float)targetSize.height / (float)targetSize.width));

    //retrieve all camera characteristics
    if (_captureProperties.size() == 0)
        captureProperties();

    if (_captureProperties.size() == 0)
        throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

    if (!_captureProperties.containsDeviceId(deviceId))
        throw SENSException(SENSType::CAM, "DeviceId does not exist!", __LINE__, __FILE__);

    //init config here
    _config = SENSCameraConfig(deviceId,
                               streamConfig,
                               SENSCameraFocusMode::UNKNOWN,
                               targetSize.width,
                               targetSize.height,
                               imgManipSize.width,
                               imgManipSize.height,
                               mirrorH,
                               mirrorV,
                               convToGrayToImgManip);
    openCamera();
    initCalibration(fovDegFallbackGuess);

    _started = true;
    return _config;
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
    /*
    {
        uint8_t mode = ACAMERA_LENS_OPTICAL_STABILIZATION_MODE_OFF;
        ACaptureRequest_setEntry_u8(_captureRequest, ACAMERA_LENS_OPTICAL_STABILIZATION_MODE, 1,
                                    &mode);

    }
    */
    //install repeating request
    ACameraCaptureSession_setRepeatingRequest(_captureSession, nullptr, 1, &_captureRequest, nullptr);
}

/*
//todo: add callback for image available and/or completely started
void SENSNdkCamera::start(std::string id, int width, int height)
{
    SENSCameraConfig config;
    config.deviceId     = id;
    config.targetWidth  = width;
    config.targetHeight = height;
    start(config);
}
 */

void SENSNdkCamera::stop()
{
    if (_started)
    {
        /*
        if (_openCameraThread)
        {
            _openCameraThread->detach();
            _openCameraThread.release();
        }
         */

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
        _started = false;
    }
}

//-----------------------------------------------------------------------------
//! Does all adjustments needed
SENSFramePtr SENSNdkCamera::processNewYuvImg(cv::Mat yuvImg)
{
    //concert yuv to rgb
    cv::Mat rgbImg;
    cv::cvtColor(yuvImg, rgbImg, cv::COLOR_YUV2RGB_NV21, 3);

    SENSFramePtr sensFrame = postProcessNewFrame(rgbImg, cv::Mat(), false);
    /*
    cv::Size inputSize = rgbImg.size();

    // Crop Video image to required aspect ratio
    int cropW = 0, cropH = 0;
    SENS::cropImage(rgbImg, _targetWdivH, cropW, cropH);

    // Mirroring (is done for most selfie cameras)
    SENS::mirrorImage(rgbImg, _config.mirrorH, _config.mirrorV);

    /////////////////////////
    // Create grayscale //
    /////////////////////////

    // Creating a grayscale version from an YUV input source is stupid.
    // We just could take the Y channel.
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
     */
    return sensFrame;
}

cv::Mat SENSNdkCamera::convertToYuv(AImage* image)
{
    int32_t height, width, rowStrideY;
    AImage_getHeight(image, &height);
    AImage_getWidth(image, &width);
    AImage_getPlaneRowStride(image, 0, &rowStrideY);

    //pointers to yuv data planes and length of yuv data planes in byte
    uint8_t *yPixel, *uPixel, *vPixel;
    int32_t  yLen, uLen, vLen;
    AImage_getPlaneData(image, 0, &yPixel, &yLen);
    AImage_getPlaneData(image, 1, &uPixel, &uLen);
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
    memcpy(yuv.data + yLen + (rowStrideY - width), uPixel, uLen);
    //We do not have to additionally copy the v plane. The u plane contains the interleaved u and v data!
    //memcpy(yuv.data + yLen + uLen, vPixel, vLen);

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

SENSFramePtr SENSNdkCamera::latestFrame()
{
    SENSFramePtr sensFrame;

    if (_started)
    {
        if (_adjustAsynchronously)
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
            if (status ==
                AMEDIA_OK) //status may be unequal to media_ok if there is no new frame available, what is ok if we are very fast
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
            // static int newImage = 0;
            //LOG_NDKCAM_INFO("imageCallback: new image %d", newImage++);
            std::lock_guard<std::mutex> lock(_threadInputMutex);
            _yuvImgToProcess = yuv;
        }
        _waitCondition.notify_one();
    }
}

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
void SENSNdkCamera::onSessionState(ACameraCaptureSession* ses,
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

const SENSCaptureProperties& SENSNdkCamera::captureProperties()
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

            SENSCameraDeviceProperties characteristics(cameraId, facing);

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
                                    //calculate a focal length in pixel that fits to this stream configuration size
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