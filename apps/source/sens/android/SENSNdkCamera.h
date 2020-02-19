#ifndef SENS_NDKCAMERA_H
#define SENS_NDKCAMERA_H

#include <string>
#include <map>
#include <thread>

#include <SENSCamera.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImageReader.h>
#include <SENSException.h>

enum class CaptureSessionState
{
    READY = 0, // session is ready
    ACTIVE,    // session is busy
    CLOSED,    // session is closed(by itself or a new session evicts)
    MAX_STATE
};

class SENSNdkCamera : public SENSCamera
{
public:
    SENSNdkCamera();
    ~SENSNdkCamera();

    void         init(SENSCamera::Facing facing) override;
    void         start(const SENSCamera::Config config) override;
    void         start(int width, int height) override;
    void         stop() override;
    SENSFramePtr getLatestFrame() override;

    //callbacks
    void onDeviceDisconnected(ACameraDevice* dev);
    void onDeviceError(ACameraDevice* dev, int err);
    void onCameraStatusChanged(const char* id, bool available);
    void onSessionState(ACameraCaptureSession* ses, CaptureSessionState state);
    void imageCallback(AImageReader* reader);

private:
    //select camera id to open
    void initOptimalCamera(SENSCamera::Facing facing);
    //start camera selected in initOptimalCamera as soon as it is available
    void openCamera();
    void createCaptureSession();

    ACameraManager_AvailabilityCallbacks _cameraManagerAvailabilityCallbacks;

    static cv::Mat convertToYuv(AImage* image);
    SENSFramePtr   processNewYuvImg(cv::Mat yuvImg);
    //run routine for asynchronous adjustment
    void run();

    ACameraManager* _cameraManager = nullptr;

    std::string        _cameraId;
    ACameraDevice*     _cameraDevice = nullptr;
    std::vector<float> _focalLenghts;
    cv::Size2f         _physicalSensorSizeMM;

    AImageReader*                   _imageReader                   = nullptr;
    ANativeWindow*                  _surface                       = nullptr;
    ACaptureSessionOutput*          _captureSessionOutput          = nullptr;
    ACaptureSessionOutputContainer* _captureSessionOutputContainer = nullptr;
    ACameraOutputTarget*            _cameraOutputTarget            = nullptr;
    ACaptureRequest*                _captureRequest                = nullptr;
    ACameraCaptureSession*          _captureSession                = nullptr;
    CaptureSessionState             _captureSessionState           = CaptureSessionState::MAX_STATE;

    //! initialized is true as soon as init was run. After that we selected a desired camera device id and can retrieve stream configuration sizes.
    volatile bool _initialized = false;
    //! flags if our camera device is available (selected by _cameraId)

    //map to track, which cameras are available (we start our camera () as soon as it is available and
    // stop it as soon as it becomes unavailable)
    std::map<std::string, bool> _cameraAvailability;
    std::mutex                  _cameraAvailabilityMutex;
    //async camera start
    std::thread             _openCameraThread;
    std::condition_variable _openCameraCV;

    //wait in start() until camera is opened
    bool                    _cameraDeviceOpened = false; // free to use ( no other apps are using it)
    std::mutex              _cameraDeviceOpenedMutex;
    std::condition_variable _cameraDeviceOpenedCV;
    camera_status_t         _cameraDeviceOpenResult = ACAMERA_OK;

    //async image processing
    std::condition_variable      _waitCondition;
    cv::Mat                      _yuvImgToProcess;
    std::mutex                   _threadInputMutex;
    SENSFramePtr                 _processedFrame;
    std::mutex                   _threadOutputMutex;
    std::unique_ptr<std::thread> _thread;
    std::atomic<bool>            _stopThread;

    std::runtime_error _threadException;
    bool               _threadHasException = false;
};

#endif //SENS_NDKCAMERA_H