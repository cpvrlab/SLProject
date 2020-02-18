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

class CameraId;

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
    void initOptimalCamera(SENSCamera::Facing facing);
    //ACameraDevice_stateCallbacks*         getDeviceListener();
    //ACameraManager_AvailabilityCallbacks* getManagerListener();
    ACameraManager_AvailabilityCallbacks _cameraManagerAvailabilityCallbacks;

    /*
    static void onSessionActive(void* ctx, ACameraCaptureSession* ses);
    static void onSessionClosed(void* ctx, ACameraCaptureSession* ses);
    static void onSessionReady(void* ctx, ACameraCaptureSession* ses);
*/
    static cv::Mat convertToYuv(AImage* image);
    SENSFramePtr   processNewYuvImg(cv::Mat yuvImg);
    //run routine for asynchronous adjustment
    void run();

    ACameraManager* _cameraManager = nullptr;

    std::string        _cameraId;
    ACameraDevice*     _cameraDevice    = nullptr;
    bool               _cameraAvailable = false; // free to use ( no other apps are using )
    std::vector<float> _focalLenghts;
    cv::Size2f         _physicalSensorSizeMM;

    //std::map<std::string, CameraId> _cameras;
    AImageReader*                   _imageReader                   = nullptr;
    ANativeWindow*                  _surface                       = nullptr;
    ACaptureSessionOutput*          _captureSessionOutput          = nullptr;
    ACaptureSessionOutputContainer* _captureSessionOutputContainer = nullptr;
    ACameraOutputTarget*            _cameraOutputTarget            = nullptr;
    ACaptureRequest*                _captureRequest                = nullptr;
    ACameraCaptureSession*          _captureSession                = nullptr;
    CaptureSessionState             _captureSessionState           = CaptureSessionState::MAX_STATE;

    volatile bool _valid = false;

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