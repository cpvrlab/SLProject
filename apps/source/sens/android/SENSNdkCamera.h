#ifndef SENS_NDKCAMERA_H
#define SENS_NDKCAMERA_H

#include <string>
#include <map>
#include <thread>

#include <SENSCamera.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImageReader.h>

class CameraId;

enum class CaptureSessionState
{
    READY = 0, // session is ready
    ACTIVE,    // session is busy
    CLOSED,    // session is closed(by itself or a new session evicts)
    MAX_STATE
};

class AvailableStreamConfigs
{
public:
    void add(cv::Size size)
    {
        _streamSizes.push_back(size);
    }

    //searches for best matching size and returns it
    cv::Size findBestMatchingSize(cv::Size requiredSize);

private:
    std::vector<cv::Size> _streamSizes;
};

class SENSNdkCamera : public SENSCamera
{
public:
    SENSNdkCamera(SENSCamera::Facing facing);
    ~SENSNdkCamera();

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
    void                                  initOptimalCamera(SENSCamera::Facing facing);
    ACameraDevice_stateCallbacks*         getDeviceListener();
    ACameraManager_AvailabilityCallbacks* getManagerListener();
    ACameraCaptureSession_stateCallbacks* getSessionListener();

    static cv::Mat convertToYuv(AImage* image);
    SENSFramePtr   processNewYuvImg(cv::Mat yuvImg);
    //run routine for asynchronous adjustment
    void run();

    ACameraManager* _cameraManager = nullptr;

    std::string            _cameraId;
    ACameraDevice*         _cameraDevice    = nullptr;
    bool                   _cameraAvailable = false; // free to use ( no other apps are using )
    std::vector<float>     _focalLenghts;
    cv::Size2f             _physicalSensorSizeMM;
    AvailableStreamConfigs _availableStreamConfig;

    //std::map<std::string, CameraId> _cameras;
    AImageReader*                   _imageReader = nullptr;
    ANativeWindow*                  _surface;
    ACaptureSessionOutput*          _captureSessionOutput;
    ACaptureSessionOutputContainer* _captureSessionOutputContainer;
    ACameraOutputTarget*            _cameraOutputTarget;
    ACaptureRequest*                _captureRequest;
    ACameraCaptureSession*          _captureSession;
    CaptureSessionState             _captureSessionState;

    volatile bool _valid = false;

    //image properties
    float              _targetWdivH = -1.0f;
    SENSCamera::Config _camConfig;

    std::condition_variable      _waitCondition;
    cv::Mat                      _yuvImgToProcess;
    std::mutex                   _threadInputMutex;
    SENSFramePtr                 _processedFrame;
    std::mutex                   _threadOutputMutex;
    std::unique_ptr<std::thread> _thread;
    std::atomic<bool>            _stopThread;

    std::exception _threadException;
    bool           _threadHasException;
};

#endif //SENS_NDKCAMERA_H