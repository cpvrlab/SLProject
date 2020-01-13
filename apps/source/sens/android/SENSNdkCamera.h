#ifndef SENS_NDKCAMERA_H
#define SENS_NDKCAMERA_H

#include <string>
#include <map>

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

class SENSNdkCamera : public SENSCamera
{
public:
    struct StreamConfig
    {
        std::string format;
        std::string direction;
        int width;
        int height;
    };

    SENSNdkCamera(SENSCamera::Facing facing);
    ~SENSNdkCamera();

    void         start(int width, int height, FocusMode focusMode) override;
    void         stop() override;
    SENSFramePtr getLatestFrame() override;

    //callbacks
    void onDeviceDisconnected(ACameraDevice* dev);
    void onDeviceError(ACameraDevice* dev, int err);
    void onCameraStatusChanged(const char* id, bool available);
    void onSessionState(ACameraCaptureSession* ses, CaptureSessionState state);

private:
    void                                  initOptimalCamera(SENSCamera::Facing facing);
    ACameraDevice_stateCallbacks*         getDeviceListener();
    ACameraManager_AvailabilityCallbacks* getManagerListener();
    ACameraCaptureSession_stateCallbacks* getSessionListener();

    SENSFramePtr adjust(cv::Mat frame);

    ACameraManager* _cameraManager = nullptr;

    std::string    _cameraId;
    ACameraDevice* _cameraDevice = nullptr;
    bool           _cameraAvailable = false; // free to use ( no other apps are using )
    std::vector<float> _focalLenghts;
    cv::Size2f _physicalSensorSizeMM;
    std::vector<StreamConfig> _availableStreamConfig;

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
    int   _targetWidth   = -1;
    int   _targetHeight  = -1;
    float _targetWdivH   = -1.0f;
    bool  _mirrorH       = false;
    bool  _mirrorV       = false;
    bool  _convertToGray = true;
};

#endif //SENS_NDKCAMERA_H