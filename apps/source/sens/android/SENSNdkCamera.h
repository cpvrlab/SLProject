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
    SENSNdkCamera(SENSCamera::Facing facing);
    ~SENSNdkCamera();

    void    start(int width, int height, FocusMode focusMode) override;
    void    stop() override;
    SENSFramePtr getLatestFrame() override;

    //callbacks
    void onDeviceState(ACameraDevice* dev);
    void onDeviceError(ACameraDevice* dev, int err);
    void onCameraStatusChanged(const char* id, bool available);
    void onSessionState(ACameraCaptureSession* ses, CaptureSessionState state);

private:
    void                                  getBackFacingCameraList();
    void                                  initOptimalCamera(SENSCamera::Facing facing);
    ACameraDevice_stateCallbacks*         getDeviceListener();
    ACameraManager_AvailabilityCallbacks* getManagerListener();
    ACameraCaptureSession_stateCallbacks* getSessionListener();

    SENSFramePtr adjust(cv::Mat frame);

    ACameraManager* _cameraManager = nullptr;

    std::map<std::string, CameraId> _cameras;
    std::string                     _activeCameraId;

    AImageReader*                   _imageReader = nullptr;
    ANativeWindow*                  _surface;
    ACaptureSessionOutput*          _captureSessionOutput;
    ACaptureSessionOutputContainer* _captureSessionOutputContainer;
    ACameraOutputTarget*            _cameraOutputTarget;
    ACaptureRequest*                _captureRequest;
    ACameraCaptureSession*          _captureSession;
    CaptureSessionState             _captureSessionState;

    volatile bool _valid = false;

    unsigned char* _imageBuffer;

    //image properties
    int   _targetWidth  = -1;
    int   _targetHeight = -1;
    float _targetWdivH  = -1.0f;
    bool _mirrorH = false;
    bool _mirrorV = false;
    bool _convertToGray = true;
};

// helper classes to hold enumerated camera
class CameraId
{
public:
    ACameraDevice*                              _device;
    std::string                                 _id;
    acamera_metadata_enum_android_lens_facing_t facing_;
    bool                                        available_; // free to use ( no other apps are using
    bool                                        owner_;     // we are the owner of the camera
    explicit CameraId(const char* id)
      : _device(nullptr),
        facing_(ACAMERA_LENS_FACING_FRONT),
        available_(false),
        owner_(false)
    {
        _id = id;
    }

    explicit CameraId(void) { CameraId(""); }
};

#endif //SENS_NDKCAMERA_H