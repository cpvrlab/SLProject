#ifndef SENS_NDKCAMERA_H
#define SENS_NDKCAMERA_H

#include <string>
#include <map>

#include <SENSCamera.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImageReader.h>
#include <CVImage.h>

class CameraId;

enum class CaptureSessionState : int32_t
{
    READY = 0, // session is ready
    ACTIVE,    // session is busy
    CLOSED,    // session is closed(by itself or a new session evicts)
    MAX_STATE
};

class SENSFrame
{
public:
    SENSFrame(cv::Mat imgBGR,
              cv::Mat imgGray,
              int     captureWidth,
              int     captureHeight,
              int     cropLR,
              int     cropTB,
              bool    mirroredH,
              bool    mirroredV)
      : _imgBGR(imgBGR),
        _imgGray(imgGray),
        _captureWidth(captureWidth),
        _captureHeight(captureHeight),
        _cropLR(cropLR),
        _cropTB(cropTB),
        _mirroredH(mirroredH),
        _mirroredV(mirroredV)
    {
    }

private:
    cv::Mat _imgBGR;
    cv::Mat _imgGray;

    int  _captureWidth;
    int  _captureHeight;
    int  _cropLR;
    int  _cropTB;
    bool _mirroredH;
    bool _mirroredV;
};

class SENSNdkCamera : public SENSCamera
{
public:
    SENSNdkCamera(SENSCamera::Facing facing);
    ~SENSNdkCamera();

    void    start(int width, int height, FocusMode focusMode) override;
    void    stop() override;
    cv::Mat getLatestFrame() override;

    //callbacks
    void onDeviceState(ACameraDevice* dev);
    void onDeviceError(ACameraDevice* dev, int err);
    void onCameraStatusChanged(const char* id, bool available);
    void onSessionState(ACameraCaptureSession* ses, CaptureSessionState state);

private:
    void                                  getBackFacingCameraList();
    ACameraDevice_stateCallbacks*         getDeviceListener();
    ACameraManager_AvailabilityCallbacks* getManagerListener();
    ACameraCaptureSession_stateCallbacks* getSessionListener();

    void adjust(cv::Mat frame, float viewportWdivH);

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

    CVPixFormat _format = PF_unknown; //!< GL pixel format

    //image properties
    int   _targetWidth  = -1;
    int   _targetHeight = -1;
    float _targetWdivH  = -1.0f;
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