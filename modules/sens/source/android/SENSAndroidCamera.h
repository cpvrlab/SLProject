//#############################################################################
//  File:      SENSAndroidCamera.h
//  Authors:   Michael Goettlicher, Luc Girod, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SENS_ANDROID_CAMERA_H
#define SENS_ANDROID_CAMERA_H

#include <string>
#include <map>
#include <thread>

#include <SENSCamera.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImageReader.h>
#include <SENSException.h>

//-----------------------------------------------------------------------------
enum class CaptureSessionState
{
    READY = 0, // session is ready
    ACTIVE,    // session is busy
    CLOSED,    // session is closed(by itself or a new session evicts)
    MAX_STATE
};
//-----------------------------------------------------------------------------
class SENSAndroidCamera : public SENSBaseCamera
{
public:
    SENSAndroidCamera();
    ~SENSAndroidCamera();

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  bool                          provideIntrinsics = true) override;

    void stop() override;

    const SENSCaptureProps& captureProperties() override;

    //callbacks
    void onDeviceDisconnected(ACameraDevice* dev);
    void onDeviceError(ACameraDevice* dev, int err);
    void onCameraStatusChanged(const char* id, bool available);
    void onSessionState(ACameraCaptureSession* ses, CaptureSessionState state);
    void imageCallback(AImageReader* reader);

private:
    //start camera selected in initOptimalCamera as soon as it is available
    void openCamera();
    void createCaptureSession();

    ACameraManager_AvailabilityCallbacks _cameraManagerAvailabilityCallbacks;

    static cv::Mat convertToYuv(AImage* image);

    ACameraManager*                 _cameraManager                 = nullptr;
    ACameraDevice*                  _cameraDevice                  = nullptr;
    AImageReader*                   _imageReader                   = nullptr;
    ANativeWindow*                  _surface                       = nullptr;
    ACaptureSessionOutput*          _captureSessionOutput          = nullptr;
    ACaptureSessionOutputContainer* _captureSessionOutputContainer = nullptr;
    ACameraOutputTarget*            _cameraOutputTarget            = nullptr;
    ACaptureRequest*                _captureRequest                = nullptr;
    ACameraCaptureSession*          _captureSession                = nullptr;

    CaptureSessionState     _captureSessionState = CaptureSessionState::MAX_STATE;
    std::condition_variable _captureSessionStateCV;
    std::mutex              _captureSessionStateMutex;

    //map to track, which cameras are available (we start our camera () as soon as it is available and
    // stop it as soon as it becomes unavailable)
    std::map<std::string, bool> _cameraAvailability;
    std::mutex                  _cameraAvailabilityMutex;
    std::condition_variable     _openCameraCV;

    //wait in start() until camera is opened
    std::atomic<bool> _cameraDeviceOpened{false}; // free to use ( no other apps are using it)

    //camera state
    cv::Size _captureSize;
};
//-----------------------------------------------------------------------------
#endif //SENS_ANDROID_CAMERA_H