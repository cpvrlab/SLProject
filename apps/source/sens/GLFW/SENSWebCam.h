#ifndef NDKCAMERA_H
#define NDKCAMERA_H

#include <string>
#include <map>
#include <Camera.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>

class CameraId;
class NDKCamera : public Camera
{
public:
    NDKCamera();
    ~NDKCamera();
private:
    void init();
    void enumerateCamera(void);

    std::map<std::string, CameraId> _cameras;
};

// helper classes to hold enumerated camera
class CameraId {
public:
    ACameraManager* _cameraMgr;
    ACameraDevice* device_;
    std::string _id;

    acamera_metadata_enum_android_lens_facing_t facing_;
    bool available_;  // free to use ( no other apps are using
    bool owner_;      // we are the owner of the camera
    explicit CameraId(const char* id)
            : device_(nullptr),
              facing_(ACAMERA_LENS_FACING_FRONT),
              available_(false),
              owner_(false) {
        _id = id;
    }

    explicit CameraId(void) { CameraId(""); }
};

#endif //NDKCAMERA_H