#include "NDKCamera.h

NDKCamera::NDKCamera()
{
    init();
}

NDKCamera::~NDKCamera()
{

}

void NDKCamera::init()
{
    _cameras.clear();
    _cameraMgr = ACameraManager_create();
    ASSERT(cameraMgr_, "Failed to create cameraManager");

    // Pick up a back-facing camera to preview
    enumerateCamera();
    ASSERT(activeCameraId_.size(), "Unknown ActiveCameraIdx");
}

void NDKCamera::enumerateCamera(void)
{

}