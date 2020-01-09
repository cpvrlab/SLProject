#ifndef CAMERA_INTERFACE_H
#define CAMERA_INTERFACE_H

#define RGB 1
#define RGBA 2
#define YUV422 3

#define CAMERA_OK 1
#define CAMERA_DISCONECTED 2
#define CAMERA_ERROR 3

#define CAMERA_BACKFACING 1
#define CAMERA_FRONTFACING 2

typedef struct CameraHandler;
typedef struct Camera;

typedef struct CameraInfo
{
    int prop;
    const char* _id;
} CameraInfo;

void initCameraHandler(CameraHandler** handler);

unsigned int getCameraList(CameraHandler* handler, CameraInfo** cameraInfo);

unsigned int getBackFacingCameraList(CameraHandler* handler, CameraInfo** cameraInfo);

int initCamera(CameraHandler* handler, CameraInfo* info, Camera** cam);

void destroyCamera(Camera** cam);

void destroyCameraHandler(CameraHandler** handler);

int cameraCaptureSession(Camera* cam, int w, int h);

int cameraLastFrame(Camera* cam, unsigned char* imageBuffer);

#endif
