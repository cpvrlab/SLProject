/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

//BEGIN_INCLUDE(all)
#include <initializer_list>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <errno.h>
#include <cassert>

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <android/sensor.h>
#include <android/log.h>
#include <android_native_app_glue.h>

#include <camera/NdkCameraCaptureSession.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraError.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadata.h>
#include <camera/NdkCameraMetadataTags.h>
#include <camera/NdkCameraWindowType.h>
#include <camera/NdkCaptureRequest.h>
#include <android/native_window.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <AppDemoNativeCameraInterface.h>
#include <string>

struct CameraHandler
{
    ACameraManager* _manager;
};

struct Camera
{
    ACameraDevice*                  _device;
    AImageReader*                   _reader;
    ANativeWindow*                  _outputNativeWindow;
    ACaptureSessionOutput*          _sessionOutput;
    ACaptureSessionOutputContainer* _outputContainer;
    ACameraOutputTarget*            _target;
    ACaptureRequest*                _request;
    int                             _status;
    int                             _format;
};

void initCameraHandler(struct CameraHandler** handlerp)
{
    CameraHandler* handler = (CameraHandler*)malloc(sizeof(struct CameraHandler));
    handler->_manager      = ACameraManager_create();
    *handlerp              = handler;
}

unsigned int getCameraList(struct CameraHandler** handlerp, struct CameraInfo** cameraInfop)
{
    ACameraIdList* cameraList = nullptr;

    CameraHandler* handler = (struct CameraHandler*)malloc(sizeof(struct CameraHandler));
    *handlerp              = handler;

    if (ACameraManager_getCameraIdList(handler->_manager, &cameraList) != ACAMERA_OK)
        return 0;

    struct CameraInfo* cameraInfo = (CameraInfo*)malloc(sizeof(CameraInfo) * cameraList->numCameras);
    *cameraInfop                  = cameraInfo;

    for (int i = 0; i < cameraList->numCameras; i++)
    {
        cameraInfo[i]._id = cameraList->cameraIds[i];
        ACameraMetadata* characteristics;
        if (ACameraManager_getCameraCharacteristics(handler->_manager, cameraList->cameraIds[i], &characteristics) == ACAMERA_OK)
        {
            ACameraMetadata_const_entry lensFacing;
            ACameraMetadata_getConstEntry(characteristics, ACAMERA_LENS_FACING, &lensFacing);

            if (*lensFacing.data.u8 == ACAMERA_LENS_FACING_BACK)
            {
                cameraInfo[i].prop = CAMERA_BACKFACING;
            }
            else
            {
                cameraInfo[i].prop = CAMERA_FRONTFACING;
            }
            ACameraMetadata_free(characteristics);
        }
    }
    unsigned int n = cameraList->numCameras;
    ACameraManager_deleteCameraIdList(cameraList);
    return n;
}

unsigned int getBackFacingCameraList(struct CameraHandler* handler, struct CameraInfo** cameraInfop)
{
    ACameraIdList* cameraList = nullptr;

    if (handler == nullptr || cameraInfop == nullptr || ACameraManager_getCameraIdList(handler->_manager, &cameraList) != ACAMERA_OK)
        return 0;

    struct CameraInfo* cameraInfo = (CameraInfo*)malloc(sizeof(CameraInfo) * cameraList->numCameras);
    *cameraInfop                  = cameraInfo;

    unsigned int n = 0;
    for (int i = 0; i < cameraList->numCameras; i++)
    {
        ACameraMetadata* characteristics;
        if (ACameraManager_getCameraCharacteristics(handler->_manager, cameraList->cameraIds[i], &characteristics) == ACAMERA_OK)
        {
            ACameraMetadata_const_entry lensFacing;
            ACameraMetadata_getConstEntry(characteristics, ACAMERA_LENS_FACING, &lensFacing);

            if (*lensFacing.data.u8 == ACAMERA_LENS_FACING_BACK)
            {
                cameraInfo[n]._id    = cameraList->cameraIds[i];
                cameraInfo[n++].prop = CAMERA_BACKFACING;
            }
            ACameraMetadata_free(characteristics);
        }
    }
    ACameraManager_deleteCameraIdList(cameraList);
    return n;
}

void cameraDisconnected(void* context, ACameraDevice* device)
{
    struct Camera* c = (struct Camera*)context;
    ACameraDevice_close(device);
    c->_status = CAMERA_DISCONECTED;
}

void cameraError(void* context, ACameraDevice* device, int error)
{
    struct Camera* c = (struct Camera*)context;
    ACameraDevice_close(device);
    c->_status = CAMERA_ERROR;
}

int initCamera(struct CameraHandler* handler, struct CameraInfo* info, struct Camera** camp)
{
    struct Camera* cam = (struct Camera*)malloc(sizeof(struct Camera));
    *camp              = cam;
    ACameraDevice_StateCallbacks callbacks;
    callbacks.onDisconnected = cameraDisconnected;
    callbacks.onError        = cameraError;

    camera_status_t openResult = ACameraManager_openCamera(handler->_manager, info->_id, &callbacks, &cam->_device);

    int result = false;
    if (openResult != ACAMERA_ERROR_PERMISSION_DENIED)
    {
        result = true;
    }

    return result;
}

/*
void onSessionClosed(void* ctx, ACameraCaptureSession* ses)
{
}
void onSessionReady(void* ctx, ACameraCaptureSession* ses)
{
}
void onSessionActive(void* ctx, ACameraCaptureSession* ses)
{
}
*/
int cameraCaptureSession(struct Camera* cam, int w, int h)
{
    if (AImageReader_new(w, h, AIMAGE_FORMAT_YUV_420_888, 2, &cam->_reader) != AMEDIA_OK)
    {
        return CAMERA_ERROR;
    }
    AImageReader_getWindow(cam->_reader, &cam->_outputNativeWindow);

    // Avoid native window to be deleted
    ANativeWindow_acquire(cam->_outputNativeWindow);

    ACaptureSessionOutput_create(cam->_outputNativeWindow, &cam->_sessionOutput);

    ACaptureSessionOutputContainer_create(&cam->_outputContainer);
    ACaptureSessionOutputContainer_add(cam->_outputContainer, cam->_sessionOutput);

    ACameraOutputTarget_create(cam->_outputNativeWindow, &cam->_target);
    ACameraDevice_createCaptureRequest(cam->_device, TEMPLATE_PREVIEW, &cam->_request);
    //todo change focus

    ACaptureRequest_addTarget(cam->_request, cam->_target);

    ACameraCaptureSession_stateCallbacks capSessionCallbacks;
    //capSessionCallbacks.onActive = onSessionActive;
    //capSessionCallbacks.onReady  = onSessionReady;
    //capSessionCallbacks.onClosed = onSessionClosed;

    ACameraCaptureSession* captureSession;
    if (ACameraDevice_createCaptureSession(cam->_device, cam->_outputContainer, &capSessionCallbacks, &captureSession) != AMEDIA_OK)
    {
        cam->_status = CAMERA_ERROR;
        return CAMERA_ERROR;
    }
    ACameraCaptureSession_setRepeatingRequest(captureSession, nullptr, 1, &cam->_request, nullptr);
    return CAMERA_OK;
}

static const int kMaxChannelValue = 262143;

static inline uint32_t YUV2RGB(int nY, int nU, int nV)
{
    nY -= 16;
    nU -= 128;
    nV -= 128;
    if (nY < 0) nY = 0;

    int nR = (int)(1192 * nY + 1634 * nV);
    int nG = (int)(1192 * nY - 833 * nV - 400 * nU);
    int nB = (int)(1192 * nY + 2066 * nU);

    nR = std::min(kMaxChannelValue, std::max(0, nR));
    nG = std::min(kMaxChannelValue, std::max(0, nG));
    nB = std::min(kMaxChannelValue, std::max(0, nB));

    nR = (nR >> 10) & 0xff;
    nG = (nG >> 10) & 0xff;
    nB = (nB >> 10) & 0xff;

    return 0xff000000 | (nR << 16) | (nG << 8) | nB;
}

static void copyToBuffer(uint8_t* buf, AImage* image)
{
    AImageCropRect srcRect;
    AImage_getCropRect(image, &srcRect);
    int32_t  yStride, uvStride;
    uint8_t *yPixel, *uPixel, *vPixel;
    int32_t  yLen, uLen, vLen;
    AImage_getPlaneRowStride(image, 0, &yStride);
    AImage_getPlaneRowStride(image, 1, &uvStride);
    AImage_getPlaneData(image, 0, &yPixel, &yLen);
    AImage_getPlaneData(image, 1, &uPixel, &uLen);
    AImage_getPlaneData(image, 2, &vPixel, &vLen);
    int32_t uvPixelStride;
    AImage_getPlanePixelStride(image, 1, &uvPixelStride);

    //buf = malloc(yLen + uLen + vLen);
    memcpy(buf, yPixel, yLen);
    memcpy(buf + yLen, uPixel, uLen);
    memcpy(buf + yLen + uLen, vPixel, vLen);
}

static void imageConverter(uint8_t* buf, AImage* image)
{
    AImageCropRect srcRect;
    AImage_getCropRect(image, &srcRect);
    int32_t  yStride, uvStride;
    uint8_t *yPixel, *uPixel, *vPixel;
    int32_t  yLen, uLen, vLen;
    AImage_getPlaneRowStride(image, 0, &yStride);
    AImage_getPlaneRowStride(image, 1, &uvStride);
    AImage_getPlaneData(image, 0, &yPixel, &yLen);
    AImage_getPlaneData(image, 1, &vPixel, &vLen);
    AImage_getPlaneData(image, 2, &uPixel, &uLen);
    int32_t uvPixelStride;
    AImage_getPlanePixelStride(image, 1, &uvPixelStride);

    int32_t height;
    int32_t width;
    AImage_getHeight(image, &height);
    AImage_getWidth(image, &width);

    uint32_t* out = (uint32_t*)(buf);
    for (int32_t row = 0; row < height; row++)
    {
        int32_t        invrow = height - row - 1;
        const uint8_t* pY     = yPixel + srcRect.left + yStride * (invrow + srcRect.top);

        int32_t        uv_row_start = uvStride * ((invrow + srcRect.top) >> 1);
        const uint8_t* pU           = uPixel + uv_row_start + (srcRect.left >> 1);
        const uint8_t* pV           = vPixel + uv_row_start + (srcRect.left >> 1);

        for (int32_t x = 0; x < width; x++)
        {
            const int32_t uv_offset = (x >> 1) * uvPixelStride;
            out[x]                  = YUV2RGB(pY[x], pU[uv_offset], pV[uv_offset]);
        }
        out += width;
    }
}

int cameraLastFrame(struct Camera* cam, unsigned char* imageBuffer)
{
    AImage* image;
    if (AImageReader_acquireLatestImage(cam->_reader, &image) == AMEDIA_OK)
    {
        int32_t format;
        int32_t height;
        int32_t width;
        AImage_getFormat(image, &format);
        AImage_getHeight(image, &height);
        AImage_getWidth(image, &width);
        //imageConverter(imageBuffer, image);
        copyToBuffer(imageBuffer, image);
        AImage_delete(image);
        return CAMERA_OK;
    }
    return CAMERA_ERROR;
}

void destroyCamera(Camera** cam)
{
    ACameraDevice_close((*cam)->_device);
    AImageReader_delete((*cam)->_reader);
    ACaptureSessionOutput_free((*cam)->_sessionOutput);
    ACaptureSessionOutputContainer_free((*cam)->_outputContainer);
    ACameraOutputTarget_free((*cam)->_target);
    ACaptureRequest_free((*cam)->_request);
    free(*cam);
    *cam = nullptr;
}

void destroyCameraHandler(CameraHandler** handler)
{
    ACameraManager_delete((*handler)->_manager);
    free(*handler);
    *handler = nullptr;
}
