#include "SENSNdkARCore.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>
#include "SENS.h"
#include "SENSUtils.h"

SENSNdkARCore::SENSNdkARCore(ANativeActivity* activity)
: _activity(activity)
{
}

SENSNdkARCore::~SENSNdkARCore()
{
    if (_arSession != nullptr)
    {
        ArSession_destroy(_arSession);
        ArFrame_destroy(_arFrame);
    }
}

void SENSNdkARCore::initCameraTexture()
{
    glGenTextures(1, &_cameraTextureId);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, _cameraTextureId);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

bool SENSNdkARCore::init(int w, int h, int manipW, int manipH, bool convertManipToGray)
{ 
    if (_arSession == nullptr)
    {
        configure(w, h, manipW, manipH, convertManipToGray);
        JNIEnv* env;
        _activity->vm->GetEnv((void**)&env, JNI_VERSION_1_6);
        _activity->vm->AttachCurrentThread(&env, NULL);

        initCameraTexture();

        jobject         activityObj = env->NewGlobalRef(_activity->clazz);
        ArInstallStatus install_status;

        if (ArCoreApk_requestInstall(env, activityObj, true, &install_status) != AR_SUCCESS)
        {
            env->DeleteGlobalRef(activityObj);
            _activity->vm->DetachCurrentThread();
            return false;
        }

        if (ArSession_create(env, activityObj, &_arSession) != AR_SUCCESS)
        {
            env->DeleteGlobalRef(activityObj);
            _activity->vm->DetachCurrentThread();
            return false;
        }

        if (!_arSession)
        {
            env->DeleteGlobalRef(activityObj);
            _activity->vm->DetachCurrentThread();
            return false;
        }

        ArSession_setDisplayGeometry(_arSession, 0, _config.targetWidth, _config.targetHeight);

        // ----- config -----
        ArConfig* arConfig = nullptr;
        ArConfig_create(_arSession, &arConfig);

        if (!arConfig)
        {
            env->DeleteGlobalRef(activityObj);
            _activity->vm->DetachCurrentThread();
            return false;
        }

        // Deph texture has values between 0 millimeters to 8191 millimeters. 8m is not enough in our case
        // https://developers.google.com/ar/reference/c/group/ar-frame#arframe_acquiredepthimage
        ArConfig_setDepthMode(_arSession, arConfig, AR_DEPTH_MODE_DISABLED);

        ArConfig_setInstantPlacementMode(_arSession, arConfig, AR_INSTANT_PLACEMENT_MODE_DISABLED);

        ArSession_setCameraTextureName(_arSession, _cameraTextureId);

        if (ArSession_configure(_arSession, arConfig) != AR_SUCCESS)
        {
            env->DeleteGlobalRef(activityObj);
            _activity->vm->DetachCurrentThread();
            return false;
        }

        ArConfig_destroy(arConfig);

        // -- arFrame The world state resulting from an update
        // Create with: ArFrame_create
        // Allocate with: ArSession_update
        // Release with: ArFrame_destroy

        ArFrame_create(_arSession, &_arFrame);
        if (!_arFrame)
        {
            env->DeleteGlobalRef(activityObj);
            _activity->vm->DetachCurrentThread();
            return false;
        }

        env->DeleteGlobalRef(activityObj);
        _activity->vm->DetachCurrentThread();
    }
    if (ArSession_resume(_arSession) != AR_SUCCESS)
        return false;

    return true;
}

bool SENSNdkARCore::update(cv::Mat& intrinsic, cv::Mat& view)
{
    if (!_arSession)
        return false;

    if (ArSession_update(_arSession, _arFrame) != AR_SUCCESS)
        return false;

    ArCamera* arCamera;
    ArFrame_acquireCamera(_arSession, _arFrame, &arCamera);

    view = cv::Mat::eye(4, 4, CV_32F);
    intrinsic = cv::Mat::eye(4, 4, CV_32F);

    ArCamera_getViewMatrix(_arSession, arCamera, view.ptr<float>(0));
    ArCamera_getProjectionMatrix(_arSession, arCamera, /*near=*/0.1f, /*far=*/100.f, intrinsic.ptr<float>(0));

    view = view.t();
    intrinsic = intrinsic.t();
    updateFrame(intrinsic);

    ArTrackingState camera_tracking_state;
    ArCamera_getTrackingState(_arSession, arCamera, &camera_tracking_state);

    ArCamera_release(arCamera);
    // If the camera isn't tracking don't bother rendering other objects.
    if (camera_tracking_state != AR_TRACKING_STATE_TRACKING)
        return false;

    return true;
}

void SENSNdkARCore::updateFrame(cv::Mat& intrinsic)
{
    ArImage * arImage;
    if (ArFrame_acquireCameraImage(_arSession, _arFrame, &arImage) != AR_SUCCESS)
        return;

    cv::Mat yuv = convertToYuv(arImage);
    cv::Mat bgr;

    ArImage_release(arImage);

    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21, 3);

    std::lock_guard<std::mutex> lock(_frameMutex);
    _frame = std::make_unique<SENSFrameBase>(bgr, intrinsic);
}

SENSFramePtr SENSNdkARCore::latestFrame()
{
    SENSFrameBasePtr frameBase;
    {
        std::lock_guard<std::mutex> lock(_frameMutex);
        frameBase = _frame;
    }
    SENSFramePtr latestFrame;
    if (frameBase)
        latestFrame = processNewFrame(frameBase->imgBGR, frameBase->intrinsics);
    return latestFrame;
}

SENSFramePtr SENSNdkARCore::processNewFrame(cv::Mat& bgrImg, cv::Mat intrinsics)
{
    //todo: accessing config readonly should be no problem  here, as the config only changes when camera is stopped
    cv::Size inputSize = bgrImg.size();

    // Crop Video image to required aspect ratio
    int cropW = 0, cropH = 0;
    SENS::cropImage(bgrImg, (float)_config.targetWidth / (float)_config.targetHeight, cropW, cropH);

    cv::Mat manipImg;
    float   scale = 1.0f;

    manipImg = bgrImg;
    //problem: eingangsbild 16:9 -> targetImg 4:3 -> crop left and right -> manipImg 16:9 -> weiterer crop oben und unten -> FALSCH
    if (_config.manipWidth > 0 && _config.manipHeight > 0)
    {
        int cropW = 0, cropH = 0;
        SENS::cropImage(manipImg, (float)_config.manipWidth / (float)_config.manipHeight, cropW, cropH);
        scale = (float)_config.manipWidth / (float)manipImg.size().width;
        cv::resize(manipImg, manipImg, cv::Size(), scale, scale);
    }

    // Create grayscale
    if (_config.convertManipToGray)
    {
        cv::cvtColor(manipImg, manipImg, cv::COLOR_BGR2GRAY);
    }

    SENSFramePtr sensFrame = std::make_unique<SENSFrame>(bgrImg,
                                                         manipImg,
                                                         false,
                                                         false,
                                                         1 / scale,
                                                         intrinsics);

    return sensFrame;
}

cv::Mat SENSNdkARCore::convertToYuv(ArImage* arImage)
{
    int32_t height, width, rowStrideY;
    ArImage_getHeight(_arSession, arImage, &height);
    ArImage_getWidth(_arSession, arImage, &width);
    ArImage_getPlaneRowStride(_arSession, arImage, 0, &rowStrideY);

    //pointers to yuv data planes and length of yuv data planes in byte
    const uint8_t *yPixel, *vPixel;
    int32_t  yLen, vLen;
    ArImage_getPlaneData(_arSession, arImage, (int32_t)0, &yPixel, &yLen);
    ArImage_getPlaneData(_arSession, arImage, (int32_t)2, &vPixel, &vLen);

    cv::Mat yuv(height + (height / 2), rowStrideY, CV_8UC1);
    memcpy(yuv.data, yPixel, yLen);
    memcpy(yuv.data + yLen + (rowStrideY - width), vPixel, vLen + 1);

    if (rowStrideY > width)
    {
        cv::Rect roi(0, 0, width, yuv.rows);
        cv::Mat  roiYuv = yuv(roi);
        return roiYuv;
    }
    else
        return yuv;
}

int SENSNdkARCore::getPointCloud(float ** mapPoints, float confidanceValue)
{
    // Update and render point cloud.
    ArPointCloud* arPointCloud = nullptr;
    ArStatus pointCloudStatus = ArFrame_acquirePointCloud(_arSession, _arFrame, &arPointCloud);
    int n;
    float * mp;

    ArPointCloud_getNumberOfPoints(_arSession, arPointCloud, &n);

    ArPointCloud_getData(_arSession, arPointCloud, &mp);

    if (pointCloudStatus != AR_SUCCESS)
        return 0;

    int nbPoints = 0;
    for (int i = 0; i < n; i++)
    {
        int idx = i * 4;
        if (mp[idx+3] >= confidanceValue)
        {
            *mapPoints[nbPoints] = mp[idx];
            *mapPoints[nbPoints + 1] = mp[idx+1];
            *mapPoints[nbPoints + 2] = mp[idx+2];
            nbPoints++;
        }
    }

    ArPointCloud_release(arPointCloud);

    return nbPoints;
}

void SENSNdkARCore::setDisplaySize(int w, int h)
{
    if (_arSession)
        ArSession_setDisplayGeometry(_arSession, 0, w, h);
}

bool SENSNdkARCore::resume()
{
    const ArStatus status = ArSession_resume(_arSession);
    return (status == AR_SUCCESS);
}

void SENSNdkARCore::pause()
{
    if (_arSession != nullptr)
        ArSession_pause(_arSession);
}