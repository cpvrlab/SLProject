//#############################################################################
//  File:      SENSAndroidARCore.cpp
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "SENSAndroidARCore.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>
#include "SENS.h"
#include "SENSUtils.h"
#include "SENSGLTextureReader.h"

//-----------------------------------------------------------------------------
SENSAndroidARCore::SENSAndroidARCore(JavaVM*     jvm,
                                     JNIEnv*     env,
                                     jobject     context,
                                     jobject     activity,
                                     std::string appName,
                                     std::string writableDir)
{
    checkAvailability(env, context, activity);
    _arSession = nullptr;
    _jvm       = jvm;

    _appName     = appName;
    _writableDir = writableDir;
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCore::checkAvailability(JNIEnv* env,
                                          void*   context,
                                          void*   activity)
{
    ArAvailability availability;
    ArCoreApk_checkAvailability(env, context, &availability);

    if (availability == AR_AVAILABILITY_UNKNOWN_CHECKING)
    {
        // TODO(dgj1): this could theoretically go on forever (or until the stack-memory is all used up).
        // add a limit to recursion?
        std::this_thread::sleep_for(std::chrono::microseconds((int)(200000)));
        checkAvailability(env, context, activity);
    }
    else if (availability == AR_AVAILABILITY_SUPPORTED_NOT_INSTALLED ||
             availability == AR_AVAILABILITY_SUPPORTED_APK_TOO_OLD ||
             availability == AR_AVAILABILITY_SUPPORTED_INSTALLED)
    {
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCore::isAvailable()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    std::string className = "ch/cpvr/" + _appName + "/GLES3Lib";
    jclass      clazz     = env->FindClass(className.c_str());
    jmethodID   methodid  = env->GetStaticMethodID(clazz, "checkARAvailability", "()Z");

    _available = env->CallStaticBooleanMethod(clazz, methodid);
    return _available;
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCore::checkInstalled(JNIEnv* env,
                                       void*   context,
                                       void*   activity)
{
    ArAvailability availability;
    ArCoreApk_checkAvailability(env, context, &availability);
    if (availability == AR_AVAILABILITY_SUPPORTED_INSTALLED)
        return true;
    return false;
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCore::isInstalled()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    std::string className = "ch/cpvr/" + _appName + "/GLES3Lib";
    jclass      clazz     = env->FindClass(className.c_str());
    jmethodID   methodid  = env->GetStaticMethodID(clazz, "checkARInstalled", "()Z");

    _installed = env->CallStaticBooleanMethod(clazz, methodid);
    return _installed;
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCore::askInstall(JNIEnv* env, void* context, void* activity)
{
    ArInstallStatus install_status;
    ArCoreApk_requestInstall(env, activity, true, &install_status);
    if (install_status == AR_AVAILABILITY_SUPPORTED_INSTALLED)
    {
        _installed = true;
        return true;
    }

    _installed = false;
    return false;
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCore::install()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    std::string className = "ch/cpvr/" + _appName + "/GLES3Lib";
    jclass      clazz     = env->FindClass(className.c_str());
    jmethodID   methodid  = env->GetStaticMethodID(clazz, "askARInstall", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}
//-----------------------------------------------------------------------------
SENSAndroidARCore::~SENSAndroidARCore()
{
    reset();
}
//-----------------------------------------------------------------------------
void SENSAndroidARCore::reset()
{
    if (_arSession != nullptr)
    {
        pause();
        ArSession_destroy(_arSession);
        ArFrame_destroy(_arFrame);
        _arSession = nullptr;
        if (_texImgReader)
        {
            delete _texImgReader;
            _texImgReader = nullptr;
        }
    }
}
//-----------------------------------------------------------------------------
// NOTE(dgj1): targetHeight is automatically calculated based on reported aspect ratio of GPU texture
bool SENSAndroidARCore::init(unsigned int textureId,
                             bool         retrieveCpuImg,
                             int          targetWidth)
{
    if (textureId > 0)
        _cameraTextureId = textureId;

    if (_texImgReader)
    {
        delete _texImgReader;
        _texImgReader = nullptr;
    }

    _retrieveCpuImg    = retrieveCpuImg;
    _cpuImgTargetWidth = targetWidth;

    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    std::string className = "ch/cpvr/" + _appName + "/GLES3Lib";
    jclass      clazz     = env->FindClass(className.c_str());
    jmethodID   methodid  = env->GetStaticMethodID(clazz, "initAR", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCore::init(JNIEnv* env, void* context, void* activity)
{
    if (!checkAvailability(env, context, activity)) return false;
    if (_arSession != nullptr) return false;

    ArInstallStatus install_status;
    ArCoreApk_requestInstall(env, activity, false, &install_status);

    if (install_status == AR_AVAILABILITY_SUPPORTED_NOT_INSTALLED) return false;
    if (ArSession_create(env, activity, &_arSession) != AR_SUCCESS) return false;
    if (!_arSession) return false;

    // todo: what do we have to set here? the display size? is it needed at all for our case?
    // ArSession_setDisplayGeometry(_arSession, 0, _config.targetWidth, _config.targetHeight);

    // ----- config -----
    ArConfig* arConfig = nullptr;
    ArConfig_create(_arSession, &arConfig);

    if (!arConfig)
    {
        ArSession_destroy(_arSession);
        _arSession = nullptr;
        return false;
    }

    // Depth texture has values between 0 millimeters to 8191 millimeters. 8m is not enough in our case
    // https://developers.google.com/ar/reference/c/group/ar-frame#arframe_acquiredepthimage
    ArConfig_setDepthMode(_arSession, arConfig, AR_DEPTH_MODE_DISABLED);
    ArConfig_setLightEstimationMode(_arSession, arConfig, AR_LIGHT_ESTIMATION_MODE_ENVIRONMENTAL_HDR);
    ArConfig_setInstantPlacementMode(_arSession, arConfig, AR_INSTANT_PLACEMENT_MODE_DISABLED);

    ArSession_setCameraTextureName(_arSession, _cameraTextureId);

    if (ArSession_configure(_arSession, arConfig) != AR_SUCCESS)
    {
        ArConfig_destroy(arConfig);
        ArSession_destroy(_arSession);
        _arSession = nullptr;
        return false;
    }
    _envLightI[0] = 1.0f;
    _envLightI[1] = 1.0f;
    _envLightI[2] = 1.0f;

    ArConfig_destroy(arConfig);

    // -- arFrame The world state resulting from an update
    // Create with: ArFrame_create
    // Allocate with: ArSession_update
    // Release with: ArFrame_destroy

    ArFrame_create(_arSession, &_arFrame);
    if (!_arFrame)
    {
        ArSession_destroy(_arSession);
        _arSession = nullptr;
        return false;
    }

    // retrieve intrinsics and frame size
    ArCamera* arCamera;
    ArFrame_acquireCamera(_arSession, _arFrame, &arCamera);

    ArCameraIntrinsics* arIntrinsics = nullptr;
    ArCameraIntrinsics_create(_arSession, &arIntrinsics);
    ArCamera_getTextureIntrinsics(_arSession, arCamera, arIntrinsics);

    ArCameraIntrinsics_getFocalLength(_arSession, arIntrinsics, &_fx, &_fy);
    ArCameraIntrinsics_getPrincipalPoint(_arSession, arIntrinsics, &_cx, &_cy);
    ArCameraIntrinsics_getImageDimensions(_arSession,
                                          arIntrinsics,
                                          &_inputFrameW,
                                          &_inputFrameH);

    ArCameraIntrinsics_destroy(arIntrinsics);
    ArCamera_release(arCamera);

    if (_cpuImgTargetWidth > 0)
    {
        float aspect        = (float)_inputFrameH / (float)_inputFrameW;
        _cpuImgTargetHeight = aspect * _cpuImgTargetWidth;
    }

    pause();
    return true;
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCore::update(cv::Mat& pose)
{
    if (!_arSession) return false;
    ArStatus status = ArSession_update(_arSession, _arFrame);
    if (status != AR_SUCCESS) return false;

    ArCamera* arCamera;
    ArFrame_acquireCamera(_arSession, _arFrame, &arCamera);

    cv::Mat view = cv::Mat::eye(4, 4, CV_32F);
    ArCamera_getViewMatrix(_arSession, arCamera, view.ptr<float>(0));

    // conversions to sl camera pose:
    // from row- to column-major
    view = view.t();
    // inversion
    cv::Mat wRc = (view.rowRange(0, 3).colRange(0, 3)).t();
    cv::Mat wtc = -wRc * view.rowRange(0, 3).col(3);
    cv::Mat wTc = cv::Mat::eye(4, 4, CV_32F);
    wRc.copyTo(wTc.colRange(0, 3).rowRange(0, 3));
    wtc.copyTo(wTc.rowRange(0, 3).col(3));
    // axis direction adaption (x = -y, y = x, z = z, t = t)
    pose = cv::Mat::eye(4, 4, CV_32F);
    wTc.col(1) *= -1;
    wTc.col(1).copyTo(pose.col(0));
    wTc.col(0).copyTo(pose.col(1));
    wTc.col(2).copyTo(pose.col(2));
    wTc.col(3).copyTo(pose.col(3));

    cv::Mat intrinsics = cv::Mat::eye(3, 3, CV_64F);
    {
        ArCameraIntrinsics* arIntrinsics = nullptr;
        ArCameraIntrinsics_create(_arSession, &arIntrinsics);
        ArCamera_getTextureIntrinsics(_arSession, arCamera, arIntrinsics);

        float fx, fy, cx, cy;
        ArCameraIntrinsics_getFocalLength(_arSession, arIntrinsics, &fx, &fy);
        ArCameraIntrinsics_getPrincipalPoint(_arSession, arIntrinsics, &cx, &cy);

        ArCameraIntrinsics_getImageDimensions(_arSession,
                                              arIntrinsics,
                                              &_inputFrameW,
                                              &_inputFrameH);

        ArCameraIntrinsics_destroy(arIntrinsics);

        intrinsics.at<double>(0, 0) = fx;
        intrinsics.at<double>(1, 1) = fy;
        intrinsics.at<double>(0, 2) = cx;
        intrinsics.at<double>(1, 2) = cy;
        /*
        std::stringstream ss;
        ss << intrinsics;
        Utils::log("SENSAndroidARCore", "intrinsics %s", ss.str().c_str());
         */
    }

    updateCamera(intrinsics);

    //---- LIGHT ESTIMATE ----//
    // Get light estimation value.
    /*
    ArLightEstimate*     arLightEstimate;
    ArLightEstimateState arLightEstimateState;
    ArLightEstimate_create(_arSession, &arLightEstimate);

    ArFrame_getLightEstimate(_arSession, _arFrame, arLightEstimate);
    ArLightEstimate_getState(_arSession, arLightEstimate, &arLightEstimateState);

    // Set light intensity to default. Intensity value ranges from 0.0f to 1.0f.
    // The first three components are color scaling factors.
    // The last one is the average pixel intensity in gamma space.
    if (arLightEstimateState == AR_LIGHT_ESTIMATE_STATE_VALID)
    {
        //ArLightEstimate_getColorCorrection(_arSession, arLightEstimate, _lightColor);
        ArLightEstimate_getEnvironmentalHdrMainLightIntensity(_arSession, arLightEstimate, _envLightI);
    }

    ArLightEstimate_destroy(arLightEstimate);
    arLightEstimate = nullptr;
     */
    //---------------------

    ArTrackingState camera_tracking_state;
    ArCamera_getTrackingState(_arSession, arCamera, &camera_tracking_state);

    if (_fetchPointCloud)
        doFetchPointCloud();

    ArCamera_release(arCamera);
    // If the camera isn't tracking don't bother rendering other objects.
    if (camera_tracking_state != AR_TRACKING_STATE_TRACKING)
        return false;

    return true;
}
//-----------------------------------------------------------------------------
void SENSAndroidARCore::updateCamera(cv::Mat& intrinsics)
{
    cv::Mat image;

    if (_retrieveCpuImg)
    {
        if (!_texImgReader)
        {
            // ATTENTION: for the current implementation the gpu texture (preview image) has to have the same aspect ratio as the cpu image
            // check aspect ratio
            _texImgReader = new SENSGLTextureReader(_cameraTextureId, true, _cpuImgTargetWidth, _cpuImgTargetHeight);
        }

        HighResTimer t;
        image = _texImgReader->readImageFromGpu();

        Utils::log("SENSAndroidARCore", "readImageFromGPU: %fms", t.elapsedTimeInMilliSec());
        /*
        Utils::log("imagesize", "orig w:%d h:%d", image.size().width, image.size().height);
        Utils::log("imagesize", "orig imageSize w:%d h:%d", _inputFrameW, _inputFrameH);
        Utils::log("imagesize", "orig calib cx:%f cy:%f", intrinsics.at<double>(0, 2),
                   intrinsics.at<double>(1, 2));
        */
        // adapt intrinsics to cpu image size
    }

    updateFrame(image, intrinsics, _inputFrameW, _inputFrameH, true);
}
//-----------------------------------------------------------------------------
void SENSAndroidARCore::lightComponentIntensity(float* component)
{
    component[0] = _envLightI[0];
    component[1] = _envLightI[1];
    component[2] = _envLightI[2];
}
//-----------------------------------------------------------------------------
cv::Mat SENSAndroidARCore::convertToYuv(ArImage* arImage)
{
    int32_t height, width, rowStrideY;
    ArImage_getHeight(_arSession, arImage, &height);
    ArImage_getWidth(_arSession, arImage, &width);
    ArImage_getPlaneRowStride(_arSession, arImage, 0, &rowStrideY);

    // pointers to yuv data planes and length of yuv data planes in byte
    const uint8_t *yPixel, *vPixel;
    int32_t        yLen, vLen;
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
//-----------------------------------------------------------------------------
bool SENSAndroidARCore::resume()
{
    if (_pause && _arSession != nullptr)
    {
        const ArStatus status = ArSession_resume(_arSession);
        if (status == AR_SUCCESS)
        {
            _pause   = false;
            _started = true; // for SENSBaseCamera
        }
        else
            Utils::log("ErlebAR", "SENSAndroidARCore resume failed!!!");
    }
    return !_pause;
}
//-----------------------------------------------------------------------------
void SENSAndroidARCore::pause()
{
    _started = false; // for SENSBaseCamera
    if (!_arSession) return;

    if (AR_SUCCESS == ArSession_pause(_arSession))
        _pause = true;
    else
        Utils::log("ErlebAR", "SENSAndroidARCore pause failed!!!");
}
//-----------------------------------------------------------------------------
void SENSAndroidARCore::retrieveCaptureProperties()
{
    // the SENSBaseCamera needs to have a valid frame, otherwise we cannot estimate the fov correctly
    /*
    if(!_frame)
    {
        resume();
        HighResTimer t;
        cv::Mat pose;
        do {
            update(pose);
            std::this_thread::sleep_for(200ms);
        }
        while(!_frame && t.elapsedTimeInSec() < 10.f);

        Utils::log("SENSAndroidARCore", "retrieveCaptureProperties update for %fs", t.elapsedTimeInSec());
        pause();
    }

    if(_frame)
    {
        std::string      deviceId = "ARKit";
        SENSCameraFacing facing = SENSCameraFacing::BACK;

        float focalLengthPix = -1.f;
        if(!_frame->intrinsics.empty())
        {
            focalLengthPix = 0.5 * (_frame->intrinsics.at<double>(0, 0) + _frame->intrinsics.at<double>(1, 1));
        }
        SENSCameraDeviceProps devProp(deviceId, facing);
        //here we have to use the cpu image size (which is not the same as the gpu image size)
        devProp.add(_inputFrameW, _inputFrameH, focalLengthPix);
        _captureProperties.push_back(devProp);
    }
    else
        Utils::warnMsg("SENSAndroidARCore", "retrieveCaptureProperties: Could not retrieve a valid frame!", __LINE__, __FILE__);
    */
    std::string      deviceId = "ARKit";
    SENSCameraFacing facing   = SENSCameraFacing::BACK;

    float                 focalLengthPix = 0.5 * (_fx + _fy);
    SENSCameraDeviceProps devProp(deviceId, facing);

    // here we have to use the cpu image size (which is not the same as the gpu image size)
    devProp.add(_inputFrameW, _inputFrameH, focalLengthPix);
    _captureProperties.push_back(devProp);
}
//-----------------------------------------------------------------------------
const SENSCaptureProps& SENSAndroidARCore::captureProperties()
{
    if (_captureProperties.size() == 0)
        retrieveCaptureProperties();

    return _captureProperties;
}
//-----------------------------------------------------------------------------
//! This function is needed to correctly use arcore as a camera in SENSCVCamera
/*! This function does not really start the camera as for the arcore
 * implementation, the frame gets only updated with a call to arcore::update.
 */
const SENSCameraConfig& SENSAndroidARCore::start(std::string                   deviceId,
                                                 const SENSCameraStreamConfig& streamConfig,
                                                 bool                          provideIntrinsics)
{
    // define capture properties
    if (_captureProperties.size() == 0)
        retrieveCaptureProperties();

    if (_captureProperties.size() == 0)
        throw SENSException(SENSType::CAM,
                            "Could not retrieve camera properties!",
                            __LINE__,
                            __FILE__);

    if (!_captureProperties.containsDeviceId(deviceId))
        throw SENSException(SENSType::CAM,
                            "DeviceId does not exist!",
                            __LINE__,
                            __FILE__);

    SENSCameraFacing             facing = SENSCameraFacing::UNKNOWN;
    const SENSCameraDeviceProps* props  = _captureProperties.camPropsForDeviceId(deviceId);
    if (props)
        facing = props->facing();

    // init config here before processStart
    _config = SENSCameraConfig(deviceId,
                               streamConfig,
                               facing,
                               SENSCameraFocusMode::CONTINIOUS_AUTO_FOCUS);
    // inform camera listeners
    processStart();

    _started = true;
    return _config;
}
//-----------------------------------------------------------------------------
void SENSAndroidARCore::doFetchPointCloud()
{
    ArPointCloud* arPointCloud = nullptr;

    if (ArFrame_acquirePointCloud(_arSession,
                                  _arFrame,
                                  &arPointCloud) == AR_SUCCESS)
    {
        int    n;
        float* mp;

        ArPointCloud_getNumberOfPoints(_arSession, arPointCloud, &n);
        if (n > 0)
        {
            ArPointCloud_getData(_arSession, arPointCloud, &mp);

            _pointCloud = cv::Mat(n, 4, CV_32F);
            std::memcpy(_pointCloud.data, mp, n * 4 * sizeof(float));
            /*
            int nbPoints = 0;
            for (int i = 0; i < n; i++)
            {
                int idx = i * 4;
                if (mp[idx + 3] >= confidanceValue)
                {
                    *mapPoints[nbPoints] = mp[idx];
                    *mapPoints[nbPoints + 1] = mp[idx + 1];
                    *mapPoints[nbPoints + 2] = mp[idx + 2];
                    nbPoints++;
                }
            }
            */
        }
        ArPointCloud_release(arPointCloud);
    }
}
//-----------------------------------------------------------------------------
