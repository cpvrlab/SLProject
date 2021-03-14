#include "SENSNdkARCore.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>
#include "SENS.h"
#include "SENSUtils.h"

SENSNdkARCore::SENSNdkARCore(JavaVM* jvm, JNIEnv* env, jobject context, jobject activity)
{
    checkAvailability(env, context, activity);
    _arSession = nullptr;
    _waitInit = false;
    _context = env->NewGlobalRef((jobject)context);;
    _activity = activity;
    _jvm = jvm;
    gActivity = env->NewGlobalRef(activity);
    Utils::log("ErlebAR", "init request install  %p", _activity);

}

void SENSNdkARCore::checkAvailability(JNIEnv* env, void* context, void * activity)
{
    ArAvailability availability;
    ArCoreApk_checkAvailability(env, context, &availability);

    _available = true;
    if (availability == AR_AVAILABILITY_UNKNOWN_CHECKING)
    {
        std::this_thread::sleep_for(std::chrono::microseconds((int)(200000)));
        checkAvailability(env, context, activity);
    }
    else if (availability == AR_AVAILABILITY_SUPPORTED_NOT_INSTALLED ||
             availability == AR_AVAILABILITY_SUPPORTED_APK_TOO_OLD   ||
             availability == AR_AVAILABILITY_SUPPORTED_INSTALLED)
    {
        Utils::log("ErlebAR", "request install");
        ArInstallStatus install_status;
        ArCoreApk_requestInstall(env, activity, true, &install_status);
    }
    else
    {
        _available = false;
    }
}

SENSNdkARCore::~SENSNdkARCore()
{
    reset();
}

void SENSNdkARCore::reset()
{
    if (_arSession != nullptr)
    {
        pause();
        ArSession_destroy(_arSession);
        ArFrame_destroy(_arFrame);
        _arSession = nullptr;
        glDeleteTextures(1, &_cameraTextureId);
    }
}

void SENSNdkARCore::initCameraTexture()
{
    glGenTextures(1, &_cameraTextureId);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, _cameraTextureId);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

bool SENSNdkARCore::waitInit()
{
    return _waitInit;
}

bool SENSNdkARCore::init()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "initAR", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSNdkARCore::init(JNIEnv* env, void* context, void* activity)
{
    if (!_available) {
        return false;
    }
    if (_arSession != nullptr) {
        return false;
    }

    ArInstallStatus install_status;
    ArCoreApk_requestInstall(env, activity, false, &install_status);

    if (install_status == AR_AVAILABILITY_SUPPORTED_NOT_INSTALLED)
    {
        return false;
    }

    if (ArSession_create(env, activity, &_arSession) != AR_SUCCESS)
    {
        return false;
    }

    if (!_arSession)
    {
        return false;
    }

    //todo: what do we have to set here? the display size? is it needed at all for our case?
    //ArSession_setDisplayGeometry(_arSession, 0, _config.targetWidth, _config.targetHeight);

    // ----- config -----
    ArConfig* arConfig = nullptr;
    ArConfig_create(_arSession, &arConfig);

    if (!arConfig)
    {
        ArSession_destroy(_arSession);
        _arSession = nullptr;
        return false;
    }

    // Deph texture has values between 0 millimeters to 8191 millimeters. 8m is not enough in our case
    // https://developers.google.com/ar/reference/c/group/ar-frame#arframe_acquiredepthimage
    ArConfig_setDepthMode(_arSession, arConfig, AR_DEPTH_MODE_DISABLED);
    ArConfig_setLightEstimationMode(_arSession, arConfig, AR_LIGHT_ESTIMATION_MODE_ENVIRONMENTAL_HDR);
    ArConfig_setInstantPlacementMode(_arSession, arConfig, AR_INSTANT_PLACEMENT_MODE_DISABLED);

    initCameraTexture();
    ArSession_setCameraTextureName(_arSession, _cameraTextureId);

    if (ArSession_configure(_arSession, arConfig) != AR_SUCCESS)
    {
        ArConfig_destroy(arConfig);
        ArSession_destroy(_arSession);
        _arSession = nullptr;
        return false;
    }

    /*
    _lightColor[0] = 1.0f;
    _lightColor[1] = 1.0f;
    _lightColor[2] = 1.0f;
    _lightColor[3] = 1.0f;
    */
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

    pause();
    return true;
}

bool SENSNdkARCore::update(cv::Mat& pose)
{
    if (!_arSession)
        return false;

    if (ArSession_update(_arSession, _arFrame) != AR_SUCCESS)
        return false;

    ArCamera* arCamera;
    ArFrame_acquireCamera(_arSession, _arFrame, &arCamera);

    cv::Mat view = cv::Mat::eye(4, 4, CV_32F);
    ArCamera_getViewMatrix(_arSession, arCamera, view.ptr<float>(0));

    //convertions to sl camera pose:
    //from row- to column-major
    view = view.t();
    //inversion
    cv::Mat wRc = (view.rowRange(0, 3).colRange(0, 3)).t();
    cv::Mat wtc = -wRc * view.rowRange(0, 3).col(3);
    cv::Mat wTc = cv::Mat::eye(4, 4, CV_32F);
    wRc.copyTo(wTc.colRange(0, 3).rowRange(0, 3));
    wtc.copyTo(wTc.rowRange(0, 3).col(3));
    //axis direction adaption (x = -y, y = x, z = z, t = t)
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
        ArCamera_getImageIntrinsics(_arSession, arCamera, arIntrinsics);

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
        Utils::log("SENSNdkARCore", "intrinsics %s", ss.str().c_str());
         */
    }

    updateCamera(intrinsics);

    //---- LIGHT ESTIMATE ----//
    // Get light estimation value.
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
    //---------------------

    ArTrackingState camera_tracking_state;
    ArCamera_getTrackingState(_arSession, arCamera, &camera_tracking_state);

    ArCamera_release(arCamera);
    // If the camera isn't tracking don't bother rendering other objects.
    if (camera_tracking_state != AR_TRACKING_STATE_TRACKING)
        return false;

    return true;
}

void SENSNdkARCore::updateCamera(cv::Mat& intrinsics)
{
    ArImage* arImage;
    if (ArFrame_acquireCameraImage(_arSession, _arFrame, &arImage) != AR_SUCCESS)
        return;

    cv::Mat yuv = convertToYuv(arImage);
    cv::Mat bgr;

    ArImage_release(arImage);

    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21, 3);

    updateFrame(bgr, intrinsics, true);
}



void SENSNdkARCore::lightComponentIntensity(float * component)
{
    component[0] = _envLightI[0];
    component[1] = _envLightI[1];
    component[2] = _envLightI[2];
}
cv::Mat SENSNdkARCore::convertToYuv(ArImage* arImage)
{
    int32_t height, width, rowStrideY;
    ArImage_getHeight(_arSession, arImage, &height);
    ArImage_getWidth(_arSession, arImage, &width);
    ArImage_getPlaneRowStride(_arSession, arImage, 0, &rowStrideY);

    //pointers to yuv data planes and length of yuv data planes in byte
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

bool SENSNdkARCore::resume()
{
    if (_pause && _arSession != nullptr)
    {
        const ArStatus status = ArSession_resume(_arSession);
        if (status == AR_SUCCESS)
        {
            _pause = false;
            _started = true; //for SENSCameraBase
        }
    }
    return !_pause;
}

void SENSNdkARCore::pause()
{
    _pause = true;
    _started = false; //for SENSCameraBase
    if (_arSession != nullptr)
        if( AR_SUCCESS == ArSession_pause(_arSession))
            Utils::log("SENSNdkARCore", "success");
}

void SENSNdkARCore::retrieveCaptureProperties()
{
    if (!_started || _waitInit)
    {

        return;
    }
    //the SENSCameraBase needs to have a valid frame, otherwise we cannot estimate the fov correctly
    if(!_frame)
    {
        resume();
        HighResTimer t;
        cv::Mat pose;
        do {
            update(pose);
        }
        while(!_frame && t.elapsedTimeInSec() < 5.f);

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
        SENSCameraDeviceProperties devProp(deviceId, facing);
        devProp.add(_frame->imgBGR.cols, _frame->imgBGR.rows, focalLengthPix);
        _captureProperties.push_back(devProp);
    }
    else
        Utils::warnMsg("SENSiOSARCore", "retrieveCaptureProperties: Could not retrieve a valid frame!", __LINE__, __FILE__);
}

const SENSCaptureProperties& SENSNdkARCore::captureProperties()
{
    if(_captureProperties.size() == 0)
        retrieveCaptureProperties();

    return _captureProperties;
}

//This function does not really start the camera as for the arcore iplementation, the frame gets only updated with a call to arcore::update.
//This function is needed to correctly use arcore as a camera in SENSCVCamera
const SENSCameraConfig& SENSNdkARCore::start(std::string    deviceId,
                                             const SENSCameraStreamConfig& streamConfig,
                                             bool                          provideIntrinsics)
{
    //define capture properties
    if(_captureProperties.size() == 0)
        retrieveCaptureProperties();

    if (_captureProperties.size() == 0)
        throw SENSException(SENSType::CAM, "Could not retrieve camera properties!", __LINE__, __FILE__);

    if (!_captureProperties.containsDeviceId(deviceId))
        throw SENSException(SENSType::CAM, "DeviceId does not exist!", __LINE__, __FILE__);

    SENSCameraFacing                  facing = SENSCameraFacing::UNKNOWN;
    const SENSCameraDeviceProperties* props  = _captureProperties.camPropsForDeviceId(deviceId);
    if (props)
        facing = props->facing();

    //init config here before processStart
    _config = SENSCameraConfig(deviceId,
                               streamConfig,
                               facing,
                               SENSCameraFocusMode::CONTINIOUS_AUTO_FOCUS);
    //inform camera listeners
    processStart();

    _started = true;
    return _config;
}


