//#############################################################################
//  File:      SENSAndroidARCore.h
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SENS_ANDROID_ARCORE_H
#define SENS_ANDROID_ARCORE_H

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <arcore_c_api.h>
#include <opencv2/opencv.hpp>
#include <SENSARBaseCamera.h>
#include <jni.h>

//-----------------------------------------------------------------------------
class SENSGLTextureReader;
//-----------------------------------------------------------------------------
class SENSAndroidARCore : public SENSARBaseCamera
{
public:
    SENSAndroidARCore(JavaVM*     jvm,
                      JNIEnv*     env,
                      jobject     context,
                      jobject     activity,
                      std::string appName,
                      std::string writableDir);
    ~SENSAndroidARCore();

    bool init(unsigned int textureId      = 0,
              bool         retrieveCpuImg = false,
              int          targetWidth    = -1) override;
    bool init(JNIEnv* env, void* context, void* activity);
    bool isReady() override { return _arSession != nullptr; }
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& pose) override;
    void lightComponentIntensity(float* components) override;
    bool checkAvailability(JNIEnv* env, void* context, void* activity);
    bool isAvailable() override;
    bool checkInstalled(JNIEnv* env, void* context, void* activity);
    bool isInstalled() override;
    bool askInstall(JNIEnv* env, void* context, void* activity);
    bool install() override;
    bool installRefused() override { return _installRefused; };
    void installRefused(bool b) override { _installRefused = b; };

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  bool                          provideIntrinsics) override;

    // from camera (does nothing in this case, because we have no image callbacks but we update arcore and the frame actively
    void stop() override { _started = false; };

    const SENSCaptureProps& captureProperties() override;

private:
    void doFetchPointCloud();

    ArSession* _arSession      = nullptr;
    ArFrame*   _arFrame        = nullptr;
    bool       _waitInit       = false;
    bool       _available      = false;
    bool       _installed      = false;
    bool       _installRefused = false;

    GLuint _fbo = 0;
    GLuint _pbo = 0;

    GLuint _cameraTextureId;
    // float          _lightColor[4];
    float   _envLightI[3];
    JavaVM* _jvm;

    // needed to find functions
    std::string _appName;
    std::string _writableDir;

    SENSGLTextureReader* _texImgReader = nullptr;

    cv::Mat convertToYuv(ArImage* arImage);
    void    updateCamera(cv::Mat& intrinsics);

    void retrieveCaptureProperties();

    float _fx = 0.f, _fy = 0.f, _cx = 0.f, _cy = 0.f;
};
//-----------------------------------------------------------------------------
#endif
