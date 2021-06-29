#ifndef SENS_NDK_ARCORE_H
#define SENS_NDK_ARCORE_H

#include <android_native_app_glue.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <arcore_c_api.h>
#include <opencv2/opencv.hpp>
#include <SENSARCore.h>

class SENSNdkARCore : public SENSARCore
{
public:
    SENSNdkARCore(ANativeActivity* activity);
    ~SENSNdkARCore();

    bool init() override;
    bool isReady() override { return _arSession != nullptr; }
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& pose);
    //SENSFramePtr latestFrame() override;
    //void setDisplaySize(int w, int h) override;
    void lightComponentIntensity(float * components);

    //int getCameraOpenGLTexture();
    int getPointCloud(float** mapPoints, float confidanceValue);

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  bool                          provideIntrinsics) override;
    //from camera (does nothing in this case, because we have no image callbacks but we update arcore and the frame actively
    void stop() override { _started = false; };

    const SENSCaptureProperties& captureProperties() override;

private:
    ANativeActivity* _activity  = nullptr;
    ArSession*       _arSession = nullptr;
    ArFrame*         _arFrame   = nullptr;

    GLuint _cameraTextureId;
    float _envLightI[3];
    //float _lightColor[4];

    void    checkAvailability(JNIEnv* env, jobject context);
    void    initCameraTexture();
    cv::Mat convertToYuv(ArImage* arImage);
    void    updateCamera(cv::Mat& intrinsics);

    void retrieveCaptureProperties();
};

#endif
