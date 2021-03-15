#ifndef SENS_NDK_ARCORE_H
#define SENS_NDK_ARCORE_H

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <arcore_c_api.h>
#include <opencv2/opencv.hpp>
#include <SENSARCore.h>
#include <jni.h>

class SENSNdkARCore : public SENSARCore
{
public:
    SENSNdkARCore(JavaVM* jvm, JNIEnv* env, jobject context, jobject activity);
    ~SENSNdkARCore();

    bool init() override;
    bool init(JNIEnv* env, void* context, void* activity);
    bool isReady() override { return _arSession != nullptr; }
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& pose);
    bool waitInit();
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
    ArSession*       _arSession = nullptr;
    ArFrame*         _arFrame   = nullptr;
    bool _waitInit = false;

    GLuint _cameraTextureId;
	//float          _lightColor[4];
	float            _envLightI[3];
	void * _context;
	void * _activity;
    JavaVM* _jvm;
    jobject gActivity;

    void    checkAvailability(JNIEnv* env, void* context, void * activity);
    void    initCameraTexture();
    cv::Mat convertToYuv(ArImage* arImage);
    void    updateCamera(cv::Mat& intrinsics);

    void retrieveCaptureProperties();
};

#endif
