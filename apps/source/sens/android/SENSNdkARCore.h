#ifndef SENS_NDK_ARCORE_H
#define SENS_NDK_ARCORE_H

#include <android_native_app_glue.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <arcore_c_api.h>
#include <opencv2/opencv.hpp>
#include <sens/SENSARCore.h>

class SENSNdkARCore : public SENSARCore
{
public:
    SENSNdkARCore(ANativeActivity* activity);
    ~SENSNdkARCore();

    bool init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray) override;
    bool isReady() override { return _arSession != nullptr; }
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& pose);
    //SENSFramePtr latestFrame() override;
    //void setDisplaySize(int w, int h) override;

    //int getCameraOpenGLTexture();
    int getPointCloud(float** mapPoints, float confidanceValue);

private:
    ANativeActivity* _activity  = nullptr;
    ArSession*       _arSession = nullptr;
    ArFrame*         _arFrame   = nullptr;

    GLuint _cameraTextureId;

    void    initCameraTexture();
    cv::Mat convertToYuv(ArImage* arImage);
    void    updateFrame(cv::Mat& intrinsics);
};

#endif
