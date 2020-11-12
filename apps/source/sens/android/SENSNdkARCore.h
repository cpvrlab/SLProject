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

    bool init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray);
    bool isReady() { return _arSession != nullptr; }
    bool isRunning() { return !_pause; }
    void reset();
	bool resume() override;
	void pause() override;
    bool update(cv::Mat& intrinsic, cv::Mat& view);
    SENSFramePtr latestFrame();

    void setDisplaySize(int w, int h);
    int getCameraOpenGLTexture();
	int getPointCloud(float ** mapPoints, float confidanceValue);

private:

    ANativeActivity* _activity  = nullptr;
    ArSession*       _arSession = nullptr;
    ArFrame*         _arFrame   = nullptr;
    bool             _pause     = true;   
    GLuint           _cameraTextureId;
    std::mutex       _frameMutex;

    void initCameraTexture();
    cv::Mat convertToYuv(ArImage* arImage);
    SENSFramePtr processNewFrame(cv::Mat& bgrImg, cv::Mat intrinsics);
    void updateFrame(cv::Mat& intrinsic);
};

#endif
