#ifndef SENS_NDK_ARCOREJAVA_H
#define SENS_NDK_ARCOREJAVA_H

#include <android_native_app_glue.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <arcore_c_api.h>
#include <opencv2/opencv.hpp>
#include <SENSARCore.h>
#include <jni.h>

class SENSNdkARCoreJava : public SENSARCore
{
public:
    SENSNdkARCoreJava(JavaVM* vm, jobject* activityContext);
    ~SENSNdkARCoreJava();

    bool init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray) override;
    bool isReady() override { return _isReady; }
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& pose);
    //SENSFramePtr latestFrame() override;
    //void setDisplaySize(int w, int h) override;

private:
    JavaVM* _vm     = nullptr;
    jobject _object = nullptr;
    bool _isReady;
};

#endif
