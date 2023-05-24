//#############################################################################
//  File:      SENSAndroidARCoreJava.h
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SENS_ANDROID_ARCOREJAVA_H
#define SENS_ANDROID_ARCOREJAVA_H

#include <android_native_app_glue.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <arcore_c_api.h>
#include <opencv2/opencv.hpp>
#include <SENSARBaseCamera.h>
#include <jni.h>

//-----------------------------------------------------------------------------
class SENSAndroidARCoreJava : public SENSARBaseCamera
{
public:
    SENSAndroidARCoreJava(JavaVM* vm, jobject* activityContext);
    ~SENSAndroidARCoreJava() { ; }

    bool init(int  targetWidth,
              int  targetHeight,
              int  manipWidth,
              int  manipHeight,
              bool convertManipToGray) override;
    bool isReady() override { return _isReady; }
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& pose) override;

private:
    JavaVM* _vm     = nullptr;
    jobject _object = nullptr;
    bool    _isReady;
};
//-----------------------------------------------------------------------------
#endif
