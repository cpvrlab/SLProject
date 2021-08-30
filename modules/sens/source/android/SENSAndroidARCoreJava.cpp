//#############################################################################
//  File:      SENSAndroidARCoreJava.cpp
//  Authors:   Michael Goettlicher, Marcus Hudritsch, Luc Girod
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "SENSARBaseCamera.h"
#include "SENSAndroidARCoreJava.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>

//-----------------------------------------------------------------------------
SENSAndroidARCoreJava::SENSAndroidARCoreJava(JavaVM*  vm,
                                             jobject* activityContext)
  : _vm(vm), _object(*activityContext)
{
    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass    clazz    = env->GetObjectClass(_object);
    jmethodID methodId = env->GetMethodID(clazz,
                                          "ARCoreAvailable",
                                          "()Z");

    _available = env->CallBooleanMethod(_object, methodId);

    _vm->DetachCurrentThread();
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCoreJava::init(int  targetWidth,
                                 nt   targetHeight,
                                 int  manipWidth,
                                 int  manipHeight,
                                 bool convertManipToGray)
{
    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass    clazz    = env->GetObjectClass(_object);
    jmethodID methodId = env->GetMethodID(clazz,
                                          "InitARCore",
                                          "(II)Z");

    bool isReady = env->CallBooleanMethod(_object, methodId, targetWidth, targetHeight);

    _vm->DetachCurrentThread();
    _running = true;
    return _isReady;
}
//-----------------------------------------------------------------------------
void SENSAndroidARCoreJava::reset()
{
    return;
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCoreJava::resume()
{
    Utils::log("AAAAAAAAAAAAAAA", "resume");
    if (_running)
        return true;

    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass    clazz    = env->GetObjectClass(_object);
    jmethodID methodId = env->GetMethodID(clazz,
                                          "ResumeARCore",
                                          "()Z");
    bool      success  = env->CallBooleanMethod(_object, methodId);

    _vm->DetachCurrentThread();

    _running = success;
    Utils::log("AAAAAAAAAAAAAAA", "resumed");
    return _running;
}
//-----------------------------------------------------------------------------
void SENSAndroidARCoreJava::pause()
{
    if (!_running)
        return;
    _running = false;

    // stop locations manager
    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass    clazz    = env->GetObjectClass(_object);
    jmethodID methodId = env->GetMethodID(clazz,
                                          "PauseARCore",
                                          "()V");
    env->CallVoidMethod(_object, methodId);

    _vm->DetachCurrentThread();
}
//-----------------------------------------------------------------------------
bool SENSAndroidARCoreJava::update(cv::Mat& pose)
{
    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);
    jclass    clazz    = env->GetObjectClass(_object);
    jmethodID methodId = env->GetMethodID(clazz,
                                          "UpdateARCore",
                                          "()Z");

    bool tracking = env->CallBooleanMethod(_object, methodId);

    _vm->DetachCurrentThread();

    return tracking;
}
//-----------------------------------------------------------------------------
