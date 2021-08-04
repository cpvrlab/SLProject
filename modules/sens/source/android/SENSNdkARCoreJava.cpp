#include "SENSARCore.h"
#include "SENSNdkARCoreJava.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>

/*
static SENSNdkARCoreJava* gARCore = nullptr;
SENSNdkARCoreJava*        GetARCorePtr()
{
    if (gARCore == nullptr)
        Utils::log("SENSNdkARCoreJava", "Global ARCoreJava has not been initialized");
    return gARCore;
}
*/

SENSNdkARCoreJava::SENSNdkARCoreJava(JavaVM* vm, jobject* activityContext)
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

SENSNdkARCoreJava::~SENSNdkARCoreJava()
{}

bool SENSNdkARCoreJava::init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray)
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

void SENSNdkARCoreJava::reset()
{
    return;
}

bool SENSNdkARCoreJava::resume()
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
    bool success  = env->CallBooleanMethod(_object, methodId);

    _vm->DetachCurrentThread();

    _running = success;
    Utils::log("AAAAAAAAAAAAAAA", "resumed");
    return _running;
}

void SENSNdkARCoreJava::pause()
{
    if (!_running)
        return;
    _running = false;

    //stop locations manager
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

bool SENSNdkARCoreJava::update(cv::Mat& pose)
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

    //cv::Mat view = cv::Mat::eye(4, 4, CV_32F);

    return tracking;
}
