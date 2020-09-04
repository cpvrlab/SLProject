#include "SENSNdkGps.h"
#include <jni.h>

#include <Utils.h>

SENSNdkGps::SENSNdkGps(JavaVM* vm, jobject* activityContext, jclass* clazz)
  : _vm(vm)
{
    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    //allocate object SENSGps java class
    jobject o = env->AllocObject(*clazz);
    _object = env->NewGlobalRef(o);

    //set java activity context
    //jclass clazz = env->GetObjectClass(_object);
    jmethodID methodId0 = env->GetMethodID(*clazz,
                                           "init",
                                           "(Landroid/content/Context;)V");
    env->CallVoidMethod(_object, methodId0, *_context);

    _vm->DetachCurrentThread();
}

void SENSNdkGps::init(bool granted)
{
    Utils::log("SENSNdkGps", "init called");
    _permissionGranted = granted;
}

bool SENSNdkGps::start()
{
    if(!_permissionGranted)
        return false;

    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass clazz = env->GetObjectClass(_object);
    jmethodID methodId = env->GetMethodID(clazz,
                                          "start",
                                          "()V");
    env->CallVoidMethod(_object, methodId);

    _vm->DetachCurrentThread();

    return true;
}

void SENSNdkGps::stop()
{
    //stop locations manager
}

