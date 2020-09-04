#include "SENSNdkGps.h"
#include <jni.h>

#include <Utils.h>

SENSNdkGps::SENSNdkGps(JavaVM* vm, jobject* activityContext, jclass* clazz, jobject* object)
  : _vm(vm),
    _clazz(clazz),
    _object(object),
    _context(activityContext)
{
}

void SENSNdkGps::init(bool granted)
{
    Utils::log("SENSNdkGps", "init called");
    _permissionGranted = granted;
    if(granted)
    {
        JNIEnv* env;
        _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
        _vm->AttachCurrentThread(&env, NULL);

        jclass clazz = env->GetObjectClass(*_object);
        jmethodID methodId0 = env->GetMethodID(clazz,
                                                     "init",
                                                     "(Landroid/content/Context;)V");
        env->CallVoidMethod(*_object, methodId0, *_context);

        _initialized = true;
    }
}

bool SENSNdkGps::start()
{
    if(!_initialized)
        return false;

    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass clazz = env->GetObjectClass(*_object);
    jmethodID methodId = env->GetMethodID(clazz,
                                          "start",
                                          "()V");

    if(methodId == nullptr)
        return false;

    //jobject   handler  = env->NewObject(*_clazz, methodId);
    env->CallVoidMethod(*_object, methodId);

    return true;
}

void SENSNdkGps::stop()
{
    //stop locations manager
}

