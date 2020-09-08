#include "SENSNdkOrientation.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>

static SENSNdkOrientation* gOrientationPtr = nullptr;
SENSNdkOrientation*        GetOrientationPtr()
{
	if(gOrientationPtr== nullptr)
		Utils::log("SENSNdkOrientation", "Global orientation pointer has not been initialized");
    return gOrientationPtr;
}

SENSNdkOrientation::SENSNdkOrientation(JavaVM* vm, jobject* activityContext, jclass* clazz)
  : _vm(vm)
{
    gOrientationPtr = this;

    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    //allocate object SENSOrientation java class
    jobject o = env->AllocObject(*clazz);
    _object   = env->NewGlobalRef(o);

    //set java activity context
    jmethodID methodId = env->GetMethodID(*clazz,
                                          "init",
                                          "(Landroid/content/Context;)V");
    env->CallVoidMethod(_object, methodId, *activityContext);

    _vm->DetachCurrentThread();
}

bool SENSNdkOrientation::start()
{
    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass    clazz    = env->GetObjectClass(_object);
    jmethodID methodId = env->GetMethodID(clazz,
                                          "start",
                                          "()V");
    env->CallVoidMethod(_object, methodId);

    _vm->DetachCurrentThread();

    _running = true;
    return true;
}

void SENSNdkOrientation::stop()
{
    if(!_running)
        return;
    _running = false;

    //stop locations manager
    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass    clazz    = env->GetObjectClass(_object);
    jmethodID methodId = env->GetMethodID(clazz,
                                          "stop",
                                          "()V");
    env->CallVoidMethod(_object, methodId);

    _vm->DetachCurrentThread();
}

void SENSNdkOrientation::updateOrientation(const SENSOrientation::Quat& orientation)
{
    Utils::log("SENSNdkOrientation", "updateLocation");
    setOrientation(orientation);
}

extern "C" JNIEXPORT void JNICALL
Java_ch_cpvr_wai_SENSOrientation_onOrientationQuat(JNIEnv* env, jclass obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW)
{
    Utils::log("SENSNdkOrientation", "onLocationLLA");
    GetOrientationPtr()->updateOrientation({quatX, quatY, quatZ, quatW});
}
