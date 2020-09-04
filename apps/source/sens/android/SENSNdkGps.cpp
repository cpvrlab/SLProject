#include "SENSNdkGps.h"
#include <jni.h>
#include <assert.h>
#include <Utils.h>

static SENSNdkGps* gGpsPtr = nullptr;
SENSNdkGps*        GetGpsPtr()
{
    Utils::log("SENSNdkGps", "Global gps pointer has not been initialized");
    return gGpsPtr;
}

SENSNdkGps::SENSNdkGps(JavaVM* vm, jobject* activityContext, jclass* clazz)
  : _vm(vm)
{
    gGpsPtr = this;

    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    //allocate object SENSGps java class
    jobject o = env->AllocObject(*clazz);
    _object   = env->NewGlobalRef(o);

    //set java activity context
    //jclass clazz = env->GetObjectClass(_object);
    jmethodID methodId = env->GetMethodID(*clazz,
                                          "init",
                                          "(Landroid/content/Context;)V");
    env->CallVoidMethod(_object, methodId, *activityContext);

    _vm->DetachCurrentThread();
}

void SENSNdkGps::init(bool granted)
{
    Utils::log("SENSNdkGps", "init called");
    _permissionGranted = granted;
}

bool SENSNdkGps::start()
{
    if (!_permissionGranted)
        return false;

    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass    clazz    = env->GetObjectClass(_object);
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

void SENSNdkGps::updateLocation(double latitudeDEG,
                                double longitudeDEG,
                                double altitudeM,
                                float  accuracyM)
{
    Utils::log("SENSGps", "updateLocation");
    setLocation(latitudeDEG, longitudeDEG, altitudeM, accuracyM);
}

extern "C" JNIEXPORT void JNICALL
Java_ch_cpvr_wai_SENSGps_onLocationLLA(JNIEnv* env,
                                       jclass  obj,
                                       jdouble latitudeDEG,
                                       jdouble longitudeDEG,
                                       jdouble altitudeM,
                                       jfloat  accuracyM)
{
    Utils::log("SENSGps", "onLocationLLA");
    GetGpsPtr()->updateLocation(latitudeDEG, longitudeDEG, altitudeM, accuracyM);
}
