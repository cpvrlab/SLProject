#include <jni.h>
#include <Utils.h>
#include "SENSAndroidPermissions.h"

SENSAndroidPermissions::SENSAndroidPermissions(JavaVM* jvm) : _jvm(jvm)
{
}

void SENSAndroidPermissions::askPermissions()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "askPermissions", "()V");

    env->CallStaticVoidMethod(clazz, methodid);
}

bool SENSAndroidPermissions::hasCameraPermission()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "hasCameraPermission", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSAndroidPermissions::hasGPSPermission()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "hasGPSPermission", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSAndroidPermissions::hasInternetPermission()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "hasInternetPermission", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSAndroidPermissions::hasStoragePermission()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "hasStoragePermission", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSAndroidPermissions::canShowCameraPermissionDialog()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "canShowCameraPermissionDialog", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSAndroidPermissions::canShowGPSPermissionDialog()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "canShowGPSPermissionDialog", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSAndroidPermissions::canShowInternetPermissionDialog()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "canShowInternetPermissionDialog", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSAndroidPermissions::canShowStoragePermissionDialog()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "canShowStoragePermissionDialog", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSAndroidPermissions::isLocationEnabled()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "isLocationEnabled", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

void SENSAndroidPermissions::askEnabledLocation()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass    clazz    = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "askEnabledLocation", "()V");

    return env->CallStaticVoidMethod(clazz, methodid);
}
