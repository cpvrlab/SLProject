#include <jni.h>
#include <Utils.h>
#include "SENSNdkPermissions.h"

SENSNdkPermissions::SENSNdkPermissions(JavaVM* jvm): 
   _jvm(jvm)
{
}

void SENSNdkPermissions::askPermissions()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "askPermissions", "()V");

    env->CallStaticVoidMethod(clazz, methodid);
}

bool SENSNdkPermissions::hasCameraPermission()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "hasCameraPermission", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSNdkPermissions::hasGPSPermission()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "hasGPSPermission", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSNdkPermissions::hasInternetPermission()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "hasInternetPermission", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSNdkPermissions::hasStoragePermission()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "hasStoragePermission", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSNdkPermissions::canShowCameraPermissionDialog()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "canShowCameraPermissionDialog", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSNdkPermissions::canShowGPSPermissionDialog()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "canShowGPSPermissionDialog", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSNdkPermissions::canShowInternetPermissionDialog()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "canShowInternetPermissionDialog", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSNdkPermissions::canShowStoragePermissionDialog()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "canShowStoragePermissionDialog", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

bool SENSNdkPermissions::isLocationEnabled()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "isLocationEnabled", "()Z");

    return env->CallStaticBooleanMethod(clazz, methodid);
}

void SENSNdkPermissions::askEnabledLocation()
{
    JNIEnv* env;
    _jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    jclass clazz = env->FindClass("ch/cpvr/wai/GLES3Lib");
    jmethodID methodid = env->GetStaticMethodID(clazz, "askEnabledLocation", "()V");

    return env->CallStaticVoidMethod(clazz, methodid);
}
