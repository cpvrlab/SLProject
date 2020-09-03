#include "SENSNdkGps.h"
#include <jni.h>

SENSNdkGps::SENSNdkGps(JavaVM* vm, jobject* activityContext, jclass* clazz, jobject* object)
  : _vm(vm),
    _clazz(clazz),
    _object(object),
    _context(activityContext)
{
    /*
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
     */
}

bool SENSNdkGps::start()
{
    if(!_permissionGranted)
        return false;

    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jmethodID methodId0 = env->GetStaticMethodID(*_clazz,
                                                "setContext",
                                                "(Landroid/content/Context;)V");
    env->CallStaticVoidMethod(*_clazz, methodId0, *_context);

    jmethodID methodId = env->GetStaticMethodID(*_clazz,
                                          "start",
                                          "()V");

    if(methodId == nullptr)
        return false;

    //jobject   handler  = env->NewObject(*_clazz, methodId);
    env->CallStaticVoidMethod(*_clazz, methodId);

    return true;
    /*
    JNIEnv* env;
    _vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    _vm->AttachCurrentThread(&env, NULL);

    jclass c = env->FindClass("ch/cpvr/wai/SENSGps");
    //g_ctx.jniHelperClz = (*env)->NewGlobalRef(env, clz);

    if (c == nullptr)
        return false; //return JNI_ERR;

    jmethodID methodId = env->GetMethodID(c,
                                          "start",
                                          "()V");
    jobject   handler  = env->NewObject(c, methodId);
    env->CallVoidMethod(handler, methodId);

    _vm->DetachCurrentThread();
*/
    /*
    g_ctx.javaVM = vm;
    if ((*vm)->GetEnv(vm, (void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR; // JNI version not supported.
    }

    jclass  clz = (*env)->FindClass(env,
                                    "com/example/hellojnicallback/JniHandler");
    g_ctx.jniHelperClz = (*env)->NewGlobalRef(env, clz);

    jmethodID  jniHelperCtor = (*env)->GetMethodID(env, g_ctx.jniHelperClz,
                                                   "<init>", "()V");
    jobject    handler = (*env)->NewObject(env, g_ctx.jniHelperClz,
                                           jniHelperCtor);
    g_ctx.jniHelperObj = (*env)->NewGlobalRef(env, handler);
    queryRuntimeInfo(env, g_ctx.jniHelperObj);

    g_ctx.done = 0;
    g_ctx.mainActivityObj = NULL;
    return  JNI_VERSION_1_6;
    */

    /*
     *
    ANativeActivity* activity = _app->activity;
    JNIEnv*          env;
    activity->vm->GetEnv((void**)&env, JNI_VERSION_1_6);

    activity->vm->AttachCurrentThread(&env, NULL);

    jobject activityObj = env->NewGlobalRef(activity->clazz);
    jclass  clz         = env->GetObjectClass(activityObj);
    env->CallVoidMethod(activityObj,
                        env->GetMethodID(clz, "RequestCamera", "()V"));
    env->DeleteGlobalRef(activityObj);

    activity->vm->DetachCurrentThread();
     */
}

void SENSNdkGps::stop()
{
    //stop locations manager
}

void SENSNdkGps::setPermissionGranted()
{
    _permissionGranted = true;
}
