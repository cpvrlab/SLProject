/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

//BEGIN_INCLUDE(all)
#include <initializer_list>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <errno.h>
#include <cassert>

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <chrono>

#include <android/input.h>
#include <android/sensor.h>
#include <android/asset_manager.h>
#include <android/log.h>
#include <android_native_app_glue.h>

#include <AppDemoNativeSensorsInterface.h>

#include <Utils.h>

#include <WAIApp.h>

#include <string>

#include <android/SENSNdkCamera.h>

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

SENSNdkCamera* ndkCamera = nullptr;

struct Engine
{
    SensorsHandler* sensorsHandler;
    EGLDisplay      display;
    EGLSurface      surface;
    EGLContext      context;
    int32_t         width;
    int32_t         height;

    int run;

    WAIApp waiApp;

    // input stuff
    int32_t  pointersDown;
    uint64_t lastTouchMS;
};

void startNdkCamera(jboolean permission)
{
    if (permission != JNI_FALSE && !ndkCamera)
    {
        //get all information about available cameras
        ndkCamera = new SENSNdkCamera(SENSCamera::Facing::BACK);
        //start continious captureing request with certain configuration
        SENSCamera::Config camConfig;
        camConfig.targetWidth          = 640;
        camConfig.targetHeight         = 360;
        camConfig.focusMode            = SENSCamera::FocusMode::FIXED_INFINITY_FOCUS;
        camConfig.adjustAsynchronously = true;
        ndkCamera->start(camConfig);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ch_cpvr_nativewai_WAIActivity_notifyCameraPermission(
  JNIEnv*  env,
  jclass   type,
  jboolean permission)
{
    std::thread permissionHandler(&startNdkCamera, permission);
    permissionHandler.detach();
}

std::string getInternalDir(android_app* app)
{
    JavaVM* jvm            = app->activity->vm;
    JNIEnv* env            = nullptr;
    bool    threadAttached = false;

    switch (jvm->GetEnv((void**)&env, JNI_VERSION_1_6))
    {
        case JNI_OK: {
        }
        break;
        case JNI_EDETACHED: {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                LOGW("Could not attach thread to jvm\n");
                return "";
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION: {
            //TODO(dgj1): error handling
            LOGW("unsupported java version\n");
            return "";
        }
    }

    jobject objectActivity = app->activity->clazz;
    // Get File object for the external storage directory.
    jclass classContext = env->FindClass("android/app/Activity");
    if (!classContext)
    {
        //TODO(dgj1): error handling
        LOGW("could not get classContext\n");
        return "";
    }
    jmethodID methodIDgetFilesDir = env->GetMethodID(classContext, "getFilesDir", "()Ljava/io/File;");
    if (!methodIDgetFilesDir)
    {
        //TODO(dgj1): error handling
        LOGW("could not get methodIDgetExternalFilesDir\n");
        return "";
    }
    jobject    objectFile = env->CallObjectMethod(objectActivity, methodIDgetFilesDir);
    jthrowable exception  = env->ExceptionOccurred();
    if (exception)
    {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    // Call method on File object to retrieve String object.
    jclass classFile = env->GetObjectClass(objectFile);
    if (!classFile)
    {
        //TODO(dgj1): error handling
        LOGW("could not get classFile\n");
        return "";
    }
    jmethodID methodIDgetAbsolutePath = env->GetMethodID(classFile, "getAbsolutePath", "()Ljava/lang/String;");
    if (!methodIDgetAbsolutePath)
    {
        //TODO(dgj1): error handling
        LOGW("could not get methodIDgetAbsolutePath\n");
        return "";
    }
    jstring stringPath = (jstring)env->CallObjectMethod(objectFile, methodIDgetAbsolutePath);
    exception          = env->ExceptionOccurred();
    if (exception)
    {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jboolean    isCopy;
    const char* absPath = env->GetStringUTFChars(stringPath, &isCopy);
    std::string path    = std::string(absPath);

    env->ReleaseStringUTFChars(stringPath, absPath);

    if (threadAttached)
    {
        jvm->DetachCurrentThread();
    }

    return path;
}

void extractAPKFolder(android_app* app, std::string internalPath, std::string assetDirPath)
{
    JavaVM* jvm            = app->activity->vm;
    JNIEnv* env            = nullptr;
    bool    threadAttached = false;

    switch (jvm->GetEnv((void**)&env, JNI_VERSION_1_6))
    {
        case JNI_OK: {
        }
        break;
        case JNI_EDETACHED: {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                LOGW("Could not attach thread to jvm\n");
                return;
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION: {
            //TODO(dgj1): error handling
            LOGW("unsupported java version\n");
            return;
        }
    }

    std::string outputPath = Utils::unifySlashes(internalPath + "/" + assetDirPath + "/");
    if (!Utils::dirExists(outputPath))
    {
        Utils::makeDir(outputPath);
    }

    AAssetManager* mgr      = app->activity->assetManager;
    AAssetDir*     assetDir = AAssetManager_openDir(mgr, assetDirPath.c_str());
    const char*    filename = (const char*)NULL;
    while ((filename = AAssetDir_getNextFileName(assetDir)) != NULL)
    {
        std::string inputFilename = assetDirPath + "/" + std::string(filename);
        AAsset*     asset         = AAssetManager_open(mgr, inputFilename.c_str(), AASSET_MODE_STREAMING);
        int         nb_read       = 0;
        char        buf[BUFSIZ];
        std::string outputFilename = outputPath + std::string(filename);
        FILE*       out            = fopen(outputFilename.c_str(), "w");
        while ((nb_read = AAsset_read(asset, buf, BUFSIZ)) > 0)
        {
            fwrite(buf, nb_read, 1, out);
        }
        fclose(out);
        AAsset_close(asset);
    }
    AAssetDir_close(assetDir);

    if (threadAttached)
    {
        jvm->DetachCurrentThread();
    }
}

std::string getExternalDir(android_app* app)
{
    JavaVM* jvm            = app->activity->vm;
    JNIEnv* env            = nullptr;
    bool    threadAttached = false;

    switch (jvm->GetEnv((void**)&env, JNI_VERSION_1_6))
    {
        case JNI_OK: {
        }
        break;
        case JNI_EDETACHED: {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                LOGW("Could not attach thread to jvm\n");
                return "";
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION: {
            //TODO(dgj1): error handling
            LOGW("unsupported java version\n");
            return "";
        }
    }

    jobject objectActivity = app->activity->clazz;
    // Get File object for the external storage directory.
    jclass classContext = env->FindClass("android/app/Activity");
    if (!classContext)
    {
        //TODO(dgj1): error handling
        LOGW("could not get classContext\n");
        return "";
    }
    jmethodID methodIDgetExternalFilesDir = env->GetMethodID(classContext, "getExternalFilesDir", "(Ljava/lang/String;)Ljava/io/File;");
    if (!methodIDgetExternalFilesDir)
    {
        //TODO(dgj1): error handling
        LOGW("could not get methodIDgetExternalFilesDir\n");
        return "";
    }
    std::string s;
    jstring     jS         = env->NewStringUTF(s.c_str());
    jobject     objectFile = env->CallObjectMethod(objectActivity, methodIDgetExternalFilesDir, jS);
    jthrowable  exception  = env->ExceptionOccurred();
    if (exception)
    {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    // Call method on File object to retrieve String object.
    jclass classFile = env->GetObjectClass(objectFile);
    if (!classFile)
    {
        //TODO(dgj1): error handling
        LOGW("could not get classFile\n");
        return "";
    }
    jmethodID methodIDgetAbsolutePath = env->GetMethodID(classFile, "getAbsolutePath", "()Ljava/lang/String;");
    if (!methodIDgetAbsolutePath)
    {
        //TODO(dgj1): error handling
        LOGW("could not get methodIDgetAbsolutePath\n");
        return "";
    }
    jstring stringPath = (jstring)env->CallObjectMethod(objectFile, methodIDgetAbsolutePath);
    exception          = env->ExceptionOccurred();
    if (exception)
    {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jboolean    isCopy;
    const char* absPath = env->GetStringUTFChars(stringPath, &isCopy);
    std::string path    = std::string(absPath);

    env->ReleaseStringUTFChars(stringPath, absPath);

    if (threadAttached)
    {
        jvm->DetachCurrentThread();
    }

    return path;
}

jstring androidPermissionName(JNIEnv* env, const char* permissionName)
{
    jclass   classManifestPermission = env->FindClass("android/Manifest$permission");
    jfieldID idPermission            = env->GetStaticFieldID(classManifestPermission, permissionName, "Ljava/lang/String;");
    jstring  result                  = (jstring)(env->GetStaticObjectField(classManifestPermission, idPermission));

    return result;
}

bool isPermissionGranted(struct android_app* app, const char* permissionName)
{
    JavaVM* jvm            = app->activity->vm;
    JNIEnv* env            = nullptr;
    bool    threadAttached = false;

    switch (jvm->GetEnv((void**)&env, JNI_VERSION_1_6))
    {
        case JNI_OK: {
        }
        break;
        case JNI_EDETACHED: {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                LOGW("Could not attach thread to jvm\n");
                return "";
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION: {
            //TODO(dgj1): error handling
            LOGW("unsupported java version\n");
            return "";
        }
    }

    jstring stringPermission = androidPermissionName(env, permissionName);

    jclass   classPackageManager    = env->FindClass("android/content/pm/PackageManager");
    jfieldID idPermissionGranted    = env->GetStaticFieldID(classPackageManager, "PERMISSION_GRANTED", "I");
    jint     permissionGrantedValue = env->GetStaticIntField(classPackageManager, idPermissionGranted);

    jobject   activity                  = app->activity->clazz;
    jclass    classContext              = env->FindClass("android/content/Context");
    jmethodID methodCheckSelfPermission = env->GetMethodID(classContext, "checkSelfPermission", "(Ljava/lang/String;)I");
    jint      checkResult               = env->CallIntMethod(activity, methodCheckSelfPermission, stringPermission);

    bool result = (checkResult == permissionGrantedValue);

    if (threadAttached)
    {
        jvm->DetachCurrentThread();
    }

    return result;
}

void requestPermission(struct android_app* app)
{
    JavaVM* jvm            = app->activity->vm;
    JNIEnv* env            = nullptr;
    bool    threadAttached = false;

    switch (jvm->GetEnv((void**)&env, JNI_VERSION_1_6))
    {
        case JNI_OK: {
        }
        break;
        case JNI_EDETACHED: {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                LOGW("Could not attach thread to jvm\n");
                return;
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION: {
            //TODO(dgj1): error handling
            LOGW("unsupported java version\n");
            return;
        }
    }

    jobjectArray permissionArray = env->NewObjectArray(2, env->FindClass("java/lang/String"), env->NewStringUTF(""));
    env->SetObjectArrayElement(permissionArray, 0, androidPermissionName(env, "CAMERA"));
    env->SetObjectArrayElement(permissionArray, 0, androidPermissionName(env, "INTERNET"));

    jobject   activity                = app->activity->clazz;
    jclass    classContext            = env->FindClass("android/app/Activity");
    jmethodID methodRequestPermission = env->GetMethodID(classContext, "requestPermissions", "([Ljava/lang/String;I)V");

    env->CallVoidMethod(activity, methodRequestPermission, permissionArray, 0);

    if (threadAttached)
    {
        jvm->DetachCurrentThread();
    }
}

void checkAndRequestAndroidPermissions(struct android_app* app, Engine* engine)
{
    JNIEnv*          env;
    ANativeActivity* activity = app->activity;
    activity->vm->GetEnv((void**)&env, JNI_VERSION_1_6);

    activity->vm->AttachCurrentThread(&env, NULL);

    jobject activityObj = env->NewGlobalRef(activity->clazz);
    jclass  clz         = env->GetObjectClass(activityObj);
    env->CallVoidMethod(activityObj,
                        env->GetMethodID(clz, "RequestCamera", "()V"));
    env->DeleteGlobalRef(activityObj);

    activity->vm->DetachCurrentThread();

    /*bool hasPermission = isPermissionGranted(app, "CAMERA") && isPermissionGranted(app, "INTERNET");
    if (!hasPermission)
    {
        requestPermission(app);
    }*/
}

static void onInit(void* usrPtr, struct android_app* app)
{
    Engine* engine = (Engine*)usrPtr;
    checkAndRequestAndroidPermissions(app, engine);
    /*
     * Here specify the attributes of the desired configuration.
     * Below, we select an EGLConfig with at least 8 bits per color
     * component compatible with on-screen windows
     */
    const EGLint attribs[] = {EGL_BLUE_SIZE,
                              8,
                              EGL_GREEN_SIZE,
                              8,
                              EGL_RED_SIZE,
                              8,
                              EGL_DEPTH_SIZE,
                              16,
                              EGL_STENCIL_SIZE,
                              0,
                              EGL_NONE};

    EGLint     w, h, format;
    EGLint     numConfigs;
    EGLConfig  config;
    EGLSurface surface;
    EGLContext context;

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    eglInitialize(display, 0, 0);

    /* Here, the application chooses the configuration it desires.
     * find the best match if possible, otherwise use the very first one
     */
    eglChooseConfig(display, attribs, nullptr, 0, &numConfigs);
    std::unique_ptr<EGLConfig[]> supportedConfigs(new EGLConfig[numConfigs]);
    assert(supportedConfigs);
    eglChooseConfig(display, attribs, supportedConfigs.get(), numConfigs, &numConfigs);
    assert(numConfigs);
    int i;
    for (i = 0; i < numConfigs; i++)
    {
        auto&  cfg = supportedConfigs[i];
        EGLint r, g, b, d;
        if (eglGetConfigAttrib(display, cfg, EGL_RED_SIZE, &r) &&
            eglGetConfigAttrib(display, cfg, EGL_GREEN_SIZE, &g) &&
            eglGetConfigAttrib(display, cfg, EGL_BLUE_SIZE, &b) &&
            eglGetConfigAttrib(display, cfg, EGL_DEPTH_SIZE, &d) &&
            r == 8 && g == 8 && b == 8 && d == 0)
        {
            config = supportedConfigs[i];
            break;
        }
    }
    if (i == numConfigs)
    {
        config = supportedConfigs[0];
    }

    /* EGL_NATIVE_VISUAL_ID is an attribute of the EGLConfig that is
     * guaranteed to be accepted by ANativeWindow_setBuffersGeometry().
     * As soon as we picked a EGLConfig, we can safely reconfigure the
     * ANativeWindow buffers to match, using EGL_NATIVE_VISUAL_ID. */
    eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);
    surface = eglCreateWindowSurface(display, config, app->window, NULL);

    EGLint contextArgs[] = {
      EGL_CONTEXT_MAJOR_VERSION,
      3,
      EGL_CONTEXT_MINOR_VERSION,
      1,
      EGL_NONE};
    context = eglCreateContext(display, config, NULL, contextArgs);

    if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE)
    {
        LOGW("Unable to eglMakeCurrent");
        return;
    }

    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    engine->display = display;
    engine->context = context;
    engine->surface = surface;
    engine->width   = w;
    engine->height  = h;
    engine->run     = true;

    // Check openGL on the system
    auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
    for (auto name : opengl_info)
    {
        auto info = glGetString(name);
        LOGI("OpenGL Info: %s", info);
    }

    glViewport(0, 0, w, h);

    std::string path = getInternalDir(app);
    extractAPKFolder(app, path, "images");
    extractAPKFolder(app, path, "images/fonts");
    extractAPKFolder(app, path, "images/textures");
    extractAPKFolder(app, path, "videos");
    extractAPKFolder(app, path, "models");
    extractAPKFolder(app, path, "shaders");
    extractAPKFolder(app, path, "calibrations");
    extractAPKFolder(app, path, "config");

    AppDirectories dirs;
    dirs.slDataRoot  = path;
    dirs.waiDataRoot = path;
    dirs.writableDir = path + "/";

    CVImage::defaultPath = dirs.slDataRoot + "/images/textures/";

    AConfiguration* appConfig = AConfiguration_new();
    AConfiguration_fromAssetManager(appConfig, app->activity->assetManager);
    int32_t dpi = AConfiguration_getDensity(appConfig);
    AConfiguration_delete(appConfig);
    engine->waiApp.load(640, 360, w, h, 1.0, 1.0, dpi, dirs);
}

static void onClose(void* usrPtr, struct android_app* app)
{
    Engine* engine = (Engine*)usrPtr;

    if (engine->display != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (engine->context != EGL_NO_CONTEXT)
        {
            eglDestroyContext(engine->display, engine->context);
        }
        if (engine->surface != EGL_NO_SURFACE)
        {
            eglDestroySurface(engine->display, engine->surface);
        }
        eglTerminate(engine->display);
    }

    engine->display = EGL_NO_DISPLAY;
    engine->context = EGL_NO_CONTEXT;
    engine->surface = EGL_NO_SURFACE;
    engine->run     = false;
}

static void onSaveState(void* usrPtr)
{
}

static void onGainedFocus(void* usrPtr)
{
    Engine* engine = (Engine*)usrPtr;
    sensorsHandler_enableAccelerometer(engine->sensorsHandler);
}

static void onLostFocus(void* usrPtr)
{
    Engine* engine = (Engine*)usrPtr;
    sensorsHandler_disableAccelerometer(engine->sensorsHandler);
}

static void onAcceleration(void* usrPtr, float x, float y, float z)
{
    //LOGI("accel = %f %f %f\n", x, y, z);
}

static uint64_t millisecondsSinceEpoch()
{
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
    uint64_t result = ms.count();

    return result;
}

static void handleTouchDown(Engine* engine, AInputEvent* event)
{
    int     sceneViewIndex = 0; //TODO(dgj1): get this from SLProject
    int32_t x0             = AMotionEvent_getX(event, 0);
    int32_t y0             = AMotionEvent_getY(event, 0);
    int32_t touchCount     = AMotionEvent_getPointerCount(event);

    // just got a new single touch
    if (touchCount == 1)
    {
        // get time to detect double taps
        uint64_t touchNowMS   = millisecondsSinceEpoch();
        uint64_t touchDeltaMS = touchNowMS - engine->lastTouchMS;
        engine->lastTouchMS   = touchNowMS;

        if (touchDeltaMS < 250)
        {
            LOGI("double click");
            engine->waiApp.doubleClick(sceneViewIndex, MB_left, x0, y0, K_none);
        }
        else
        {
            LOGI("mouse down");
            engine->waiApp.mouseDown(sceneViewIndex, MB_left, x0, y0, K_none);
        }
    }

    // it's two fingers but one delayed (already executed mouse down)
    else if (touchCount == 2 && engine->pointersDown == 1)
    {
        LOGI("mouse up + touch 2 down");
        int x1 = AMotionEvent_getX(event, 1);
        int y1 = AMotionEvent_getY(event, 1);
        engine->waiApp.mouseUp(sceneViewIndex, MB_left, x0, y0, K_none);
        engine->waiApp.touch2Down(sceneViewIndex, x0, y0, x1, y1);
    }

    // it's two fingers at the same time
    else if (touchCount == 2)
    {
        // get time to detect double taps
        uint64_t touchNowMS   = millisecondsSinceEpoch();
        uint64_t touchDeltaMS = touchNowMS - engine->lastTouchMS;
        engine->lastTouchMS   = touchNowMS;

        int x1 = AMotionEvent_getX(event, 1);
        int y1 = AMotionEvent_getY(event, 1);

        LOGI("touch 2 down");
        engine->waiApp.touch2Down(sceneViewIndex, x0, y0, x1, y1);
    }

    engine->pointersDown = touchCount;
}

static void handleTouchUp(Engine* engine, AInputEvent* event)
{
    int     sceneViewIndex = 0; //TODO(dgj1): get this from SLProject
    int32_t x0             = AMotionEvent_getX(event, 0);
    int32_t y0             = AMotionEvent_getY(event, 0);
    int32_t touchCount     = AMotionEvent_getPointerCount(event);

    if (touchCount == 1)
    {
        LOGI("mouse up");
        engine->waiApp.mouseUp(sceneViewIndex, MB_left, x0, y0, K_none);
    }
    else if (touchCount == 2)
    {
        int32_t x1 = AMotionEvent_getX(event, 1);
        int32_t y1 = AMotionEvent_getY(event, 1);

        LOGI("touch 2 up");
        engine->waiApp.touch2Up(sceneViewIndex, x0, y0, x1, y1);
    }

    engine->pointersDown = touchCount;
}

static void handleTouchMove(Engine* engine, AInputEvent* event)
{
    int     sceneViewIndex = 0; //TODO(dgj1): get this from SLProject
    int32_t x0             = AMotionEvent_getX(event, 0);
    int32_t y0             = AMotionEvent_getY(event, 0);
    int32_t touchCount     = AMotionEvent_getPointerCount(event);

    if (touchCount == 1)
    {
        LOGI("mouse move");
        engine->waiApp.mouseMove(sceneViewIndex, x0, y0);
    }
    else if (touchCount == 2)
    {
        int32_t x1 = AMotionEvent_getX(event, 1);
        int32_t y1 = AMotionEvent_getY(event, 1);

        LOGI("touch 2 move");
        engine->waiApp.touch2Move(sceneViewIndex, x0, y0, x1, y1);
    }
}

static int32_t handleInput(struct android_app* app, AInputEvent* event)
{
    Engine* engine = (Engine*)app->userData;
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION)
    {
        int action     = AMotionEvent_getAction(event);
        int actionCode = action & AMOTION_EVENT_ACTION_MASK;

        switch (actionCode)
        {
            case AMOTION_EVENT_ACTION_DOWN:
            case AMOTION_EVENT_ACTION_POINTER_DOWN: {
                handleTouchDown(engine, event);
            }
            break;

            case AMOTION_EVENT_ACTION_UP:
            case AMOTION_EVENT_ACTION_POINTER_UP: {
                handleTouchUp(engine, event);
            }
            break;

            case AMOTION_EVENT_ACTION_MOVE: {
                handleTouchMove(engine, event);
            }
        }

        return 1;
    }
    return 0;
}

static void handleLifecycleEvent(struct android_app* app, int32_t cmd)
{
    Engine* engine = (Engine*)app->userData;
    switch (cmd)
    {
        case APP_CMD_SAVE_STATE:
            onSaveState(engine);
            break;
        case APP_CMD_INIT_WINDOW:
            if (app->window != NULL)
            {
                onInit(engine, app);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            onClose(engine, app);
            break;
        case APP_CMD_GAINED_FOCUS:
            onGainedFocus(engine);
            break;
        case APP_CMD_LOST_FOCUS:
            onLostFocus(engine);
            break;
        case APP_CMD_CONFIG_CHANGED:
            checkAndRequestAndroidPermissions(app, engine);
            break;
    }
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app* app)
{
    try
    {
        Engine engine = {};

        SensorsCallbacks callbacks;
        callbacks.onAcceleration = onAcceleration;
        callbacks.usrPtr         = &engine;

        app->onAppCmd     = handleLifecycleEvent;
        app->onInputEvent = handleInput;
        app->userData     = &engine;

        //checkAndRequestAndroidPermissions(app);

        initSensorsHandler(app, &callbacks, &engine.sensorsHandler);

        engine.run = true;
        while (engine.run)
        {
            int                         ident;
            int                         events;
            struct android_poll_source* source;

            while ((ident = ALooper_pollAll(0, NULL, &events, (void**)&source)) >= 0)
            {
                if (source != NULL)
                {
                    source->process(app, source);
                }

                if (ident == LOOPER_ID_USER)
                {
                    sensorsHandler_processEvent(engine.sensorsHandler);
                }

                // Check if we are exiting.
                if (app->destroyRequested != 0)
                {
                    onClose(&engine, app);
                    return;
                }
            }

            if (engine.display != nullptr && ndkCamera != nullptr)
            {
                SENSFramePtr sensFrame = ndkCamera->getLatestFrame();
                if (sensFrame)
                    engine.waiApp.updateVideoImage(sensFrame->imgRGB);
                else
                    engine.waiApp.updateVideoImage(cv::Mat());

                eglSwapBuffers(engine.display, engine.surface);
            }

            std::this_thread::sleep_for(10ms);
        }

        engine.waiApp.close();
    }
    catch (std::exception& e)
    {
        //todo: what do we do then?
        LOGI(e.what());
    }
}
