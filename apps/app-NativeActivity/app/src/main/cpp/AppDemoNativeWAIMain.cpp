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
#include <chrono>
#include <string>

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <android/input.h>
#include <android/sensor.h>
#include <android/asset_manager.h>
#include <android/log.h>

#include <android_native_app_glue.h>
#include <AppDemoNativeSensorsInterface.h>
#include <Utils.h>
#include <WAIApp.h>
#include <android/SENSNdkCamera.h>
#include <CV/CVImage.h>

#define ENGINE_DEBUG(...) Utils::log("Engine", __VA_ARGS__)
#define ENGINE_INFO(...) Utils::log("Engine", __VA_ARGS__)
#define ENGINE_WARN(...) Utils::log("Engine", __VA_ARGS__)
#define ASSERT(cond, fmt, ...) \
    if (!(cond)) \
    { \
        __android_log_assert(#cond, "Engine", fmt, ##__VA_ARGS__); \
    }

//#define ENGINE_DEBUG(...) // nothing
//#define ENGINE_INFO(...)  // nothing
//#define ENGINE_WARN(...)  // nothing

//bool         _cameraGranted = false;
//struct android_app* androidApp = nullptr;

class Engine
{
public:
    explicit Engine(android_app* app);

    void update();

    void onInit();
    void onTerminate();
    void onDestroy();
    void onBackButtonDown();

    static uint64_t millisecondsSinceEpoch();
    void            handleTouchMove(AInputEvent* event);
    void            handleTouchUp(AInputEvent* event);
    void            handleTouchDown(AInputEvent* event);

    bool isReady();

    void onPermissionGranted(jboolean granted);

    bool closeAppRequested() const;
    void closeAppRequested(bool state);
    //this callback can be called by the wrapped app to make native activity shutdown
    static void closeAppCallback();

private:
    void initDisplay();
    bool resumeDisplay();
    void terminateDisplay();

    void startCamera();
    void stopCamera();
    void checkAndRequestAndroidPermissions();

    std::string getInternalDir();
    std::string getExternalDir();
    void        extractAPKFolder(std::string internalPath, std::string assetDirPath);

    android_app* _app;
    //instantiated in fist call to onInit()
    WAIApp _waiApp;
    bool   _waiAppIsInitialized = false;

    AppDirectories _dirs;
    int32_t        _dpi;

    EGLConfig  _config;
    EGLDisplay _display;
    EGLSurface _surface;
    EGLContext _context;
    int32_t    _width;
    int32_t    _height;

    bool _hasFocus = false;
    //if this is set true by wai app the native activity has to initiate closing of the activity
    bool _closeAppRequest = false;

    // input stuff
    int32_t  _pointersDown;
    uint64_t _lastTouchMS;

    SENSNdkCamera* ndkCamera      = nullptr;
    bool           _cameraGranted = false;

    /*
    SensorsHandler* sensorsHandler;
    */
};

static Engine* pEngineObj = nullptr;
Engine*        GetEngine(void)
{
    ASSERT(pEngineObj, "Engine has not been initialized");
    return pEngineObj;
}

static void handleLifecycleEvent(struct android_app* app, int32_t cmd)
{
    ENGINE_DEBUG("handleLifecycleEvent: called");
    Engine* engine = reinterpret_cast<Engine*>(app->userData);
    switch (cmd)
    {
        case APP_CMD_INPUT_CHANGED:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_INPUT_CHANGED");
            break;
        case APP_CMD_INIT_WINDOW:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_INIT_WINDOW");
            engine->onInit();
            break;
        case APP_CMD_TERM_WINDOW:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_TERM_WINDOW");
            engine->onTerminate();
            //engine->onDestroy();
            break;
        case APP_CMD_WINDOW_RESIZED:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_WINDOW_RESIZED");
            break;
        case APP_CMD_WINDOW_REDRAW_NEEDED:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_WINDOW_REDRAW_NEEDED");
            break;
        case APP_CMD_CONTENT_RECT_CHANGED:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_CONTENT_RECT_CHANGED");
            break;
        case APP_CMD_GAINED_FOCUS:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_GAINED_FOCUS");
            break;
        case APP_CMD_LOST_FOCUS:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_LOST_FOCUS");
            break;
        case APP_CMD_CONFIG_CHANGED:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_CONFIG_CHANGED");
            break;
        case APP_CMD_LOW_MEMORY:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_LOW_MEMORY");
            break;
        case APP_CMD_START:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_START");
            break;
        case APP_CMD_RESUME:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_RESUME");
            break;
        case APP_CMD_SAVE_STATE:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_SAVE_STATE");
            break;
        case APP_CMD_PAUSE:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_PAUSE");
            break;
        case APP_CMD_STOP:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_STOP");
            break;
        case APP_CMD_DESTROY:
            ENGINE_DEBUG("handleLifecycleEvent: APP_CMD_DESTROY");
            engine->onDestroy();
            break;
    }
}

static int32_t handleInput(struct android_app* app, AInputEvent* event)
{
    Engine* engine = reinterpret_cast<Engine*>(app->userData);
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION)
    {
        int action     = AMotionEvent_getAction(event);
        int actionCode = action & AMOTION_EVENT_ACTION_MASK;

        switch (actionCode)
        {
            case AMOTION_EVENT_ACTION_DOWN:
            case AMOTION_EVENT_ACTION_POINTER_DOWN:
            {
                engine->handleTouchDown(event);
            }
            break;

            case AMOTION_EVENT_ACTION_UP:
            case AMOTION_EVENT_ACTION_POINTER_UP:
            {
                engine->handleTouchUp(event);
            }
            break;

            case AMOTION_EVENT_ACTION_MOVE:
            {
                engine->handleTouchMove(event);
            }
        }

        return 1;
    }
    else if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY)
    {
        if (AKeyEvent_getKeyCode(event) == AKEYCODE_BACK && AKeyEvent_getAction(event) == AKEY_EVENT_ACTION_DOWN)
        {
            // actions on back key
            engine->onBackButtonDown();
            return 1; // <-- prevent default handler
        }
    }

    return 0;
}

Engine::Engine(android_app* app)
  : _app(app)
{
}

void Engine::update()
{
    if (_display)
    {
        //ENGINE_DEBUG("eglSwapBuffers");
        _waiApp.update();
        eglSwapBuffers(_display, _surface);
    }
}

void Engine::onInit()
{
    ENGINE_DEBUG("onInit");

    startCamera();

    if (!_waiAppIsInitialized)
    {
        initDisplay();

        std::string path = getInternalDir();
        extractAPKFolder(path, "images");
        extractAPKFolder(path, "images/fonts");
        extractAPKFolder(path, "images/textures");
        extractAPKFolder(path, "videos");
        extractAPKFolder(path, "models");
        extractAPKFolder(path, "shaders");
        extractAPKFolder(path, "calibrations");
        extractAPKFolder(path, "config");
        extractAPKFolder(path, "voc");

        std::string externalPath = getExternalDir();

        _dirs.slDataRoot    = path;
        _dirs.waiDataRoot   = path + "/";
        _dirs.writableDir   = externalPath + "/";
        _dirs.vocabularyDir = path + "/voc/";
        _dirs.logFileDir    = externalPath + "/log/";

        AConfiguration* appConfig = AConfiguration_new();
        AConfiguration_fromAssetManager(appConfig, _app->activity->assetManager);
        _dpi = AConfiguration_getDensity(appConfig);
        AConfiguration_delete(appConfig);

        _waiApp.initCloseAppCallback(std::bind(&Engine::closeAppCallback));
        _waiApp.init(_width, _height, 1.0, 1.0, _dpi, _dirs);
        _waiAppIsInitialized = true;
    }
    else
    {
        if (!resumeDisplay())
        {
            _waiApp.close();
            terminateDisplay();
            initDisplay();
            _waiApp.initCloseAppCallback(std::bind(&Engine::closeAppCallback));
            _waiApp.init(_width, _height, 1.0, 1.0, _dpi, _dirs);
        }
    }

    _hasFocus = true;
}

void Engine::onTerminate()
{
    ENGINE_DEBUG("onTerminate");
    //terminateDisplay();
    //_waiApp.hide();
    stopCamera();
    _hasFocus = false;
}

void Engine::onDestroy()
{
    ENGINE_DEBUG("onDestroy");
    _waiApp.close();
    terminateDisplay();
}

void Engine::onBackButtonDown()
{
    ENGINE_DEBUG("onBackButtonDown");
    _waiApp.goBack();
}

void Engine::initDisplay()
{
    assert(_app->window);
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

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(display, 0, 0);

    /* Here, the application chooses the configuration it desires.
     * find the best match if possible, otherwise use the very first one
     */
    EGLint numConfigs;
    eglChooseConfig(display, attribs, nullptr, 0, &numConfigs);
    std::unique_ptr<EGLConfig[]> supportedConfigs(new EGLConfig[numConfigs]);
    assert(supportedConfigs);
    eglChooseConfig(display, attribs, supportedConfigs.get(), numConfigs, &numConfigs);
    assert(numConfigs);
    int       i;
    EGLConfig config;
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
    EGLint format;
    eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);
    ANativeWindow* window  = _app->window;
    EGLSurface     surface = eglCreateWindowSurface(display, config, window, NULL);

    EGLint contextArgs[] = {
      EGL_CONTEXT_MAJOR_VERSION,
      3,
      EGL_CONTEXT_MINOR_VERSION,
      1,
      EGL_NONE};
    EGLContext context = eglCreateContext(display, config, NULL, contextArgs);

    if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE)
    {
        Utils::log("WAInative", "onInit Unable to eglMakeCurrent");
        return;
    }

    EGLint w, h;
    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    _config  = config;
    _display = display;
    _context = context;
    _surface = surface;
    _width   = w;
    _height  = h;

    // Check openGL on the system
    auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
    for (auto name : opengl_info)
    {
        auto info = glGetString(name);
        ENGINE_DEBUG("WAInative", "OpenGL Info: %s", info);
    }
}

bool Engine::resumeDisplay()
{
    ANativeWindow* window = _app->window;
    _surface              = eglCreateWindowSurface(_display, _config, window, NULL);

    if (eglMakeCurrent(_display, _surface, _surface, _context) == EGL_FALSE)
    {
        Utils::log("WAInative", "onInit Unable to eglMakeCurrent");
        return false;
    }
    return true;
}

void Engine::terminateDisplay()
{
    if (_display != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (_context != EGL_NO_CONTEXT)
        {
            eglDestroyContext(_display, _context);
        }
        if (_surface != EGL_NO_SURFACE)
        {
            eglDestroySurface(_display, _surface);
        }
        eglTerminate(_display);
    }

    _display = EGL_NO_DISPLAY;
    _context = EGL_NO_CONTEXT;
    _surface = EGL_NO_SURFACE;
}

void Engine::startCamera()
{
    if (_cameraGranted)
    {
        try
        {
            if (ndkCamera)
                delete ndkCamera;

            //get all information about available cameras
            ndkCamera = new SENSNdkCamera(SENSCamera::Facing::BACK);
            //start continious captureing request with certain configuration
            SENSCamera::Config camConfig;
            camConfig.targetWidth          = 640;
            camConfig.targetHeight         = 360;
            camConfig.focusMode            = SENSCamera::FocusMode::FIXED_INFINITY_FOCUS;
            camConfig.convertToGray        = true;
            camConfig.adjustAsynchronously = true;
            ndkCamera->start(camConfig);

            _waiApp.initCamera(ndkCamera);
        }
        catch (std::exception& e)
        {
            Utils::log("SENSNdkCamera", e.what());
        }
    }
    else
    {
        checkAndRequestAndroidPermissions();
    }
}

void Engine::stopCamera()
{
    try
    {
        if (ndkCamera)
        {
            delete ndkCamera;
            ndkCamera = nullptr;
        }
    }
    catch (std::exception& e)
    {
        Utils::log("SENSNdkCamera", e.what());
    }
}

void Engine::checkAndRequestAndroidPermissions()
{
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
}

void Engine::onPermissionGranted(jboolean granted)
{
    _cameraGranted = (granted != JNI_FALSE);

    if (_cameraGranted)
    {
        startCamera();
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ch_cpvr_nativewai_WAIActivity_notifyCameraPermission(
  JNIEnv*  env,
  jclass   type,
  jboolean permission)
{
    std::thread permissionHandler(&Engine::onPermissionGranted, GetEngine(), permission);
    permissionHandler.detach();
}

std::string Engine::getInternalDir()
{
    JavaVM* jvm            = _app->activity->vm;
    JNIEnv* env            = nullptr;
    bool    threadAttached = false;

    switch (jvm->GetEnv((void**)&env, JNI_VERSION_1_6))
    {
        case JNI_OK:
        {
        }
        break;
        case JNI_EDETACHED:
        {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                Utils::log("WAInative", "Could not attach thread to jvm");
                return "";
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION:
        {
            //TODO(dgj1): error handling
            Utils::log("WAInative", "unsupported java version");
            Utils::log("WAInative", "unsupported java version");
            return "";
        }
    }

    jobject objectActivity = _app->activity->clazz;
    // Get File object for the external storage directory.
    jclass classContext = env->FindClass("android/app/Activity");
    if (!classContext)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get classContext\n");
        return "";
    }
    jmethodID methodIDgetFilesDir = env->GetMethodID(classContext, "getFilesDir", "()Ljava/io/File;");
    if (!methodIDgetFilesDir)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get methodIDgetExternalFilesDir\n");
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
        Utils::log("WAInative", "could not get classFile\n");
        return "";
    }
    jmethodID methodIDgetAbsolutePath = env->GetMethodID(classFile, "getAbsolutePath", "()Ljava/lang/String;");
    if (!methodIDgetAbsolutePath)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get methodIDgetAbsolutePath\n");
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

std::string Engine::getExternalDir()
{
    JavaVM* jvm            = _app->activity->vm;
    JNIEnv* env            = nullptr;
    bool    threadAttached = false;

    switch (jvm->GetEnv((void**)&env, JNI_VERSION_1_6))
    {
        case JNI_OK:
        {
        }
        break;
        case JNI_EDETACHED:
        {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                Utils::log("WAInative", "Could not attach thread to jvm\n");
                return "";
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION:
        {
            //TODO(dgj1): error handling
            Utils::log("WAInative", "unsupported java version\n");
            return "";
        }
    }

    jobject objectActivity = _app->activity->clazz;
    // Get File object for the external storage directory.
    jclass classContext = env->FindClass("android/app/Activity");
    if (!classContext)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get classContext\n");
        return "";
    }
    jmethodID methodIDgetExternalFilesDir = env->GetMethodID(classContext, "getExternalFilesDir", "(Ljava/lang/String;)Ljava/io/File;");
    if (!methodIDgetExternalFilesDir)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get methodIDgetExternalFilesDir\n");
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
        Utils::log("WAInative", "could not get classFile\n");
        return "";
    }
    jmethodID methodIDgetAbsolutePath = env->GetMethodID(classFile, "getAbsolutePath", "()Ljava/lang/String;");
    if (!methodIDgetAbsolutePath)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get methodIDgetAbsolutePath\n");
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

void Engine::extractAPKFolder(std::string internalPath, std::string assetDirPath)
{
    JavaVM* jvm            = _app->activity->vm;
    JNIEnv* env            = nullptr;
    bool    threadAttached = false;

    switch (jvm->GetEnv((void**)&env, JNI_VERSION_1_6))
    {
        case JNI_OK:
        {
        }
        break;
        case JNI_EDETACHED:
        {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                Utils::log("WAI", "Could not attach thread to jvm\n");
                return;
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION:
        {
            //TODO(dgj1): error handling
            Utils::log("WAInative", "unsupported java version\n");
            return;
        }
    }

    std::string outputPath = Utils::unifySlashes(internalPath + "/" + assetDirPath + "/");
    if (Utils::dirExists(outputPath))
    {
        //stop here, we assume everything is installed (uninstall the app if you added assets)
        return;
    }

    Utils::makeDir(outputPath);

    AAssetManager* mgr      = _app->activity->assetManager;
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

uint64_t Engine::millisecondsSinceEpoch()
{
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
    uint64_t result = ms.count();

    return result;
}

void Engine::handleTouchDown(AInputEvent* event)
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
        uint64_t touchDeltaMS = touchNowMS - _lastTouchMS;
        _lastTouchMS          = touchNowMS;

        if (touchDeltaMS < 250)
        {
            //Utils::log("WAInative","double click");
            _waiApp.doubleClick(sceneViewIndex, MB_left, x0, y0, K_none);
        }
        else
        {
            //Utils::log("WAInative","mouse down");
            _waiApp.mouseDown(sceneViewIndex, MB_left, x0, y0, K_none);
        }
    }

    // it's two fingers but one delayed (already executed mouse down)
    else if (touchCount == 2 && _pointersDown == 1)
    {
        //Utils::log("WAInative","mouse up + touch 2 down");
        int x1 = AMotionEvent_getX(event, 1);
        int y1 = AMotionEvent_getY(event, 1);
        _waiApp.mouseUp(sceneViewIndex, MB_left, x0, y0, K_none);
        _waiApp.touch2Down(sceneViewIndex, x0, y0, x1, y1);
    }

    // it's two fingers at the same time
    else if (touchCount == 2)
    {
        // get time to detect double taps
        uint64_t touchNowMS   = millisecondsSinceEpoch();
        uint64_t touchDeltaMS = touchNowMS - _lastTouchMS;
        _lastTouchMS          = touchNowMS;

        int x1 = AMotionEvent_getX(event, 1);
        int y1 = AMotionEvent_getY(event, 1);

        //Utils::log("WAInative","touch 2 down");
        _waiApp.touch2Down(sceneViewIndex, x0, y0, x1, y1);
    }

    _pointersDown = touchCount;
}

bool Engine::isReady()
{
    return _hasFocus;
}

void Engine::handleTouchUp(AInputEvent* event)
{
    int     sceneViewIndex = 0; //TODO(dgj1): get this from SLProject
    int32_t x0             = AMotionEvent_getX(event, 0);
    int32_t y0             = AMotionEvent_getY(event, 0);
    int32_t touchCount     = AMotionEvent_getPointerCount(event);

    if (touchCount == 1)
    {
        //Utils::log("WAInative","mouse up");
        _waiApp.mouseUp(sceneViewIndex, MB_left, x0, y0, K_none);
    }
    else if (touchCount == 2)
    {
        int32_t x1 = AMotionEvent_getX(event, 1);
        int32_t y1 = AMotionEvent_getY(event, 1);

        //Utils::log("WAInative","touch 2 up");
        _waiApp.touch2Up(sceneViewIndex, x0, y0, x1, y1);
    }

    _pointersDown = touchCount;
}

void Engine::handleTouchMove(AInputEvent* event)
{
    int     sceneViewIndex = 0; //TODO(dgj1): get this from SLProject
    int32_t x0             = AMotionEvent_getX(event, 0);
    int32_t y0             = AMotionEvent_getY(event, 0);
    int32_t touchCount     = AMotionEvent_getPointerCount(event);

    if (touchCount == 1)
    {
        //Utils::log("WAInative","mouse move");
        _waiApp.mouseMove(sceneViewIndex, x0, y0);
    }
    else if (touchCount == 2)
    {
        int32_t x1 = AMotionEvent_getX(event, 1);
        int32_t y1 = AMotionEvent_getY(event, 1);

        //Utils::log("WAInative","touch 2 move");
        _waiApp.touch2Move(sceneViewIndex, x0, y0, x1, y1);
    }
}

bool Engine::closeAppRequested() const
{
    return _closeAppRequest;
}

void Engine::closeAppRequested(bool state)
{
    _closeAppRequest = state;
}

void Engine::closeAppCallback()
{
    Utils::log("Engine", "closeAppCallback");
    GetEngine()->closeAppRequested(true);
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app* app)
{
    Engine engine(app);
    pEngineObj = &engine;

    app->userData     = reinterpret_cast<void*>(&engine);
    app->onAppCmd     = handleLifecycleEvent;
    app->onInputEvent = handleInput;
    app->userData     = &engine;

    try
    {
        while (true)
        {
            // The identifier of source (May be LOOPER_ID_MAIN, LOOPER_ID_INPUT or LOOPER_ID_USER).
            int                         ident;
            int                         events;
            struct android_poll_source* source;

            // We loop until all events are read
            while ((ident = ALooper_pollAll(engine.isReady() ? 0 : -1, NULL, &events, (void**)&source)) >= 0)
            {
                if (source != NULL)
                {
                    source->process(app, source);
                }

                // Check if we are exiting.
                if (app->destroyRequested != 0)
                {
                    Utils::log("android_main", "destroyRequested");
                    //todo: is there a reason to destroy here instead of in handleLifecycleEvent?
                    return;
                }
            }

            //if this is set true by wai app the native activity has to initiate closing of the activity
            if (engine.closeAppRequested())
            {
                Utils::log("android_main", "closeActivity");
                engine.closeAppRequested(false);
                ANativeActivity_finish(app->activity);
            }

            if (engine.isReady())
            {
                engine.update();
            }
        }
    }
    catch (std::exception& e)
    {
        Utils::log("android_main", e.what());
    }
}

/*

     void checkAndRequestAndroidPermissions(struct android_app* app)
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
    }

void startCamera()
{
    if (!androidApp)
        return;

    if (cameraGranted)
    {
        try
        {
            if (ndkCamera)
                delete ndkCamera;

            //get all information about available cameras
            ndkCamera = new SENSNdkCamera(SENSCamera::Facing::BACK);
            //start continious captureing request with certain configuration
            SENSCamera::Config camConfig;
            camConfig.targetWidth          = 640;
            camConfig.targetHeight         = 360;
            camConfig.focusMode            = SENSCamera::FocusMode::FIXED_INFINITY_FOCUS;
            camConfig.convertToGray        = true;
            camConfig.adjustAsynchronously = true;
            ndkCamera->start(camConfig);

            Engine* engine = (Engine*)androidApp->userData;
            engine->waiApp.setCamera(ndkCamera);
        }
        catch (std::exception& e)
        {
            Utils::log("SENSNdkCamera", e.what());
        }
    }
    else
    {
        checkAndRequestAndroidPermissions(androidApp);
    }
}

void stopCamera()
{
    try
    {
        if (ndkCamera)
        {
            delete ndkCamera;
            ndkCamera = nullptr;
        }
    }
    catch (std::exception& e)
    {
        Utils::log("SENSNdkCamera", e.what());
    }
}

void onPermissionGranted(jboolean granted)
{
    cameraGranted = (granted != JNI_FALSE);

    if (cameraGranted)
    {
        startCamera();
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ch_cpvr_nativewai_WAIActivity_notifyCameraPermission(
  JNIEnv*  env,
  jclass   type,
  jboolean permission)
{
    std::thread permissionHandler(&onPermissionGranted, permission);
    permissionHandler.detach();
}

std::string getInternalDir(android_app* app)
{
    JavaVM* jvm            = app->activity->vm;
    JNIEnv* env            = nullptr;
    bool    threadAttached = false;

    switch (jvm->GetEnv((void**)&env, JNI_VERSION_1_6))
    {
        case JNI_OK:
        {
        }
        break;
        case JNI_EDETACHED:
        {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                Utils::log("WAInative", "Could not attach thread to jvm");
                return "";
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION:
        {
            //TODO(dgj1): error handling
            Utils::log("WAInative", "unsupported java version");
            Utils::log("WAInative", "unsupported java version");
            return "";
        }
    }

    jobject objectActivity = app->activity->clazz;
    // Get File object for the external storage directory.
    jclass classContext = env->FindClass("android/app/Activity");
    if (!classContext)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get classContext\n");
        return "";
    }
    jmethodID methodIDgetFilesDir = env->GetMethodID(classContext, "getFilesDir", "()Ljava/io/File;");
    if (!methodIDgetFilesDir)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get methodIDgetExternalFilesDir\n");
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
        Utils::log("WAInative", "could not get classFile\n");
        return "";
    }
    jmethodID methodIDgetAbsolutePath = env->GetMethodID(classFile, "getAbsolutePath", "()Ljava/lang/String;");
    if (!methodIDgetAbsolutePath)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get methodIDgetAbsolutePath\n");
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
        case JNI_OK:
        {
        }
        break;
        case JNI_EDETACHED:
        {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                Utils::log("WAI", "Could not attach thread to jvm\n");
                return;
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION:
        {
            //TODO(dgj1): error handling
            Utils::log("WAInative", "unsupported java version\n");
            return;
        }
    }

    std::string outputPath = Utils::unifySlashes(internalPath + "/" + assetDirPath + "/");
    if (Utils::dirExists(outputPath))
    {
        //stop here, we assume everything is installed (uninstall the app if you added assets)
        return;
    }

    Utils::makeDir(outputPath);

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
        case JNI_OK:
        {
        }
        break;
        case JNI_EDETACHED:
        {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                Utils::log("WAInative", "Could not attach thread to jvm\n");
                return "";
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION:
        {
            //TODO(dgj1): error handling
            Utils::log("WAInative", "unsupported java version\n");
            return "";
        }
    }

    jobject objectActivity = app->activity->clazz;
    // Get File object for the external storage directory.
    jclass classContext = env->FindClass("android/app/Activity");
    if (!classContext)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get classContext\n");
        return "";
    }
    jmethodID methodIDgetExternalFilesDir = env->GetMethodID(classContext, "getExternalFilesDir", "(Ljava/lang/String;)Ljava/io/File;");
    if (!methodIDgetExternalFilesDir)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get methodIDgetExternalFilesDir\n");
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
        Utils::log("WAInative", "could not get classFile\n");
        return "";
    }
    jmethodID methodIDgetAbsolutePath = env->GetMethodID(classFile, "getAbsolutePath", "()Ljava/lang/String;");
    if (!methodIDgetAbsolutePath)
    {
        //TODO(dgj1): error handling
        Utils::log("WAInative", "could not get methodIDgetAbsolutePath\n");
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
        case JNI_OK:
        {
        }
        break;
        case JNI_EDETACHED:
        {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                Utils::log("WAInative", "Could not attach thread to jvm\n");
                return "";
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION:
        {
            //TODO(dgj1): error handling
            Utils::log("WAInative", "unsupported java version\n");
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
        case JNI_OK:
        {
        }
        break;
        case JNI_EDETACHED:
        {
            jint result = jvm->AttachCurrentThread(&env, nullptr);
            if (result == JNI_ERR)
            {
                //TODO(dgj1): error handling
                Utils::log("WAInative", "Could not attach thread to jvm\n");
                return;
            }
            threadAttached = true;
        }
        break;
        case JNI_EVERSION:
        {
            //TODO(dgj1): error handling
            Utils::log("WAInative", "unsupported java version\n");
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



static void onInit(void* usrPtr, struct android_app* app)
{
    Utils::log("WAInative", "onInit start");
    if (app->window == NULL)
    {
        Utils::log("WAInative native", "onInit handle return");
        return;
    }

    //Utils::log("WAInative","onInit startCamera");
    startCamera();

    Engine* engine = (Engine*)usrPtr;

    initDisplay(engine, app->window);

    std::string path = getInternalDir(app);
    extractAPKFolder(app, path, "images");
    extractAPKFolder(app, path, "images/fonts");
    extractAPKFolder(app, path, "images/textures");
    extractAPKFolder(app, path, "videos");
    extractAPKFolder(app, path, "models");
    extractAPKFolder(app, path, "shaders");
    extractAPKFolder(app, path, "calibrations");
    extractAPKFolder(app, path, "config");
    extractAPKFolder(app, path, "voc");

    std::string externalPath = getExternalDir(app);

    AppDirectories dirs;
    dirs.slDataRoot    = path;
    dirs.waiDataRoot   = path + "/";
    dirs.writableDir   = externalPath + "/";
    dirs.vocabularyDir = path + "/voc/";
    dirs.logFileDir    = externalPath + "/log/";

    CVImage::defaultPath = dirs.slDataRoot + "/images/textures/";

    AConfiguration* appConfig = AConfiguration_new();
    AConfiguration_fromAssetManager(appConfig, app->activity->assetManager);
    int32_t dpi = AConfiguration_getDensity(appConfig);
    AConfiguration_delete(appConfig);
    Utils::log("WAInative", "onInit waiApp.load");
    engine->waiApp.load(engine->width, engine->height, 1.0, 1.0, dpi, dirs);
    //engine->waiApp.loadSlam();
}

static void onTermWindow(void* usrPtr, struct android_app* app)
{
    //always completely close for now..
    Engine* engine = (Engine*)usrPtr;
    stopCamera();

    engine->waiApp.close();
    engine->waiApp.terminate();

    terminateDisplay(engine);
}

static void onDestroy(void* usrPtr)
{
    Engine* engine = (Engine*)usrPtr;

    //always completely close for now..
    //engine->waiApp.terminate();
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
            //Utils::log("WAInative","double click");
            engine->waiApp.doubleClick(sceneViewIndex, MB_left, x0, y0, K_none);
        }
        else
        {
            //Utils::log("WAInative","mouse down");
            engine->waiApp.mouseDown(sceneViewIndex, MB_left, x0, y0, K_none);
        }
    }

    // it's two fingers but one delayed (already executed mouse down)
    else if (touchCount == 2 && engine->pointersDown == 1)
    {
        //Utils::log("WAInative","mouse up + touch 2 down");
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

        //Utils::log("WAInative","touch 2 down");
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
        //Utils::log("WAInative","mouse up");
        engine->waiApp.mouseUp(sceneViewIndex, MB_left, x0, y0, K_none);
    }
    else if (touchCount == 2)
    {
        int32_t x1 = AMotionEvent_getX(event, 1);
        int32_t y1 = AMotionEvent_getY(event, 1);

        //Utils::log("WAInative","touch 2 up");
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
        //Utils::log("WAInative","mouse move");
        engine->waiApp.mouseMove(sceneViewIndex, x0, y0);
    }
    else if (touchCount == 2)
    {
        int32_t x1 = AMotionEvent_getX(event, 1);
        int32_t y1 = AMotionEvent_getY(event, 1);

        //Utils::log("WAInative","touch 2 move");
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
            case AMOTION_EVENT_ACTION_POINTER_DOWN:
            {
                handleTouchDown(engine, event);
            }
            break;

            case AMOTION_EVENT_ACTION_UP:
            case AMOTION_EVENT_ACTION_POINTER_UP:
            {
                handleTouchUp(engine, event);
            }
            break;

            case AMOTION_EVENT_ACTION_MOVE:
            {
                handleTouchMove(engine, event);
            }
        }

        return 1;
    }
    return 0;
}

static void handleLifecycleEvent(struct android_app* app, int32_t cmd)
{
    //Utils::log("WAInative","handleLifecycleEvent: called");
    Engine* engine = (Engine*)app->userData;
    switch (cmd)
    {
        case APP_CMD_SAVE_STATE:
            Utils::log("WAInative", "handleLifecycleEvent: APP_CMD_SAVE_STATE");
            break;
        case APP_CMD_INIT_WINDOW:
            Utils::log("WAInative", "handleLifecycleEvent: APP_CMD_INIT_WINDOW");
            onInit(engine, app);
            break;
        case APP_CMD_TERM_WINDOW:
            Utils::log("WAInative", "handleLifecycleEvent: APP_CMD_TERM_WINDOW");
            onTermWindow(engine, app);
            break;
        case APP_CMD_GAINED_FOCUS:
            Utils::log("WAInative", "handleLifecycleEvent: APP_CMD_GAINED_FOCUS");
            onGainedFocus(engine);
            break;
        case APP_CMD_LOST_FOCUS:
            Utils::log("WAInative", "handleLifecycleEvent: APP_CMD_LOST_FOCUS");
            onLostFocus(engine);
            break;
        case APP_CMD_CONFIG_CHANGED:
            Utils::log("WAInative", "handleLifecycleEvent: APP_CMD_CONFIG_CHANGED");
            break;
        case APP_CMD_DESTROY:
            Utils::log("WAInative", "handleLifecycleEvent: APP_CMD_DESTROY");
            onDestroy(engine);
            break;
    }
}


void android_main(struct android_app* app)
{
    try
    {
        Utils::log("WAInative", "handleLifecycleEvent: android_main");
        Engine engine = {};

        SensorsCallbacks callbacks;
        callbacks.onAcceleration = onAcceleration;
        callbacks.usrPtr         = &engine;

        app->onAppCmd     = handleLifecycleEvent;
        app->onInputEvent = handleInput;
        app->userData     = &engine;
        androidApp        = app;

        initSensorsHandler(app, &callbacks, &engine.sensorsHandler);

        while (true)
        {
            // The identifier of source (May be LOOPER_ID_MAIN, LOOPER_ID_INPUT or LOOPER_ID_USER).
            int                         ident;
            int                         events;
            struct android_poll_source* source;

            // We loop until all events are read
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
                    Utils::log("WAInative", "handleLifecycleEvent destroyRequested");
                    //onTermWindow(&engine, app);
                    return;
                }
            }

            if (engine.display != nullptr)
            {
                if (engine.waiApp.update())
                    eglSwapBuffers(engine.display, engine.surface);
            }

            //std::this_thread::sleep_for(10ms);
        }
    }
    catch (std::exception& e)
    {
        //todo: what do we do then?
        Utils::log("WAInative", e.what());
    }
}
*/