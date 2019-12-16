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

#include <android/sensor.h>
#include <android/log.h>
#include <android_native_app_glue.h>

#include <camera/NdkCameraCaptureSession.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraError.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadata.h>
#include <camera/NdkCameraMetadataTags.h>
#include <camera/NdkCameraWindowType.h>
#include <camera/NdkCaptureRequest.h>
#include <android/native_window.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>

#include <string>

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

/**
 * Our saved state data.
 */
struct saved_state
{
    float   angle;
    int32_t x;
    int32_t y;
};

static float quad[4 * 3]{
  -1,
  -1,
  0,
  1,
  -1,
  0,
  1,
  1,
  0,
  -1,
  1,
  0};

static int quadi[6]{
  0,
  1,
  2,
  0,
  2,
  3};

/**
 * Shared state for our app.
 */
struct engine
{
    struct android_app* app;

    ASensorManager*    sensorManager;
    const ASensor*     accelerometerSensor;
    ASensorEventQueue* sensorEventQueue;

    int                animating;
    EGLDisplay         display;
    EGLSurface         surface;
    EGLContext         context;
    int32_t            width;
    int32_t            height;
    struct saved_state state;
    GLuint             programId;

    GLuint texID;
    GLuint vaoID;
};

static std::string vertexShaderSource = "#version 320 es\n"
                                        "layout (location = 0) in vec3 vcoords;\n"
                                        "out vec2 texcoords;\n"
                                        "\n"
                                        "void main()\n"
                                        "{\n"
                                        "    texcoords = 0.5 * (vcoords.xy + vec2(1.0));\n"
                                        "    gl_Position = vec4(vcoords, 1.0);\n"
                                        "}\n";
static std::string fragShaderSource = "#version 320 es\n"
                                      "precision highp float;\n"
                                      "in vec2 texcoords;\n"
                                      "uniform sampler2D tex;\n"
                                      "out vec4 color;\n"
                                      "\n"
                                      "void main()\n"
                                      "{\n"
                                      //"   color = vec4(1.0f, 0.2f, 0.35f, 1.0f);\n"
                                      "     color = texture(tex, texcoords);\n"
                                      "}\n";

GLuint buildShaderFromSource(std::string source, GLenum shaderType)
{
    // Compile Shader code
    GLuint shaderHandle = glCreateShader(shaderType);

    const char* src = source.c_str();

    glShaderSource(shaderHandle, 1, &src, nullptr);
    glCompileShader(shaderHandle);

    // Check compile success
    GLint compileSuccess;
    glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &compileSuccess);

    GLint logSize = 0;
    glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &logSize);

    GLchar* log = new GLchar[logSize];

    glGetShaderInfoLog(shaderHandle, logSize, nullptr, log);

    if (!compileSuccess)
    {
        LOGW("AAAA Cannot compile shader %s\n", log);
        LOGW("AAAA %s\n", src);
        exit(1);
    }

    return shaderHandle;
}

/**
 * Initialize an EGL context for the current display.
 */
static int engine_init_display(struct engine* engine)
{
    // initialize OpenGL ES and EGL

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
    EGLint       w, h, format;
    EGLint       numConfigs;
    EGLConfig    config;
    EGLSurface   surface;
    EGLContext   context;

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
    surface = eglCreateWindowSurface(display, config, engine->app->window, NULL);

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
        return -1;
    }

    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    engine->display     = display;
    engine->context     = context;
    engine->surface     = surface;
    engine->width       = w;
    engine->height      = h;
    engine->state.angle = 0;

    // Check openGL on the system
    auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
    for (auto name : opengl_info)
    {
        auto info = glGetString(name);
        LOGI("OpenGL Info: %s", info);
    }

    GLuint vertexShaderId = buildShaderFromSource(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragShaderId   = buildShaderFromSource(fragShaderSource, GL_FRAGMENT_SHADER);

    GLuint programId = glCreateProgram();
    glAttachShader(programId, vertexShaderId);
    glAttachShader(programId, fragShaderId);
    glLinkProgram(programId);

    glDeleteShader(vertexShaderId);
    glDeleteShader(fragShaderId);

    engine->programId = programId;

    glGenTextures(1, &engine->texID);
    glBindTexture(GL_TEXTURE_2D, engine->texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    int texLoc = glGetUniformLocation(engine->programId, "tex");

    LOGW("texLoc = %d\n", texLoc);

    // loop waiting for stuff to do.

    glGenVertexArrays(1, &engine->vaoID);
    glBindVertexArray(engine->vaoID);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    GLuint vboI;
    glGenBuffers(1, &vboI);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboI);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadi), quadi, GL_STATIC_DRAW);

    int loc = 0;
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    glViewport(0, 0, 960, 1920);

    return 0;
}

/**
 * Tear down the EGL context currently associated with the display.
 */
static void engine_term_display(struct engine* engine)
{
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
    engine->animating = 0;
    engine->display   = EGL_NO_DISPLAY;
    engine->context   = EGL_NO_CONTEXT;
    engine->surface   = EGL_NO_SURFACE;
}

/**
 * Process the next input event.
 */
static int32_t engine_handle_input(struct android_app* app, AInputEvent* event)
{
    struct engine* engine = (struct engine*)app->userData;
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION)
    {
        engine->animating = 1;
        engine->state.x   = AMotionEvent_getX(event, 0);
        engine->state.y   = AMotionEvent_getY(event, 0);
        return 1;
    }
    return 0;
}

/**
 * Process the next main command.
 */
static void engine_handle_cmd(struct android_app* app, int32_t cmd)
{
    struct engine* engine = (struct engine*)app->userData;
    switch (cmd)
    {
        case APP_CMD_SAVE_STATE:
            // The system has asked us to save our current state.  Do so.
            engine->app->savedState                         = malloc(sizeof(struct saved_state));
            *((struct saved_state*)engine->app->savedState) = engine->state;
            engine->app->savedStateSize                     = sizeof(struct saved_state);
            break;
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            if (engine->app->window != NULL)
            {
                engine_init_display(engine);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            engine_term_display(engine);
            break;
        case APP_CMD_GAINED_FOCUS:
            // When our app gains focus, we start monitoring the accelerometer.
            if (engine->accelerometerSensor != NULL)
            {
                ASensorEventQueue_enableSensor(engine->sensorEventQueue,
                                               engine->accelerometerSensor);
                // We'd like to get 60 events per second (in us).
                ASensorEventQueue_setEventRate(engine->sensorEventQueue,
                                               engine->accelerometerSensor,
                                               (1000L / 60) * 1000);
            }
            break;
        case APP_CMD_LOST_FOCUS:
            // When our app loses focus, we stop monitoring the accelerometer.
            // This is to avoid consuming battery while not being used.
            if (engine->accelerometerSensor != NULL)
            {
                ASensorEventQueue_disableSensor(engine->sensorEventQueue,
                                                engine->accelerometerSensor);
            }
            // Also stop animating.
            engine->animating = 0;
            break;
    }
}

/*
 * AcquireASensorManagerInstance(void)
 *    Workaround ASensorManager_getInstance() deprecation false alarm
 *    for Android-N and before, when compiling with NDK-r15
 */
#include <dlfcn.h>
ASensorManager* AcquireASensorManagerInstance(android_app* app)
{
    if (!app)
        return nullptr;

    typedef ASensorManager* (*PF_GETINSTANCEFORPACKAGE)(const char* name);
    void*                    androidHandle             = dlopen("libandroid.so", RTLD_NOW);
    PF_GETINSTANCEFORPACKAGE getInstanceForPackageFunc = (PF_GETINSTANCEFORPACKAGE)
      dlsym(androidHandle, "ASensorManager_getInstanceForPackage");
    if (getInstanceForPackageFunc)
    {
        JNIEnv* env = nullptr;
        app->activity->vm->AttachCurrentThread(&env, NULL);

        jclass    android_content_Context = env->GetObjectClass(app->activity->clazz);
        jmethodID midGetPackageName       = env->GetMethodID(android_content_Context,
                                                       "getPackageName",
                                                       "()Ljava/lang/String;");
        jstring   packageName             = (jstring)env->CallObjectMethod(app->activity->clazz,
                                                             midGetPackageName);

        const char*     nativePackageName = env->GetStringUTFChars(packageName, 0);
        ASensorManager* mgr               = getInstanceForPackageFunc(nativePackageName);
        env->ReleaseStringUTFChars(packageName, nativePackageName);
        app->activity->vm->DetachCurrentThread();
        if (mgr)
        {
            dlclose(androidHandle);
            return mgr;
        }
    }

    typedef ASensorManager* (*PF_GETINSTANCE)();
    PF_GETINSTANCE getInstanceFunc = (PF_GETINSTANCE)
      dlsym(androidHandle, "ASensorManager_getInstance");
    // by all means at this point, ASensorManager_getInstance should be available
    assert(getInstanceFunc);
    dlclose(androidHandle);

    return getInstanceFunc();
}

void cameraDisconnected(void* context, ACameraDevice* device)
{
    ACameraDevice_close(device);

    LOGW("Camera device closed, exiting...");
    exit(1);

    //TODO(dgj1): actually implement error handling
}

void cameraError(void* context, ACameraDevice* device, int error)
{
    ACameraDevice_close(device);

    LOGW("Camera device encountered error no %i, exiting...", error);
    exit(1);

    //TODO(dgj1): actually implement error handling
}

ACameraDevice* setupCamera()
{
    //TODO(dgj1): we have to prompt the user for the camera permission... so far
    // the only way we found to do this is through the JNI -.-
    ACameraManager* manager    = ACameraManager_create();
    ACameraIdList*  cameraList = nullptr;

    if (ACameraManager_getCameraIdList(manager, &cameraList) != ACAMERA_OK)
    {
        LOGW("BBBB Could not get camera list");
        exit(1);
    }

    for (int i = 0; i < cameraList->numCameras; i++)
    {
        ACameraMetadata* characteristics;
        if (ACameraManager_getCameraCharacteristics(manager,
                                                    cameraList->cameraIds[i],
                                                    &characteristics) == ACAMERA_OK)
        {

            ACameraMetadata_const_entry lensFacing;
            ACameraMetadata_getConstEntry(characteristics, ACAMERA_LENS_FACING, &lensFacing);

            if (*lensFacing.data.u8 == ACAMERA_LENS_FACING_BACK)
            {
                LOGI("camera %i: %s is back facing\n", i, cameraList->cameraIds[i]);

                ACameraDevice_StateCallbacks callbacks;
                callbacks.onDisconnected = cameraDisconnected;
                callbacks.onError        = cameraError;

                ACameraDevice* device;

                if (ACameraManager_openCamera(manager, cameraList->cameraIds[i], &callbacks, &device) == ACAMERA_OK)
                {
                    LOGI("opened camera %s", cameraList->cameraIds[i]);
                    return device;
                }
            }
            else
            {
                LOGI("camera %i: %s is front facing\n", i, cameraList->cameraIds[i]);
            }
        }
    }

    return nullptr;
}

void onSessionClosed(void* ctx, ACameraCaptureSession* ses)
{
    LOGW("session closed");
}
void onSessionReady(void* ctx, ACameraCaptureSession* ses)
{
    LOGW("session ready");
}
void onSessionActive(void* ctx, ACameraCaptureSession* ses)
{
    LOGW("session active");
}

void onNewCameraFrame(void* context, AImageReader* reader)
{
    AImage* image;
    AImageReader_acquireNextImage(reader, &image);
}

AImageReader* cameraCapture(ACameraDevice* device)
{
    AImageReader* reader;
    if (AImageReader_new(640, 360, AIMAGE_FORMAT_YUV_420_888, 2, &reader) != AMEDIA_OK)
    {
        LOGW("Could not create image reader\n");
        return nullptr;
    }

    ANativeWindow* outputNativeWindow;
    AImageReader_getWindow(reader, &outputNativeWindow);

    // Avoid native window to be deleted
    ANativeWindow_acquire(outputNativeWindow);

    ACaptureSessionOutput* output;
    ACaptureSessionOutput_create(outputNativeWindow, &output);

    ACaptureSessionOutputContainer* outputContainer;
    ACaptureSessionOutputContainer_create(&outputContainer);
    ACaptureSessionOutputContainer_add(outputContainer, output);

    ACameraOutputTarget* target;
    ACameraOutputTarget_create(outputNativeWindow, &target);

    ACaptureRequest* request;
    ACameraDevice_createCaptureRequest(device, TEMPLATE_PREVIEW, &request);
    ACaptureRequest_addTarget(request, target);

    ACameraCaptureSession_stateCallbacks capSessionCallbacks;
    capSessionCallbacks.onActive = onSessionActive;
    capSessionCallbacks.onReady  = onSessionReady;
    capSessionCallbacks.onClosed = onSessionClosed;

    ACameraCaptureSession* captureSession;
    if (ACameraDevice_createCaptureSession(device, outputContainer, &capSessionCallbacks, &captureSession) != AMEDIA_OK)
    {
        LOGW("Could not create capture session\n");
        return nullptr;
    }
    ACameraCaptureSession_setRepeatingRequest(captureSession, nullptr, 1, &request, nullptr);

    return reader;
}

static const int kMaxChannelValue = 262143;

static inline uint32_t YUV2RGB(int nY, int nU, int nV)
{
    nY -= 16;
    nU -= 128;
    nV -= 128;
    if (nY < 0) nY = 0;

    // This is the floating point equivalent. We do the conversion in integer
    // because some Android devices do not have floating point in hardware.
    // nR = (int)(1.164 * nY + 1.596 * nV);
    // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
    // nB = (int)(1.164 * nY + 2.018 * nU);

    int nR = (int)(1192 * nY + 1634 * nV);
    int nG = (int)(1192 * nY - 833 * nV - 400 * nU);
    int nB = (int)(1192 * nY + 2066 * nU);

    nR = std::min(kMaxChannelValue, std::max(0, nR));
    nG = std::min(kMaxChannelValue, std::max(0, nG));
    nB = std::min(kMaxChannelValue, std::max(0, nB));

    nR = (nR >> 10) & 0xff;
    nG = (nG >> 10) & 0xff;
    nB = (nB >> 10) & 0xff;

    return 0xff000000 | (nR << 16) | (nG << 8) | nB;
}

void imageConverter(uint8_t* buf, AImage* image)
{
    AImageCropRect srcRect;
    AImage_getCropRect(image, &srcRect);
    int32_t  yStride, uvStride;
    uint8_t *yPixel, *uPixel, *vPixel;
    int32_t  yLen, uLen, vLen;
    AImage_getPlaneRowStride(image, 0, &yStride);
    AImage_getPlaneRowStride(image, 1, &uvStride);
    AImage_getPlaneData(image, 0, &yPixel, &yLen);
    AImage_getPlaneData(image, 1, &vPixel, &vLen);
    AImage_getPlaneData(image, 2, &uPixel, &uLen);
    int32_t uvPixelStride;
    AImage_getPlanePixelStride(image, 1, &uvPixelStride);

    int32_t height;
    int32_t width;
    AImage_getHeight(image, &height);
    AImage_getWidth(image, &width);

    uint32_t* out = (uint32_t*)(buf);
    for (int32_t row = 0; row < height; row++)
    {
        const uint8_t* pY = yPixel + srcRect.left + yStride * (row + srcRect.top);

        int32_t        uv_row_start = uvStride * ((row + srcRect.top) >> 1);
        const uint8_t* pU           = uPixel + uv_row_start + (srcRect.left >> 1);
        const uint8_t* pV           = vPixel + uv_row_start + (srcRect.left >> 1);

        for (int32_t x = 0; x < width; x++)
        {
            const int32_t uv_offset = (x >> 1) * uvPixelStride;
            out[x]                  = YUV2RGB(pY[x], pU[uv_offset], pV[uv_offset]);
        }
        out += width;
    }
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app* state)
{
    struct engine engine;

    memset(&engine, 0, sizeof(engine));
    state->userData     = &engine;
    state->onAppCmd     = engine_handle_cmd;
    state->onInputEvent = engine_handle_input;
    engine.app          = state;

    // Prepare to monitor accelerometer
    engine.sensorManager       = AcquireASensorManagerInstance(state);
    engine.accelerometerSensor = ASensorManager_getDefaultSensor(
      engine.sensorManager,
      ASENSOR_TYPE_ACCELEROMETER);
    engine.sensorEventQueue = ASensorManager_createEventQueue(
      engine.sensorManager,
      state->looper,
      LOOPER_ID_USER,
      NULL,
      NULL);

    if (state->savedState != NULL)
    {
        // We are starting with a previous saved state; restore from it.
        engine.state = *(struct saved_state*)state->savedState;
    }

    ACameraDevice* cameraDevice = setupCamera();
    AImageReader*  reader       = cameraCapture(cameraDevice);
    if (reader == nullptr)
    {
        LOGW("Could not create reader\n");
        return;
    }

    uint8_t* imageBuffer = (uint8_t*)malloc(sizeof(int) * 640 * 360);

    while (1)
    {
        // Read all pending events.
        int                         ident;
        int                         events;
        struct android_poll_source* source;

        // If not animating, we will block forever waiting for events.
        // If animating, we loop until all events are read, then continue
        // to draw the next frame of animation.
        while ((ident = ALooper_pollAll(engine.animating ? 0 : -1, NULL, &events, (void**)&source)) >= 0)
        {

            // Process this event.
            if (source != NULL)
            {
                source->process(state, source);
            }

            if (engine.display != nullptr)
            {
                AImage* image;
                if (AImageReader_acquireLatestImage(reader, &image) == AMEDIA_OK)
                {
                    int32_t format;
                    int32_t height;
                    int32_t width;
                    AImage_getFormat(image, &format);
                    AImage_getHeight(image, &height);
                    AImage_getHeight(image, &width);

                    imageConverter(imageBuffer, image);
                    uint8_t* ptr = imageBuffer + ((320 + 640 * 180) * 4);
                    LOGW("%d %d %d", ptr[0], ptr[1], ptr[2]);

                    AImage_delete(image);

                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageBuffer);
                }

                // Just fill the screen with a color.
                glClearColor(0.25f, 0.78f, 0.31f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);

                glUseProgram(engine.programId);
                //glActiveTexture(GL_TEXTURE0);
                //glUniform1i(texLoc, 0);
                glBindVertexArray(engine.vaoID);
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

                eglSwapBuffers(engine.display, engine.surface);
            }

            // If a sensor has data, process it now.
            if (ident == LOOPER_ID_USER)
            {
                if (engine.accelerometerSensor != NULL)
                {
                    ASensorEvent event;
                    while (ASensorEventQueue_getEvents(engine.sensorEventQueue,
                                                       &event,
                                                       1) > 0)
                    {
                        /*
                           LOGI("accelerometer: x=%f y=%f z=%f",
                             event.acceleration.x,
                             event.acceleration.y,
                             event.acceleration.z)
                        */
                    }
                }
            }

            // Check if we are exiting.
            if (state->destroyRequested != 0)
            {
                engine_term_display(&engine);
                return;
            }
        }
    }
}
//END_INCLUDE(all)
