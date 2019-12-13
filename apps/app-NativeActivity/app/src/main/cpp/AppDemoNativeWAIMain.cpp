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
#include <AppDemoNativeCameraInterface.h>
#include <AppDemoNativeSensorsInterface.h>

#include <string>

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

static float quad[4 * 3]
        {
                -1, -1, 0,
                1, -1, 0,
                1,  1, 0,
                -1,  1, 0
        };

static int  quadi[6]
        {
                0, 1, 2,
                0, 2, 3
        };

/**
 * Shared state for our app.
 */
struct engine
{
    struct android_app* app;
    SensorsHandler*     sensorsHandler;
    int                 animating;
    EGLDisplay          display;
    EGLSurface          surface;
    EGLContext          context;
    int32_t             width;
    int32_t             height;
    GLuint              programId;

    GLuint              texID;
    GLuint              vaoID;
    int                 run;
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

static void onInit(void * usrPtr)
{
    struct engine* engine = (struct engine *)usrPtr;
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
        return;
    }

    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    engine->display     = display;
    engine->context     = context;
    engine->surface     = surface;
    engine->width       = w;
    engine->height      = h;
    engine->run         = 1;

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

    glViewport(0, 0, w, h);

    return;
}

static void onClose(void * usrPtr)
{
    struct engine* engine = (struct engine *)usrPtr;

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

    engine->display   = EGL_NO_DISPLAY;
    engine->context   = EGL_NO_CONTEXT;
    engine->surface   = EGL_NO_SURFACE;
    engine->run       = 0;
}

static void onSaveState(void * usrPtr)
{
}

static void onGainedFocus(void* usrPtr)
{
    struct engine* engine = (struct engine *)usrPtr;
    sensorsHandler_enableAccelerometer(engine->sensorsHandler);
}

static void onLostFocus(void* usrPtr)
{
    struct engine* engine = (struct engine *)usrPtr;
    sensorsHandler_disableAccelerometer(engine->sensorsHandler);
}

static void onAcceleration(void* usrPtr, float x, float y, float z)
{
    LOGI("accel = %f %f %f\n", x, y, z);
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app* app)
{
    struct engine engine;

    SensorsCallbacks callbacks;
    callbacks.onInit         = onInit;
    callbacks.onClose        = onClose;
    callbacks.onLostFocus    = onLostFocus;
    callbacks.onGainedFocus  = onGainedFocus;
    callbacks.onSaveState    = onSaveState;
    callbacks.onAcceleration = onAcceleration;

    engine.app = app;
    initSensorsHandler(app, &callbacks, &engine.sensorsHandler);

    CameraHandler * handler;
    CameraInfo * camerasInfo;
    Camera * camera;

    initCameraHandler(&handler);
    if (getBackFacingCameraList(handler, &camerasInfo) == 0)
    {
        LOGW("Can't open camera\n");
    }

    initCamera(handler, &camerasInfo[0], &camera);
    free(camerasInfo);
    cameraCaptureSession(camera,640, 360);
    destroyCameraHandler(&handler);

    uint8_t * imageBuffer = (uint8_t*)malloc(sizeof(int) * 640 * 360);

    while (engine.run)
    {
        sensorsHandler_processEvent(engine.sensorsHandler, &engine);

        if (engine.display != nullptr)
        {
            if (cameraLastFrame(camera, imageBuffer))
            {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 640, 360, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageBuffer);
            }

            // Just fill the screen with a color.
            glClearColor(0.25f, 0.78f, 0.31f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glUseProgram(engine.programId);
            glBindVertexArray(engine.vaoID);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            eglSwapBuffers(engine.display, engine.surface);
        }
    }
    destroyCamera(&camera);
}
//END_INCLUDE(all)
