#include <jni.h>
#include <stdafx.h>
#include <SLInterface.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLEnums.h>
#include <SLGLEnums.h>

//-----------------------------------------------------------------------------
// Some global variable for the JNI interface
JNIEnv *environment;    //! Pointer to JAVA environment used in ray tracing callback
int svIndex;        //!< SceneView index
//-----------------------------------------------------------------------------
/*! Java Native Interface (JNI) function declarations. These functions are
called by the Java interface class GLES3Lib. The function name follows the pattern
Java_{package name}_{JNI class name}_{function name}(JNIEnv* env,jobject obj,*);
In the function implementations we simply forward the C++ framework.
*/
extern "C"
{
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onInit(JNIEnv *env, jobject obj, jint width, jint height, jint dpi, jstring filePath);
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateAndPaint(JNIEnv *env, jobject obj);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onResize(JNIEnv *env, jobject obj, jint width, jint height);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMenuButton(JNIEnv *env, jobject obj);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseDown(JNIEnv *env, jobject obj, jint button, jint x, jint y);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseUp(JNIEnv *env, jobject obj, jint button, jint x, jint y);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseMove(JNIEnv *env, jobject obj, jint x, jint y);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Up(JNIEnv *env, jobject obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Down(JNIEnv *env, jobject obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Move(JNIEnv *env, jobject obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onDoubleClick(JNIEnv *env, jobject obj, jint button, jint x, jint y);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onRotationPYR(JNIEnv *env, jobject obj, jfloat pitchRAD, jfloat yawRAD, jfloat rollRAD);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onRotationQUAT(JNIEnv *env, jobject obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onClose(JNIEnv *env, jobject obj);
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_shouldClose(JNIEnv *env, jobject obj);
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesRotation(JNIEnv *env, jobject obj);
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesVideoImage(JNIEnv *env, jobject obj);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_passImageMat(JNIEnv *env, jobject obj, jlong matAddr);
};

//-----------------------------------------------------------------------------
//! Native ray tracing callback function that calls the Java class method GLES3Lib.RaytracingCallback
bool Java_renderRaytracingCallback()
{
    jclass klass = environment->FindClass("ch/fhnw/comgr/GLES3Lib");
    jmethodID method = environment->GetStaticMethodID(klass, "RaytracingCallback", "()Z");
    return environment->CallStaticBooleanMethod(klass,method);
}

//-----------------------------------------------------------------------------
//! Native OpenGL info string print functions used in onInit
static void printGLString(const char *name, GLenum s)
{
    const char *v = (const char *) glGetString(s);
    SL_LOG("GL %s = %s\n", name, v);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onInit(JNIEnv *env, jobject obj, jint width, jint height, jint dpi, jstring filePath)
{
    environment = env;
    const char *nativeString = env->GetStringUTFChars(filePath, 0);
    string devicePath(nativeString);
    env->ReleaseStringUTFChars(filePath, nativeString);

    SLVstring *cmdLineArgs = new SLVstring();

    SL_LOG("GUI            : Android");

    string device_path_msg = "Device path:" + devicePath;
    SL_LOG(device_path_msg.c_str());

    slCreateScene(*cmdLineArgs,
                  devicePath + "/shaders/",
                  devicePath + "/models/",
                  devicePath + "/textures/",
                  devicePath + "/calibrations/",
                  devicePath + "/config/"
    );

    svIndex = slCreateSceneView((int) width,
                                (int) height,
                                (int) dpi,
                                C_menu,
                                (void *) &Java_renderRaytracingCallback);
    delete cmdLineArgs;
}
//-----------------------------------------------------------------------------
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateAndPaint(JNIEnv *env, jobject obj)
{
    return slUpdateAndPaint(svIndex);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onResize(JNIEnv *env, jobject obj, jint width, jint height)
{
    slResize(svIndex, width, height);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMenuButton(JNIEnv *env, jobject obj)
{
    SL_LOG("onMenuButton");
    slCommand(svIndex, C_menu);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseDown(JNIEnv *env, jobject obj, jint button, jint x, jint y)
{
    SL_LOG("mouse_down");
    slMouseDown(svIndex, MB_left, x, y, K_none);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseUp(JNIEnv *env, jobject obj, jint button, jint x, jint y)
{
    slMouseUp(svIndex, MB_left, x, y, K_none);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseMove(JNIEnv *env, jobject obj, jint x, jint y)
{
    slMouseMove(svIndex, x, y);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Down(JNIEnv *env, jobject obj, jint x1, jint y1, jint x2, jint y2)
{
    slTouch2Down(svIndex, x1, y1, x2, y2);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Up(JNIEnv *env, jobject obj, jint x1, jint y1, jint x2, jint y2)
{
    slTouch2Up(svIndex, x1, y1, x2, y2);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Move(JNIEnv *env, jobject obj, jint x1, jint y1, jint x2, jint y2)
{
    slTouch2Move(svIndex, x1, y1, x2, y2);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onDoubleClick(JNIEnv *env, jobject obj, jint button, jint x, jint y)
{
    slDoubleClick(svIndex, MB_left, x, y, K_none);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onRotationPYR(JNIEnv *env, jobject obj, jfloat pitchRAD, jfloat yawRAD, jfloat rollRAD)
{
    slRotationPYR(svIndex, pitchRAD, yawRAD, rollRAD);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onRotationQUAT(JNIEnv *env, jobject obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW)
{
    slRotationQUAT(svIndex, quatX, quatY, quatZ, quatW);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onClose(JNIEnv *env, jobject obj)
{
    SL_LOG("onClose\n ");
    slTerminate();
    exit(0);
}
//-----------------------------------------------------------------------------
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_shouldClose(JNIEnv *env, jobject obj)
{
    return slShouldClose();
}
//-----------------------------------------------------------------------------
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesRotation(JNIEnv *env, jobject obj)
{
    return slUsesRotation(svIndex);
}
//-----------------------------------------------------------------------------
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesVideoImage(JNIEnv *env, jobject obj)
{
    return slUsesVideo();//slUsesVideoImage();
}
//-----------------------------------------------------------------------------
JNIEXPORT void Java_ch_fhnw_comgr_GLES3Lib_passImageMat(JNIEnv *env, jobject obj, jlong matAddr)
{
    cv::Mat &mGr = *(cv::Mat *) matAddr;
}
//-----------------------------------------------------------------------------