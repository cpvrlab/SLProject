//#############################################################################
//  File:      AppDemoAndroidJNI.cpp
//  Author:    Marcus Hudritsch, Zingg Pascal
//  Date:      Spring 2017
//  Purpose:   Android Java native interface into the SLProject C++ library
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Zingg Pascal
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <jni.h>
#include <stdafx.h>
#include <SLInterface.h>
#include <SLScene.h>
#include <AppDemoGui.h>

//-----------------------------------------------------------------------------
// Some global variable for the JNI interface
JNIEnv *environment;    //! Pointer to JAVA environment used in ray tracing callback
int svIndex;            //!< SceneView index
//-----------------------------------------------------------------------------
/*! Java Native Interface (JNI) function declarations. These functions are
called by the Java interface class GLES3Lib. The function name follows the pattern
Java_{package name}_{JNI class name}_{function name}(JNIEnv* env,jobject obj,*);
The functions mostly forward to the C-Interface functions of SLProject declared
in SLInterface.h.
*/
extern "C"
{
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onInit              (JNIEnv *env, jobject obj, jint width, jint height, jint dpi, jstring filePath);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTerminate         (JNIEnv *env, jobject obj);
JNIEXPORT jboolean  JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateAndPaint    (JNIEnv *env, jobject obj);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onResize            (JNIEnv *env, jobject obj, jint width, jint height);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseDown         (JNIEnv *env, jobject obj, jint button, jint x, jint y);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseUp           (JNIEnv *env, jobject obj, jint button, jint x, jint y);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseMove         (JNIEnv *env, jobject obj, jint x, jint y);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Up          (JNIEnv *env, jobject obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Down        (JNIEnv *env, jobject obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Move        (JNIEnv *env, jobject obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onDoubleClick       (JNIEnv *env, jobject obj, jint button, jint x, jint y);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onClose             (JNIEnv *env, jobject obj);
JNIEXPORT jboolean  JNICALL Java_ch_fhnw_comgr_GLES3Lib_shouldClose         (JNIEnv *env, jobject obj);
JNIEXPORT jboolean  JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesRotation        (JNIEnv *env, jobject obj);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onRotationQUAT      (JNIEnv *env, jobject obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW);
JNIEXPORT jboolean  JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesLocation        (JNIEnv *env, jobject obj);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_onLocationLLA       (JNIEnv *env, jobject obj, jdouble latitudeDEG, jdouble longitudeDEG, jdouble altitudeM, jfloat accuracyM);
JNIEXPORT jint      JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoType        (JNIEnv *env, jobject obj);
JNIEXPORT jint      JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoSizeIndex   (JNIEnv *env, jobject obj);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_grabVideoFileFrame  (JNIEnv *env, jobject obj);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoImage      (JNIEnv *env, jobject obj, jint imgWidth, jint imgHeight, jbyteArray srcBuffer);
JNIEXPORT void      JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoYUVPlanes  (JNIEnv *env, jobject obj, jint  srcW, jint srcH,
                                                                             jbyteArray yBuf, jint ySize, jint yPixStride, jint yLineStride,
                                                                             jbyteArray uBuf, jint uSize, jint uPixStride, jint uLineStride,
                                                                             jbyteArray vBuf, jint vSize, jint vPixStride, jint vLineStride);
};

//-----------------------------------------------------------------------------
extern void appDemoLoadScene(SLScene* s, SLSceneView* sv, SLSceneID sceneID);

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
    SL_LOG(device_path_msg.c_str(),0);

    ////////////////////////////////////////////////////
    slCreateAppAndScene(  *cmdLineArgs,
                          devicePath + "/shaders/",
                          devicePath + "/models/",
                          devicePath + "/textures/",
                          devicePath + "/videos/",
                          devicePath + "/fonts/",
                          devicePath + "/calibrations/",
                          devicePath + "/config/",
                          "AppDemoAndroid",
                          (void*)appDemoLoadScene);
    ////////////////////////////////////////////////////

    // This load the GUI configs that are locally stored
    AppDemoGui::loadConfig(dpi);

    ////////////////////////////////////////////////////////////////////
    svIndex = slCreateSceneView((int) width,
                                (int) height,
                                (int) dpi,
                                SID_Revolver,
                                (void *) &Java_renderRaytracingCallback,
                                0,
                                0,
                                (void*)AppDemoGui::build);
    ////////////////////////////////////////////////////////////////////

    delete cmdLineArgs;
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTerminate(JNIEnv *env, jobject obj)
{
    AppDemoGui::saveConfig();

    slTerminate();
}
//-----------------------------------------------------------------------------
JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateAndPaint(JNIEnv *env, jobject obj)
{
    return slUpdateAndPaint(svIndex);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onResize(JNIEnv *env, jobject obj, jint width, jint height)
{
    slResize(svIndex, width, height);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseDown(JNIEnv *env, jobject obj, jint button, jint x, jint y)
{
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
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onRotationQUAT(JNIEnv *env, jobject obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW)
{
    slRotationQUAT(quatX, quatY, quatZ, quatW);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onClose(JNIEnv *env, jobject obj)
{
    SL_LOG("onClose\n ");

    // This saves the GUI configs
    AppDemoGui::saveConfig();

    slTerminate();
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_shouldClose(JNIEnv *env, jobject obj, jboolean doClose)
{
    slShouldClose(doClose);
}
//-----------------------------------------------------------------------------
JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_shouldClose(JNIEnv *env, jobject obj)
{
    return slShouldClose();
}
//-----------------------------------------------------------------------------
JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesRotation(JNIEnv *env, jobject obj)
{
    return slUsesRotation();
}
//-----------------------------------------------------------------------------
JNIEXPORT jint JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoType(JNIEnv *env, jobject obj)
{
    return slGetVideoType();
}
//-----------------------------------------------------------------------------
JNIEXPORT jint JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoSizeIndex(JNIEnv *env, jobject obj)
{
    return slGetVideoSizeIndex();
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_grabVideoFileFrame(JNIEnv *env, jobject obj)
{
    return slGrabVideoFileFrame();
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoImage(JNIEnv *env, jobject obj, jint imgWidth, jint imgHeight, jbyteArray imgBuffer)
{
    SLuchar* srcLumaPtr = reinterpret_cast<SLuchar*>(env->GetByteArrayElements(imgBuffer, 0));

    if (srcLumaPtr == nullptr)
        SL_EXIT_MSG("copyVideoImage: No image data pointer passed!");

    slCopyVideoImage(imgWidth, imgHeight, PF_yuv_420_888, srcLumaPtr, true);
}
//-----------------------------------------------------------------------------

JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoYUVPlanes(JNIEnv *env, jobject obj, jint  srcW, jint srcH,
                                                                      jbyteArray yBuf, jint ySize, jint yPixStride, jint yLineStride,
                                                                      jbyteArray uBuf, jint uSize, jint uPixStride, jint uLineStride,
                                                                      jbyteArray vBuf, jint vSize, jint vPixStride, jint vLineStride)
{
    // Cast jbyteArray to unsigned char pointer
    SLuchar* y = reinterpret_cast<SLuchar*>(env->GetByteArrayElements(yBuf, 0));
    SLuchar* u = reinterpret_cast<SLuchar*>(env->GetByteArrayElements(uBuf, 0));
    SLuchar* v = reinterpret_cast<SLuchar*>(env->GetByteArrayElements(vBuf, 0));

    if (y == nullptr) SL_EXIT_MSG("copyVideoYUVPlanes: No pointer for y-buffer passed!");
    if (u == nullptr) SL_EXIT_MSG("copyVideoYUVPlanes: No pointer for u-buffer passed!");
    if (v == nullptr) SL_EXIT_MSG("copyVideoYUVPlanes: No pointer for v-buffer passed!");

    slCopyVideoYUVPlanes(srcW, srcH,
                         y, ySize, yPixStride, yLineStride,
                         u, uSize, uPixStride, uLineStride,
                         v, vSize, vPixStride, vLineStride);
}
//-----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onLocationLLA(JNIEnv *env,
                                                                 jobject obj,
                                                                 jdouble latitudeDEG,
                                                                 jdouble longitudeDEG,
                                                                 jdouble altitudeM,
                                                                 jfloat accuracyM)
{
    slLocationLLA(latitudeDEG, longitudeDEG, altitudeM, accuracyM);
}
//-----------------------------------------------------------------------------
JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesLocation(JNIEnv *env, jobject obj)
{
    return slUsesLocation();
}
//-----------------------------------------------------------------------------