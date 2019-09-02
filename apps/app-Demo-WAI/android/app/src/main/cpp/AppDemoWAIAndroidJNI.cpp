//#############################################################################
//  File:      AppDemoAndroidJNI.cpp
//  Author:    Marcus Hudritsch
//  Date:      Spring 2017
//  Purpose:   Android Java native interface into the SLProject C++ library
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Zingg Pascal
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <jni.h>
#include <stdafx.h>
#include <SLInterface.h>
#include <SLScene.h>
#include <SLApplication.h>
#include <CVCapture.h>
#include <AppDemoGui.h>
#include <AppDemoGui.h>
#include <AppWAI.h>

//-----------------------------------------------------------------------------
// Some global variable for the JNI interface
JNIEnv *environment;    //! Pointer to JAVA environment used in ray tracing callback
int svIndex;            //!< SceneView index
AppWAIDirectories dirs;
//-----------------------------------------------------------------------------
/*! Java Native Interface (JNI) function declarations. These functions are
called by the Java interface class GLES3Lib. The function name follows the pattern
Java_{package name}_{JNI class name}_{function name}(JNIEnv* env,jclass obj,*);
The functions mostly forward to the C-Interface functions of SLProject declared
in SLInterface.h.
*/
extern "C"
{
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onInit              (JNIEnv *env, jclass obj, jint width, jint height, jint dpi, jstring filePath);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTerminate         (JNIEnv *env, jclass obj);
JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateTracking    (JNIEnv *env, jclass obj);
JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateScene       (JNIEnv *env, jclass obj);
JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_onPaintAllViews     (JNIEnv *env, jclass obj);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onResize            (JNIEnv *env, jclass obj, jint width, jint height);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseDown         (JNIEnv *env, jclass obj, jint button, jint x, jint y);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseUp           (JNIEnv *env, jclass obj, jint button, jint x, jint y);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseMove         (JNIEnv *env, jclass obj, jint x, jint y);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Up          (JNIEnv *env, jclass obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Down        (JNIEnv *env, jclass obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Move        (JNIEnv *env, jclass obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onDoubleClick       (JNIEnv *env, jclass obj, jint button, jint x, jint y);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onClose             (JNIEnv *env, jclass obj);
JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesRotation        (JNIEnv *env, jclass obj);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onRotationQUAT      (JNIEnv *env, jclass obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW);
JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesLocation        (JNIEnv *env, jclass obj);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onLocationLLA       (JNIEnv *env, jclass obj, jdouble latitudeDEG, jdouble longitudeDEG, jdouble altitudeM, jfloat accuracyM);
JNIEXPORT jint     JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoType        (JNIEnv *env, jclass obj);
JNIEXPORT jint     JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoSizeIndex   (JNIEnv *env, jclass obj);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_grabVideoFileFrame  (JNIEnv *env, jclass obj);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoImage      (JNIEnv *env, jclass obj, jint imgWidth, jint imgHeight, jbyteArray srcBuffer);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_onSetupExternalDir  (JNIEnv *env, jclass obj, jstring externalDirPath);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoYUVPlanes  (JNIEnv *env, jclass obj, jint  srcW, jint srcH,
                                                                            jbyteArray yBuf, jint ySize, jint yPixStride, jint yLineStride,
                                                                            jbyteArray uBuf, jint uSize, jint uPixStride, jint uLineStride,
                                                                            jbyteArray vBuf, jint vSize, jint vPixStride, jint vLineStride);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_setCameraSize       (JNIEnv *env, jclass obj, jint sizeIndex, jint sizeIndexMax, jint width, jint height);
JNIEXPORT void     JNICALL Java_ch_fhnw_comgr_GLES3Lib_setDeviceParameter  (JNIEnv *env, jclass obj, jstring parameter, jstring value);
};


//-----------------------------------------------------------------------------
// external functions application code not in SLProject
extern bool onUpdateTracking();
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//! Native OpenGL info string print functions used in onInit
static void printGLString(const char *name, GLenum s)
{
    const char *v = (const char *) glGetString(s);
    SL_LOG("GL %s = %s\n", name, v);
}
//-----------------------------------------------------------------------------
std::string jstring2stdstring(JNIEnv *env, jstring jStr)
{
    if (!jStr) return "";
    jboolean isCopy;
    const char* chars = env->GetStringUTFChars(jStr, &isCopy);
    std::string stdString(chars);
    env->ReleaseStringUTFChars(jStr, chars);
    return stdString;
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onInit(JNIEnv *env, jclass obj, jint width, jint height, jint dpi, jstring filePath)
{
    environment = env;
    const char *nativeString = env->GetStringUTFChars(filePath, 0);
    dirs.slDataRoot = std::string(nativeString);
    dirs.waiDataRoot = std::string(nativeString);
    env->ReleaseStringUTFChars(filePath, nativeString);

    CVImage::defaultPath = dirs.slDataRoot + "/images/textures/";
    CVCapture::instance()->loadCalibrations(SLApplication::getComputerInfos(), // deviceInfo string
                                            dirs.slDataRoot + "/config/",           // for calibrations made
                                            dirs.slDataRoot + "/calibrations/",     // for calibInitPath
                                            dirs.slDataRoot + "/videos/");          // for videos



    svIndex = WAIApp::load(width, height, 1.0, 1.0, dpi, &dirs);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTerminate(JNIEnv *env, jclass obj)
{
    //AppDemoGui::saveConfig();
    slTerminate();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateTracking(JNIEnv *env, jclass obj)
{
    return WAIApp::update();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateScene(JNIEnv *env, jclass obj)
{
    return slUpdateScene();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_onPaintAllViews(JNIEnv *env, jclass obj)
{
    return slPaintAllViews();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onResize(JNIEnv *env, jclass obj, jint width, jint height)
{
    slResize(svIndex, width, height);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseDown(JNIEnv *env, jclass obj, jint button, jint x, jint y)
{
    slMouseDown(svIndex, MB_left, x, y, K_none);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseUp(JNIEnv *env, jclass obj, jint button, jint x, jint y)
{
    slMouseUp(svIndex, MB_left, x, y, K_none);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseMove(JNIEnv *env, jclass obj, jint x, jint y)
{
    slMouseMove(svIndex, x, y);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Down(JNIEnv *env, jclass obj, jint x1, jint y1, jint x2, jint y2)
{
    slTouch2Down(svIndex, x1, y1, x2, y2);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Up(JNIEnv *env, jclass obj, jint x1, jint y1, jint x2, jint y2)
{
    slTouch2Up(svIndex, x1, y1, x2, y2);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Move(JNIEnv *env, jclass obj, jint x1, jint y1, jint x2, jint y2)
{
    slTouch2Move(svIndex, x1, y1, x2, y2);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onDoubleClick(JNIEnv *env, jclass obj, jint button, jint x, jint y)
{
    slDoubleClick(svIndex, MB_left, x, y, K_none);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onRotationQUAT(JNIEnv *env, jclass obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW)
{
    slRotationQUAT(quatX, quatY, quatZ, quatW);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onClose(JNIEnv *env, jclass obj)
{
    SL_LOG("onClose\n ");

    // This saves the GUI configs
    // AppDemoGui::saveConfig();

    slTerminate();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesRotation(JNIEnv *env, jclass obj)
{
    return slUsesRotation();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
jint JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoType(JNIEnv *env, jclass obj)
{
    return (int)CVCapture::instance()->videoType();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
jint JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoSizeIndex(JNIEnv *env, jclass obj)
{
    return -1;//CVCapture::instance()->activeCalib->camSizeIndex();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_grabVideoFileFrame(JNIEnv *env, jclass obj)
{
    float scrWdivH = SLApplication::scene->sceneView(0)->scrWdivH();
    return CVCapture::instance()->grabAndAdjustForSL(scrWdivH);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoImage(JNIEnv *env, jclass obj, jint imgWidth, jint imgHeight, jbyteArray imgBuffer)
{
    SLuchar* srcLumaPtr = reinterpret_cast<SLuchar*>(env->GetByteArrayElements(imgBuffer, 0));

    if (srcLumaPtr == nullptr)
        SL_EXIT_MSG("copyVideoImage: No image data pointer passed!");

    float scrWdivH = SLApplication::scene->sceneView(0)->scrWdivH();
    CVCapture::instance()->loadIntoLastFrame(scrWdivH, imgWidth, imgHeight, PF_yuv_420_888, srcLumaPtr, true);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoYUVPlanes(JNIEnv *env, jclass obj, jint  srcW, jint srcH,
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

    float scrWdivH = SLApplication::scene->sceneView(0)->scrWdivH();
    CVCapture::instance()->copyYUVPlanes(scrWdivH, srcW, srcH,
                                           y, ySize, yPixStride, yLineStride,
                                           u, uSize, uPixStride, uLineStride,
                                           v, vSize, vPixStride, vLineStride);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onLocationLLA(JNIEnv *env,
                                                       jclass obj,
                                                       jdouble latitudeDEG,
                                                       jdouble longitudeDEG,
                                                       jdouble altitudeM,
                                                       jfloat accuracyM)
{
    slLocationLLA(latitudeDEG, longitudeDEG, altitudeM, accuracyM);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
jboolean JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesLocation(JNIEnv *env, jclass obj)
{
    return slUsesLocation();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onSetupExternalDir(JNIEnv *env,
                                                            jclass obj,
                                                            jstring  externalDirPath)
{
    std::string externalDirPathNative = jstring2stdstring(env, externalDirPath);
    dirs.writableDir = externalDirPathNative;
    slSetupExternalDir(externalDirPathNative);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_setCameraSize(JNIEnv *env,
                                                                 jclass obj,
                                                                 jint sizeIndex,
                                                                 jint sizeIndexMax,
                                                                 jint width,
                                                                 jint height)
{
    CVCapture::instance()->setCameraSize(sizeIndex, sizeIndexMax, width, height);
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_setDeviceParameter(JNIEnv *env,
                                                                      jclass obj,
                                                                      jstring parameter,
                                                                      jstring value)
{
    std::string par = jstring2stdstring(env, parameter);
    std::string val = jstring2stdstring(env, value);
    slSetDeviceParameter(par.c_str(), val.c_str());
}
//-----------------------------------------------------------------------------