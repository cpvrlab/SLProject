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
#include <AppDemoSceneView.h>

//-----------------------------------------------------------------------------
// Some global variable for the JNI interface
JNIEnv *environment;    //! Pointer to JAVA environment used in ray tracing callback
int svIndex;            //!< SceneView index
//-----------------------------------------------------------------------------
/*! Java Native Interface (JNI) function declarations. These functions are
called by the Java interface class GLES3Lib. The function name follows the pattern
Java_{package name}_{JNI class name}_{function name}(JNIEnv* env,jclass obj,*);
The functions mostly forward to the C-Interface functions of SLProject declared
in SLInterface.h.
*/
extern "C"
{
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onInit              (JNIEnv *env, jclass obj, jint width, jint height, jint dpi, jstring filePath);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTerminate         (JNIEnv *env, jclass obj);
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateVideo       (JNIEnv *env, jclass obj);
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateScene       (JNIEnv *env, jclass obj);
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_onPaintAllViews     (JNIEnv *env, jclass obj);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onResize            (JNIEnv *env, jclass obj, jint width, jint height);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseDown         (JNIEnv *env, jclass obj, jint button, jint x, jint y);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseUp           (JNIEnv *env, jclass obj, jint button, jint x, jint y);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onMouseMove         (JNIEnv *env, jclass obj, jint x, jint y);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Up          (JNIEnv *env, jclass obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Down        (JNIEnv *env, jclass obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTouch2Move        (JNIEnv *env, jclass obj, jint x1, jint y1, jint x2, jint y2);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onDoubleClick       (JNIEnv *env, jclass obj, jint button, jint x, jint y);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onClose             (JNIEnv *env, jclass obj);
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesRotation        (JNIEnv *env, jclass obj);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onRotationQUAT      (JNIEnv *env, jclass obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW);
JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesLocation        (JNIEnv *env, jclass obj);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onLocationLLA       (JNIEnv *env, jclass obj, jdouble latitudeDEG, jdouble longitudeDEG, jdouble altitudeM, jfloat accuracyM);
JNIEXPORT jint JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoType        (JNIEnv *env, jclass obj);
JNIEXPORT jint JNICALL Java_ch_fhnw_comgr_GLES3Lib_getVideoSizeIndex   (JNIEnv *env, jclass obj);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_grabVideoFileFrame  (JNIEnv *env, jclass obj);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoImage      (JNIEnv *env, jclass obj, jint imgWidth, jint imgHeight, jbyteArray srcBuffer);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onSetupExternalDir  (JNIEnv *env, jclass obj, jstring externalDirPath);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoYUVPlanes  (JNIEnv *env, jclass obj, jint  srcW, jint srcH,
                                                                        jbyteArray yBuf, jint ySize, jint yPixStride, jint yLineStride,
                                                                        jbyteArray uBuf, jint uSize, jint uPixStride, jint uLineStride,
                                                                        jbyteArray vBuf, jint vSize, jint vPixStride, jint vLineStride);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_setCameraSize       (JNIEnv *env, jclass obj, jint sizeIndex, jint sizeIndexMax, jint width, jint height);
JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES3Lib_setDeviceParameter  (JNIEnv *env, jclass obj, jstring parameter, jstring value);
};


//-----------------------------------------------------------------------------
// external functions application code not in SLProject
extern void appDemoLoadScene(SLScene* s, SLSceneView* sv, SLSceneID sceneID);
extern bool onUpdateVideo();
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
//! Alternative SceneView creation C-function passed by slCreateSceneView
SLuint createAppDemoSceneView()
{
    SLSceneView* appDemoSV = new AppDemoSceneView();
    return appDemoSV->index();
}
//-----------------------------------------------------------------------------
//! Creates the scene and sceneview instance
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onInit(JNIEnv *env, jclass obj, jint width, jint height, jint dpi, jstring filePath)
{
    environment = env;
    const char *nativeString = env->GetStringUTFChars(filePath, 0);
    string devicePath(nativeString);
    env->ReleaseStringUTFChars(filePath, nativeString);

    SLVstring *cmdLineArgs = new SLVstring();

    SL_LOG("GUI            : Android");

    string device_path_msg = "Device path:" + devicePath;
    SL_LOG(device_path_msg.c_str(),0);

    SLApplication::calibFilePath = devicePath + "/config/"; //thats where calibrations are stored an loaded from
    SLApplication::calibIniPath = devicePath + "/calibrations/";
    CVImage::defaultPath = devicePath + "/textures/";
    CVCapture::instance()->loadCalibrations(SLApplication::getComputerInfos(), // deviceInfo string
                                            SLApplication::calibFilePath,           // for calibrations made
                                            devicePath + "/videos/");          // for videos

    ////////////////////////////////////////////////////
    slCreateAppAndScene(  *cmdLineArgs,
                          devicePath + "/shaders/",
                          devicePath + "/models/",
                          devicePath + "/textures/",
                          devicePath + "/fonts/",
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
                                (void*)createAppDemoSceneView,
                                (void*)AppDemoGui::build);
    ////////////////////////////////////////////////////////////////////

    delete cmdLineArgs;
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onTerminate(JNIEnv *env, jclass obj)
{
    AppDemoGui::saveConfig();
    slTerminate();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateVideo(JNIEnv *env, jclass obj)
{
    return onUpdateVideo();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_onUpdateScene(JNIEnv *env, jclass obj)
{
    return slUpdateScene();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_onPaintAllViews(JNIEnv *env, jclass obj)
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
    AppDemoGui::saveConfig();

    slTerminate();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesRotation(JNIEnv *env, jclass obj)
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
    return CVCapture::instance()->activeCamera->camSizeIndex();
}
//-----------------------------------------------------------------------------
//! Grabs a frame from a video file using OpenCV
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_grabVideoFileFrame(JNIEnv *env, jclass obj)
{
    SLSceneView* sv = SLApplication::scene->sceneView(0);
    CVCapture* capture = CVCapture::instance();

    // Get the current capture size of the videofile
    CVSize videoSizeBefore = capture->captureSize;

    // If viewportWdivH is negative the viewport aspect will be adapted to the video
    // aspect ratio. No cropping will be applied.
    // Android doesn't know the video file frame size before grab
    float viewportWdivH = sv->viewportWdivH();
    if (sv->viewportSameAsVideo())
        viewportWdivH = -1;

    capture->grabAndAdjustForSL(viewportWdivH);

    // If video aspect has changed we need to tell the new viewport to the sceneview
    CVSize videoSizeAfter = capture->captureSize;
    if (sv->viewportSameAsVideo() && videoSizeBefore != videoSizeAfter)
        sv->setViewportFromRatio(SLVec2i(videoSizeAfter.width, videoSizeAfter.height),
                                 sv->viewportAlign(),
                                 sv->viewportSameAsVideo());
}
//-----------------------------------------------------------------------------
//! Copies the video image data to the CVCapture class
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_copyVideoImage(JNIEnv *env, jclass obj, jint imgWidth, jint imgHeight, jbyteArray imgBuffer)
{
    SLuchar* srcLumaPtr = reinterpret_cast<SLuchar*>(env->GetByteArrayElements(imgBuffer, 0));

    if (srcLumaPtr == nullptr)
        SL_EXIT_MSG("copyVideoImage: No image data pointer passed!");

    SLSceneView* sv = SLApplication::scene->sceneView(0);
    CVCapture* capture = CVCapture::instance();
    float videoImgWdivH = (float)imgWidth / (float)imgHeight;

    if (sv->viewportSameAsVideo())
    {
        // If video aspect has changed we need to tell the new viewport to the sceneview
        if (Utils::abs(videoImgWdivH - sv->viewportWdivH()) > 0.01f)
            sv->setViewportFromRatio(SLVec2i(imgWidth, imgHeight), sv->viewportAlign(), true);
    }

    capture->loadIntoLastFrame(sv->viewportWdivH(), imgWidth, imgHeight, PF_yuv_420_888, srcLumaPtr, true);

}
//-----------------------------------------------------------------------------
//! This function is not in use and was an attempt to copy the data faster.
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

    // If viewportWdivH is negative the viewport aspect will be adapted to the video
    // aspect ratio. No cropping will be applied.
    float viewportWdivH = SLApplication::scene->sceneView(0)->viewportWdivH();
    if (SLApplication::scene->sceneView(0)->viewportSameAsVideo())
        viewportWdivH = -1;

    CVCapture::instance()->copyYUVPlanes(viewportWdivH, srcW, srcH,
                                           y, ySize, yPixStride, yLineStride,
                                           u, uSize, uPixStride, uLineStride,
                                           v, vSize, vPixStride, vLineStride);
}
//-----------------------------------------------------------------------------
//! Copies the GPS information to the SLApplicaiton class
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
//! Asks the SLApplicaiton class if the GPS sensor data is requested
extern "C" JNIEXPORT
bool JNICALL Java_ch_fhnw_comgr_GLES3Lib_usesLocation(JNIEnv *env, jclass obj)
{
    return slUsesLocation();
}
//-----------------------------------------------------------------------------
extern "C" JNIEXPORT
void JNICALL Java_ch_fhnw_comgr_GLES3Lib_onSetupExternalDir(JNIEnv *env,
                                                            jclass obj,
                                                            jstring externalDirPath)
{
    std::string externalDirPathNative = jstring2stdstring(env, externalDirPath);
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
