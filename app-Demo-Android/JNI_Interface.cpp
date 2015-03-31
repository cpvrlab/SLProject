//#############################################################################
//  File:      JNI_Interface.cpp
//  Purpose:   Delcaration of the main Scene Library C-Interface. Only these 
//             functions should called by the OS-dependend GUI applications. 
//             These functions can be called from any C, C++ or ObjectiveC GUI 
//             framework or by a native API such as Java Native Interface 
//             (JNI). See the implementation for more information.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#if defined(SL_OS_ANDROID) && defined(SL_GUI_JAVA)

#include <stdafx.h>
#include <SLInterface.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLEnums.h>

//-----------------------------------------------------------------------------
// Some global variable for the JNI interface
JNIEnv* environment;    //! Pointer to JAVA environment used in ray tracing callback
SLint   svIndex;        //!< SceneView index
//-----------------------------------------------------------------------------
/*! Java Native Interface (JNI) function deklarations. These functions are 
called by the Java interface class GLES2Lib. The function name follows the pattern
Java_{package name}_{JNI class name}_{function name}(JNIEnv* env,jobject obj,*);
In the function implementations we simply forward the C++ framework.
*/
extern "C"
{  JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES2Lib_onInit          (JNIEnv* env, jobject obj, jint width, jint height, jint dpi, jstring filePath);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onUpdateAndPaint(JNIEnv* env, jobject obj);
   JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES2Lib_onResize        (JNIEnv* env, jobject obj, jint width, jint height);
   JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES2Lib_onMenuButton    (JNIEnv* env, jobject obj);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onMouseDown     (JNIEnv* env, jobject obj, jint button, jint x, jint y);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onMouseUp       (JNIEnv* env, jobject obj, jint button, jint x, jint y);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onMouseMove     (JNIEnv* env, jobject obj, jint x, jint y);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onTouch2Up      (JNIEnv* env, jobject obj, jint x1, jint y1, jint x2, jint y2);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onTouch2Down    (JNIEnv* env, jobject obj, jint x1, jint y1, jint x2, jint y2);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onTouch2Move    (JNIEnv* env, jobject obj, jint x1, jint y1, jint x2, jint y2);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onDoubleClick   (JNIEnv* env, jobject obj, jint button, jint x, jint y);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onRotationPYR   (JNIEnv* env, jobject obj, jfloat pitchRAD, jfloat yawRAD, jfloat rollRAD);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onRotationQUAT  (JNIEnv* env, jobject obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW);
   JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES2Lib_onClose         (JNIEnv* env, jobject obj);
   JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_shouldClose     (JNIEnv* env, jobject obj);
};

//! Native ray tracing callback function that calls the Java class method GLES2Lib.RaytracingCallback
bool Java_renderRaytracingCallback()
{
	jclass klass = environment->FindClass("ch/fhnw/comgr/GLES2Lib");
	jmethodID method = environment->GetStaticMethodID(klass, "RaytracingCallback", "()Z");
   return environment->CallStaticObjectMethod(klass, method);
}

//! Native OpenGL info string print functions used in onInit
static void printGLString(const char *name, GLenum s) 
{
    const char *v = (const char *) glGetString(s);
    SL_LOG("GL %s = %s\n", name, v);
}

JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES2Lib_onInit(JNIEnv* env, jobject obj, jint width, jint height, jint dpi, jstring filePath)
{   
    environment = env;
    const char *nativeString = env->GetStringUTFChars(filePath, 0);
    string devicePath(nativeString);
    env->ReleaseStringUTFChars(filePath, nativeString);
    
    SLVstring* cmdLineArgs = new SLVstring();
    
    SL_LOG("GUI             : Android");

    slCreateScene(devicePath + "/shaders/",
                  devicePath + "/models/", 
                  devicePath + "/textures/");

    svIndex = slCreateSceneView((int)width, 
                                (int)height, 
                                (int)dpi,
                                cmdSceneMeshLoad,
                                *cmdLineArgs,
                                (void*)&Java_renderRaytracingCallback);
    delete cmdLineArgs;
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onUpdateAndPaint(JNIEnv* env, jobject obj)
{
   return slUpdateAndPaint(svIndex);
}

JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES2Lib_onResize(JNIEnv* env, jobject obj,  jint width, jint height)
{
    slResize(svIndex, width, height);
}

JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES2Lib_onMenuButton(JNIEnv* env, jobject obj)
{  
   SL_LOG("onMenuButton");
   slCommand(svIndex, cmdMenu);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onMouseDown (JNIEnv* env, jobject obj, jint button, jint x, jint y)
{
   slMouseDown(svIndex, ButtonLeft, x, y, KeyNone);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onMouseUp(JNIEnv* env, jobject obj, jint button, jint x, jint y)
{
   slMouseUp(svIndex, ButtonLeft, x, y, KeyNone);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onMouseMove(JNIEnv* env, jobject obj, jint x, jint y)
{
   slMouseMove(svIndex, x, y);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onTouch2Down(JNIEnv* env, jobject obj, jint x1, jint y1, jint x2, jint y2)
{
   slTouch2Down(svIndex, x1, y1, x2, y2);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onTouch2Up(JNIEnv* env, jobject obj, jint x1, jint y1, jint x2, jint y2)
{
   slTouch2Up(svIndex, x1, y1, x2, y2);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onTouch2Move(JNIEnv* env, jobject obj, jint x1, jint y1, jint x2, jint y2)
{
   slTouch2Move(svIndex, x1, y1, x2, y2);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onDoubleClick(JNIEnv* env, jobject obj, jint button, jint x, jint y)
{
   slDoubleClick(svIndex, ButtonLeft, x, y, KeyNone);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onRotationPYR(JNIEnv* env, jobject obj, jfloat pitchRAD, jfloat yawRAD, jfloat rollRAD)
{
   slRotationPYR(svIndex, pitchRAD, yawRAD, rollRAD);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_onRotationQUAT(JNIEnv* env, jobject obj, jfloat quatX, jfloat quatY, jfloat quatZ, jfloat quatW)
{
   slRotationQUAT(svIndex, quatX, quatY, quatZ, quatW);
}

JNIEXPORT void JNICALL Java_ch_fhnw_comgr_GLES2Lib_onClose(JNIEnv* env, jobject obj)
{
   SL_LOG("onClose\n ");
   slTerminate();
   exit(0);
}

JNIEXPORT bool JNICALL Java_ch_fhnw_comgr_GLES2Lib_shouldClose(JNIEnv* env, jobject obj)
{
    return slShouldClose();
}

#endif // defined(SL_OS_ANDROID) && defined(SL_GUI_JAVA)