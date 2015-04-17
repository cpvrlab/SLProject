//#############################################################################
//  File:      SL/SL.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_H
#define SL_H

//-----------------------------------------------------------------------------
// Include standard C++ libraries
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <typeinfo>
#include <string>
#include <algorithm>
#include <map>
#include <chrono>
#include <thread>
#include <atomic>
#include <functional>
#include <random>

//-----------------------------------------------------------------------------
// Include standard C libraries
#include <stdio.h>               // for the old ANSI C IO functions
#include <stdlib.h>              // srand, rand
#include <float.h>               // for defines like FLT_MAX & DBL_MAX
#include <limits.h>              // for defines like UINT_MAX
#include <assert.h>              // for debug asserts
#include <time.h>                // for clock()
#include <sys/stat.h>            // for file info used in SLUtils
#include <math.h>                // for math functions
#include <string.h>              // for string functions


/////////////////////////////////////////////////////////
// Preprocessor constant defintions used in the SLProject
/////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
/* Determine one of the following operating systems:
SL_OS_MACOSX   :Apple Mac OSX
SL_OS_MACIOS   :Apple iOS
SL_OS_WINDOWS  :Microsoft desktop Windows XP, 7, 8, ...
SL_OS_ANDROID  :Google Android
SL_OS_LINUX    :Linux destop OS

With the OS definition the following constants are defined:
SL_GLES2: Supports only OpenGL ES2
SL_MEMLEAKDETECT: The memoryleak detector NVWA is used
SL_OVR: Support for Oculus Rift SDK
SL_USE_DISCARD_STEREOMODES: The discard stereo modes can be used (SLCamera)
*/

#ifdef __APPLE__
    #ifdef TARGET_OS_IOS
        #define SL_OS_MACIOS
        #define SL_GLES2
    #else
        #define SL_OS_MACOSX
        #define SL_OVR
        #if defined(_DEBUG)
            //#define SL_MEMLEAKDETECT  // nvwa doesn't work under OSX/clang
        #endif
    #endif
#elif defined(ANDROID) || defined(ANDROID_NDK)
    #define SL_OS_ANDROID
    #define SL_GLES2
#elif defined(_WIN32)
    #define SL_OS_WINDOWS
    #define SL_USE_DISCARD_STEREOMODES
    #define SL_OVR
    #ifdef _DEBUG
        #define SL_MEMLEAKDETECT
        //#define _NO_DEBUG_HEAP 1
    #endif
    #define STDCALL __stdcall
#elif defined(linux) || defined(__linux) || defined(__linux__)
    #define SL_OS_LINUX
    //#define SL_OVR
    #define SL_USE_DISCARD_STEREOMODES
    #ifdef _DEBUG
        //#define SL_MEMLEAKDETECT  // nvwa doesn't work under OSX/clang
    #endif
#else
    #error "SL has not been ported to this OS"
#endif

//-----------------------------------------------------------------------------
/* With one of the following constants the GUI system must be defined. This
has to be done in the project settings (pro files for QtCreator or in the
Visual Studio project settings):

SL_GUI_QT   :Qt on OSX, Windows, Linux or Android
SL_GUI_OBJC :ObjectiveC on iOS
SL_GUI_GLFW :GLFW on OSX, Windows or Linux
SL_GUI_JAVA :Java on Android (with the VS-Android project)
*/

//-----------------------------------------------------------------------------
#if defined(SL_OS_MACIOS)
    #include <sys/time.h>
    #import <OpenGLES/ES2/gl.h>
    #import <OpenGLES/ES2/glext.h>
    #include <zlib.h>
    #include <functional>
    #include <thread>
    #include <chrono>
    #include <random>
#elif defined(SL_OS_MACOSX)
    #include <sys/time.h>
    #include <functional>
    #include <thread>
    #include <chrono>
    #include <random>
    #include <GL/glew.h>
#elif defined(SL_OS_ANDROID)
    #include <sys/time.h>
    #include <android/log.h>
    #include <functional>
    #include <thread>
    #include <chrono>
    #include <random>
    #if defined(SL_GUI_QT)
        #include <QGLWidget>
    #elif defined(SL_GUI_JAVA)
        #include <jni.h>
        #include <GLES2/gl2.h>
        #include <GLES2/gl2ext.h>
    #else
        #error "This GUI system is not supported under Android"
    #endif
#elif defined(SL_OS_WINDOWS)
    #include <functional>
    #include <thread>
    #include <chrono>
    #include <random>
    #include <windows.h>
    #include <GL/glew.h>
#elif defined(SL_OS_LINUX)
    #include <sys/time.h>
    #include <functional>
    #include <thread>
    #include <chrono>
    #include <random>
    #include <GL/glew.h>
#else
    #error "SL has not been ported to this OS"
#endif

//-----------------------------------------------------------------------------
using namespace std;



//-----------------------------------------------------------------------------
// Determine compiler
#if defined(__GNUC__)
    #undef _MSC_VER
#endif

#if defined(_MSC_VER)
    #define SL_COMP_MSVC
    #define SL_STDCALL __stdcall
    #define SL_DEPRECATED __declspec(deprecated)
    #define _CRT_SECURE_NO_DEPRECATE // visual 8 secure crt warning
#elif defined(__BORLANDC__)
    #define SL_COMP_BORLANDC
    #define SL_STDCALL Stdcall
    #define SL_DEPRECATED   // @todo Does this compiler support deprecated attributes
#elif defined(__INTEL_COMPILER)
    #define SL_COMP_INTEL
    #define SL_STDCALL Stdcall
    #define SL_DEPRECATED   // @todo does this compiler support deprecated attributes
#elif defined(__GNUC__)
    #define SL_COMP_GNUC
    #define SL_STDCALL
    #define SL_DEPRECATED __attribute__((deprecated))
#else 
    #error "SL has not been ported to this compiler"
#endif

//-----------------------------------------------------------------------------
// Redefinition of standard types for platform independency
typedef std::string     SLstring;
#ifndef SL_OS_ANDROID
    typedef std::wstring    SLwstring;
#endif
typedef GLchar          SLchar;  // char is signed [-128 ... 127]!
typedef unsigned char   SLuchar;
typedef signed long     SLlong;
typedef unsigned long   SLulong;
typedef GLbyte          SLbyte;  // byte is signed [-128 ... 127]!
typedef GLubyte         SLubyte;
typedef GLshort         SLshort;
typedef GLushort        SLushort; 
typedef GLint           SLint;
typedef GLuint          SLuint;
typedef int64_t         SLint64;
typedef uint64_t        SLuint64;
typedef GLsizei         SLsizei;
typedef GLfloat         SLfloat;
#ifdef SL_HAS_DOUBLE
typedef GLdouble        SLdouble;
typedef GLfloat         SLreal;
#else
typedef GLfloat         SLreal;
#endif
typedef bool            SLbool; 
typedef GLenum          SLenum;
typedef GLbitfield      SLbitfield;
typedef GLfloat         SLfloat;

// all std::vectors begin with SLV*
typedef std::vector<SLbyte>   SLVbyte;
typedef std::vector<SLubyte>  SLVubyte;
typedef std::vector<SLchar>   SLVchar;
typedef std::vector<SLuchar>  SLVuchar;
typedef std::vector<SLshort>  SLVshort;
typedef std::vector<SLushort> SLVushort;
typedef std::vector<SLint>    SLVint;
typedef std::vector<SLuint>   SLVuint;
typedef std::vector<SLlong>   SLVlong;
typedef std::vector<SLulong>  SLVulong;
typedef std::vector<SLfloat>  SLVfloat;
typedef std::vector<SLstring> SLVstring;

//-----------------------------------------------------------------------------
// Bit manipulation makros for ones that forget it always
#define SL_GETBIT(VAR, BITVAL) VAR&BITVAL
#define SL_SETBIT(VAR, BITVAL) VAR|=BITVAL
#define SL_DELBIT(VAR, BITVAL) VAR&=~BITVAL
#define SL_TOGBIT(VAR, BITVAL) if (VAR&BITVAL) VAR &=~BITVAL; else VAR|=BITVAL

//-----------------------------------------------------------------------------
// Prevention for warnings in XCode
#define UNUSED_PARAMETER(r)  ((void)(x))

//-----------------------------------------------------------------------------
// Some debuging and error handling functions and macros 
#define SL_EXIT_MSG(M)  SL::exitMsg((M), __LINE__, __FILE__)
#define SL_WARN_MSG(M)  SL::warnMsg((M), __LINE__, __FILE__)
#if defined(SL_OS_ANDROID)
#define SL_LOG(...)  __android_log_print(ANDROID_LOG_INFO, "SLProject", __VA_ARGS__) 
#else
#define SL_LOG(...)  printf(__VA_ARGS__);
#endif

//-----------------------------------------------------------------------------
//! Class SL with static error message and exit functions.
class SL
{
    public:
    //! SL::Exit terminates the application with a message. No leak cheching.
    static void exitMsg(const SLchar* msg, const SLint line, const SLchar* file);
   
    //! SL::Warn message output
    static void warnMsg(const SLchar* msg, const SLint line, const SLchar* file);
};
//-----------------------------------------------------------------------------
#endif
