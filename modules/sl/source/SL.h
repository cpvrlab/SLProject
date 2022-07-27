//#############################################################################
//  File:      sl/SL.h
//  Date:      October 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_H
#define SL_H

//////////////////////////////////////////////////////////
// Preprocessor constant definitions used in the SLProject
//////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
/* Determine one of the following operating systems:
SL_OS_MACOS    :Apple Mac OSX
SL_OS_MACIOS   :Apple iOS
SL_OS_WINDOWS  :Microsoft desktop Windows XP, 7, 8, ...
SL_OS_ANDROID  :Goggle Android
SL_OS_LINUX    :Linux desktop OS

With the OS definition the following constants are defined:
SL_GLES : Any version of OpenGL ES
SL_GLES3: Supports only OpenGL ES3
SL_USE_DISCARD_STEREOMODES: The discard stereo modes can be used (SLCamera)
*/

#ifdef __APPLE__
#    include <TargetConditionals.h>
#    if TARGET_OS_IOS
#        define SL_OS_MACIOS
#        define SL_GLES
#        define SL_GLES3
#    else
#        define SL_OS_MACOS
#        if defined(_DEBUG)

#        endif
#    endif
#elif defined(ANDROID) || defined(ANDROID_NDK)
#    define SL_OS_ANDROID
#    define SL_GLES
#    define SL_GLES3
#elif defined(_WIN32)
#    define SL_OS_WINDOWS
#    define SL_USE_DISCARD_STEREOMODES
#    ifdef _DEBUG
#        define _GLDEBUG
#    endif
#    define STDCALL __stdcall
#elif defined(linux) || defined(__linux) || defined(__linux__)
#    define SL_OS_LINUX
#    define SL_USE_DISCARD_STEREOMODES
#    ifdef _DEBUG
#    endif
#elif defined(__EMSCRIPTEN__)
#    define SL_EMSCRIPTEN
#    define SL_USE_DISCARD_STEREOMODES
#    define SL_GLES
#    define SL_GLES3
#else
#    error "SL has not been ported to this OS"
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
#    include <chrono>
#    include <functional>
#    include <random>
#    include <sys/time.h>
#    include <thread>
#    include <CoreServices/CoreServices.h> // for system info
#    include <zlib.h>
#elif defined(SL_OS_MACOS)
#    include <chrono>
#    include <functional>
#    include <random>
#    include <ctime>
#    include <thread>
//#    include <CoreServices/CoreServices.h> // for system info
#    include <sys/sysctl.h> // for system info
#elif defined(SL_OS_ANDROID)
#    include <sys/time.h>
#    include <sys/system_properties.h>
#    include <chrono>
#    include <functional>
#    include <random>
#    include <sstream>
#    include <thread>
#elif defined(SL_OS_WINDOWS)
#    include <chrono>
#    include <functional>
#    include <random>
#    include <thread>
#    include <windows.h>
#elif defined(SL_OS_LINUX)
#    include <chrono>
#    include <functional>
#    include <random>
#    include <sstream>
#    include <sys/time.h>
#    include <thread>
#elif defined(SL_EMSCRIPTEN)
#    include <random>
#else
#    error "SL has not been ported to this OS"
#endif

#include <Utils.h>

//-----------------------------------------------------------------------------
using std::string;
using std::vector;

//-----------------------------------------------------------------------------
// Determine compiler
#if defined(__GNUC__)
#    undef _MSC_VER
#endif

#if defined(_MSC_VER)
#    define SL_COMP_MSVC
#    define SL_STDCALL __stdcall
#    define SL_DEPRECATED __declspec(deprecated)
#    define _CRT_SECURE_NO_DEPRECATE // visual 8 secure crt warning
#elif defined(__BORLANDC__)
#    define SL_COMP_BORLANDC
#    define SL_STDCALL Stdcall
#    define SL_DEPRECATED // @todo Does this compiler support deprecated attributes
#elif defined(__INTEL_COMPILER)
#    define SL_COMP_INTEL
#    define SL_STDCALL Stdcall
#    define SL_DEPRECATED // @todo does this compiler support deprecated attributes
#elif defined(__GNUC__)
#    define SL_COMP_GNUC
#    define SL_STDCALL
#    define SL_DEPRECATED __attribute__((deprecated))
#else
#    error "SL has not been ported to this compiler"
#endif

//-----------------------------------------------------------------------------
// Redefinition of standard types for platform independence
typedef string SLstring;
#ifndef SL_OS_ANDROID
typedef std::wstring SLwstring;
#endif
typedef char           SLchar;     // analog to GLchar (char is signed [-128 ... 127]!)
typedef unsigned char  SLuchar;    // analog to GLuchar
typedef signed long    SLlong;     // analog to GLlong
typedef unsigned long  SLulong;    // analog to GLulong
typedef signed char    SLbyte;     // analog to GLbyte
typedef unsigned char  SLubyte;    // analog to GLubyte
typedef short          SLshort;    // analog to GLshort
typedef unsigned short SLushort;   // analog to GLushort
typedef int            SLint;      // analog to GLint
typedef unsigned int   SLuint;     // analog to GLuint
typedef int            SLsizei;    // analog to GLsizei
typedef float          SLfloat;    // analog to GLfloat
typedef double         SLdouble;   // analog to GLdouble
typedef bool           SLbool;     // analog to GLbool
typedef unsigned int   SLenum;     // analog to GLenum
typedef unsigned int   SLbitfield; // analog to GLbitfield
typedef int64_t        SLint64;
typedef uint64_t       SLuint64;

// All 1D vectors begin with SLV*
typedef vector<SLbool>   SLVbool;
typedef vector<SLbyte>   SLVbyte;
typedef vector<SLubyte>  SLVubyte;
typedef vector<SLchar>   SLVchar;
typedef vector<SLuchar>  SLVuchar;
typedef vector<SLshort>  SLVshort;
typedef vector<SLushort> SLVushort;
typedef vector<SLint>    SLVint;
typedef vector<SLuint>   SLVuint;
typedef vector<SLlong>   SLVlong;
typedef vector<SLulong>  SLVulong;
typedef vector<SLfloat>  SLVfloat;
typedef vector<SLstring> SLVstring;
typedef vector<size_t>   SLVsize_t;

// All 2D vectors begin with SLVV*
typedef vector<vector<SLfloat>>  SLVVfloat;
typedef vector<vector<SLuchar>>  SLVVuchar;
typedef vector<vector<SLushort>> SLVVushort;
typedef vector<vector<SLuint>>   SLVVuint;
typedef vector<vector<SLchar>>   SLVVchar;
typedef vector<vector<SLshort>>  SLVVshort;
typedef vector<vector<SLint>>    SLVVint;

//-----------------------------------------------------------------------------
// Shortcut for size of a vector
template<class T>
inline SLuint
SL_sizeOfVector(const T& vector)
{
    return (SLint)(vector.capacity() * sizeof(typename T::value_type));
}
//-----------------------------------------------------------------------------
// Bit manipulation macros for ones that forget it always
#define SL_GETBIT(VAR, VAL) VAR& VAL
#define SL_SETBIT(VAR, VAL) VAR |= VAL
#define SL_DELBIT(VAR, VAL) VAR &= ~VAL
#define SL_TOGBIT(VAR, VAL) \
    if (VAR & VAL) \
        VAR &= ~VAL; \
    else \
        VAR |= VAL
//-----------------------------------------------------------------------------
// Prevention for warnings in XCode
#define UNUSED_PARAMETER(r) ((void)(x))

//-----------------------------------------------------------------------------
// Some debugging and error handling macros
#define SL_LOG(...) Utils::log("SLProject", __VA_ARGS__)
#define SL_EXIT_MSG(message) Utils::log("SLProject Error", (message))
#define SL_WARN_MSG(message) Utils::warnMsg("SLProject", (message), __LINE__, __FILE__)
//-----------------------------------------------------------------------------
#endif
