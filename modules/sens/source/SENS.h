#ifndef SENS_H
#define SENS_H

#include <chrono>
#include <Utils.h>

#ifdef __APPLE__
#    include <TargetConditionals.h>
#    if TARGET_OS_IOS
#        define SENS_OS_MACIOS
#        define SENS_GLES
#        define SENS_GLES3
#    else
#        define SENS_OS_MACOS
#        if defined(_DEBUG)

#        endif
#    endif
#elif defined(ANDROID) || defined(ANDROID_NDK)
#    define SENS_OS_ANDROID
#    define SENS_GLES
#    define SENS_GLES3
#elif defined(_WIN32)
#    define SENS_OS_WINDOWS
#    define SENS_USE_DISCARD_STEREOMODES
#    ifdef _DEBUG
#        define _GLDEBUG
#    endif
#    define STDCALL __stdcall
#elif defined(linux) || defined(__linux) || defined(__linux__)
#    define SENS_OS_LINUX
#    define SENS_USE_DISCARD_STEREOMODES
#    ifdef _DEBUG
#    endif
#else
#    error "SL has not been ported to this OS"
#endif

enum class SENSType
{
    CAM,
    VIDEO,
    GPS,
    ORIENTATION
};

using SENSClock        = std::chrono::high_resolution_clock;
using SENSTimePt       = std::chrono::high_resolution_clock::time_point;
using SENSMicroseconds = std::chrono::microseconds;

#define SENS_DEBUG(...) Utils::log("SENS DEBUG", __VA_ARGS__);
//#define SENS_DEBUG
//#define SENS_INFO(...) Utils::log("SENS INFO", __VA_ARGS__);
#define SENS_INFO
#define SENS_WARN(...) Utils::log("SENS WARN", __VA_ARGS__);
//#define SENS_WARN

#endif //SENS_H
