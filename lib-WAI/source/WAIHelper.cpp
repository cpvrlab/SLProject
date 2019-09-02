#include "WAIHelper.h"

#ifdef WAI_OS_ANDROID
#    include <android/log.h>
#endif

DebugLogCallback debugLogCallback;

void registerDebugCallback(DebugLogCallback callback)
{
    if (callback)
    {
        debugLogCallback = callback;
        debugLogCallback("callback registered");
    }
}

#define MAX_LOG_LENGTH 9999
void WAI_LOG(const char* format, ...)
{
    char    buffer[MAX_LOG_LENGTH];
    va_list args;
    va_start(args, format);
    int ret = vsnprintf(buffer, MAX_LOG_LENGTH, format, args);
    va_end(args);

    if (!debugLogCallback)
    {
#ifdef __APPLE__
        printf("%s\n", buffer);
#elif defined(WAI_OS_ANDROID)
        __android_log_print(ANDROID_LOG_INFO, "lib-WAI", "%s\n", buffer);
#elif defined(WAI_OS_WINDOWS)
        printf("%s\n", buffer);
#elif defined(WAI_OS_LINUX)
        printf("%s\n", buffer);
#endif
    }
    else
    {
        debugLogCallback(buffer);
    }
}
