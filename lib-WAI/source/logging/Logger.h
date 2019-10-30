/*!
 * \file
 * \brief Logging utilities for the SFV project
 *
 * Include the overall logging level once (best in WAIHelper.h):
 * #incude "logging/LogLevel{Debug, Info, Warn, Error}.h"
 *
 * We work with macros to have a 0 overhead for unused log statements
 * (unused log statements are not going to be evaluated)
 *
 * You have the following macros to work with:
 * WAI_DEBUG
 * WAI_INFO
 * WAI_WARN
 * WAI_ERROR
 *
 * There is a WAI_EXIT_MESSAGE. In addition to the logging
 * it will print the file and the line number. It is usefull prior to a
 * crash to gather more information (use it rarely).
 *
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The WAIHelper.h has to be included before the logger include guard!
//Otherwise preprocessor defines may not be setup in the right order¨!
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifndef WAI_LOG_LEVEL
#    include <WAIHelper.h>
#endif

#ifndef WAI_LOGGER
#    define WAI_LOGGER

#    include <string>
#    include <memory>
#    include <fstream>
#    include <cstdarg>
#    include <mutex>
#    include "FileLog.h"
#    include <WAIHelper.h>

#    ifndef WAI_DEBUG
#        define WAI_DEBUG(...) Logger::debug(__VA_ARGS__)
#        define WAI_DEBUG_TIMER_START(...) StaticTimer::start()
#        define WAI_DEBUG_TIMER_ELAPSED_MILLIS(...) Logger::debug("%s: %fms", __VA_ARGS__, StaticTimer::elapsedTimeInMilliSec());
#    endif

#    ifndef WAI_INFO
#        define WAI_INFO(...) Logger::info(__VA_ARGS__)
#        define WAI_INFO_TIMER_START(...) StaticTimer::start();
#        define WAI_INFO_TIMER_ELAPSED_MILLIS(...) Logger::info("%s: %fms", __VA_ARGS__, StaticTimer::elapsedTimeInMilliSec());
#    endif

#    ifndef WAI_WARN
#        define WAI_WARN(...) Logger::warn(__VA_ARGS__)
#    endif

#    ifndef WAI_ERROR
#        define WAI_ERROR(...) Logger::error(__VA_ARGS__)
#    endif

#    ifndef WAI_LOGGER_PIPE
//! the default logger pipe is standard output
#        define WAI_LOGGER_PIPE std::cout
#    endif

//! the exit message can used to log a message with line and file info prior to a crashing point.
#    define WAI_EXIT_MESSAGE(M) Logger::exitMessage(M, __FILE__, __LINE__);

class /*WAI_API*/ Log
{
    public:
    virtual void post(const std::string& message) = 0;
};

class /*WAI_API*/ Logger
{
    public:
    static void debug(const char* format, ...);
    static void info(const char* format, ...);
    static void warn(const char* format, ...);
    static void error(const char* format, ...);
    static void exitMessage(const std::string& message, const std::string& file, const int line);

    static std::unique_ptr<Log> _log;

    static void initFileLog(const std::string logDir, bool forceFlush);
    static void flushFileLog();

    private:
    static void print(const char* prefix, const char* format, va_list args);

    static std::unique_ptr<FileLog> _fileLog;
    static std::mutex               _loggerMutex;
};

#endif // end WAI_LOGGER
