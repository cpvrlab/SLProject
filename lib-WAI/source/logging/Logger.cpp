#include "Logger.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <time.h>

#define MESSAGE_SIZE 4096

//the resource paths have to be setup before the FileLog can be correctly instantiated (for android)
std::unique_ptr<FileLog> Logger::_fileLog;
std::mutex               Logger::_loggerMutex;

class DefaultLog : public Log
{
    virtual void post(const std::string& message)
    {
        WAI_LOGGER_PIPE << message << std::endl;
    }
};

//instantiation of DefaultLog
std::unique_ptr<Log> Logger::_log = std::make_unique<DefaultLog>();

void Logger::print(const char* prefix, const char* format, va_list args)
{
    std::unique_lock<std::mutex> lock(_loggerMutex);
    char                         message[MESSAGE_SIZE];
    vsnprintf(message, MESSAGE_SIZE, format, args);
    std::stringstream ss;
    ss << "[" << prefix << "]" << message;
    std::string line = ss.str();
    Logger::_log->post(line);
    if (Logger::_fileLog)
        Logger::_fileLog->post(line);
}

void Logger::debug(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    Logger::print("DEBUG", format, args);
    va_end(args);
}

void Logger::info(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    Logger::print(" INFO", format, args);
    va_end(args);
}

void Logger::warn(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    Logger::print(" WARN", format, args);
    va_end(args);
}

void Logger::error(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    Logger::print("ERROR", format, args);
    va_end(args);
}

void Logger::exitMessage(const std::string& message, const std::string& file, const int line)
{
    WAI_LOGGER_PIPE << "[ EXIT] on " << file << "(" << line << "):" << std::endl;
    WAI_LOGGER_PIPE << message << std::endl;
}

void Logger::initFileLog(const std::string logDir, bool forceFlush)
{
    Logger::_fileLog = std::make_unique<FileLog>(logDir, forceFlush);
}

void Logger::flushFileLog()
{
    if (Logger::_fileLog)
        Logger::_fileLog->flush();
}
