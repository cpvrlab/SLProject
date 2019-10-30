#include "FileLog.h"

#include <iostream>
#include <sstream>
#include <time.h>
#include <Utils.h>

FileLog::FileLog(std::string logDir, bool forceFlush)
  : _forceFlush(forceFlush)
{
    try
    {
        if (!Utils::dirExists(logDir))
            Utils::makeDir(logDir);

        time_t      now      = time(nullptr);
        std::string fileName = logDir + "/" + std::to_string(now) + "_log.txt";
        _logFile.open(fileName, std::ofstream::out);

        if (!_logFile.is_open())
            throw std::runtime_error("Could not open log file!");

        //As long as the logger is not instantiated one can not log.
        //But if this fails, the information will be logged which is going to fail.
        //So no SFV logging and exception handling here!
    }
    catch (...)
    {
        std::cout << "[Error] Could not open log file!" << std::endl;
    }
}

FileLog::~FileLog()
{
    _logFile.flush();
    _logFile.close();
}

void FileLog::flush()
{
    _logFile.flush();
}

void FileLog::post(const std::string& message)
{
    _logFile << message + std::string("\n");
    if (_forceFlush)
        _logFile.flush();
}
