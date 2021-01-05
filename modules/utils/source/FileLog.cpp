#include "FileLog.h"

#include <iostream>
#include <sstream>
#include <time.h>
#include <Utils.h>

namespace Utils
{
FileLog::FileLog(std::string logDir, bool forceFlush)
  : _forceFlush(forceFlush)
{
    if (!Utils::dirExists(logDir))
        Utils::makeDir(logDir);

    time_t      now      = time(nullptr);
    std::string fileName = logDir + "/" + std::to_string(now) + "_log.txt";
    _logFile.open(fileName, std::ofstream::out);

    if (!_logFile.is_open())
    {
        std::string msg = "Could not create log file in dir: " + logDir;
        Utils::errorMsg("Utils", msg.c_str(), __LINE__, __FILE__);
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
    _logFile << message;
    if (_forceFlush)
        _logFile.flush();
}
};
