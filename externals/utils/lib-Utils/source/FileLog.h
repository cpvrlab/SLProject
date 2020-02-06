#ifndef CPLVRLAB_FILE_LOG_H
#define CPLVRLAB_FILE_LOG_H

#include <string>
#include <fstream>

namespace Utils
{
class FileLog
{
public:
    FileLog(std::string logDir, bool forceFlush);
    virtual ~FileLog();
    void flush();
    void post(const std::string& message);

private:
    std::ofstream _logFile;
    bool          _forceFlush;
};
};

#endif // CPLVRLAB_FILE_LOG_H
