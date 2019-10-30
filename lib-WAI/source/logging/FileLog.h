#ifndef WAI_FILE_LOG_H
#define WAI_FILE_LOG_H

#include <string>
#include <fstream>

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

#endif // WAI_FILE_LOG_H
