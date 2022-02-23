#ifndef UTILS_CUSTOMLOG_H
#define UTILS_CUSTOMLOG_H

#include <memory>
#include <string>

namespace Utils
{
//! Logger interface
class CustomLog
{
public:
    virtual void post(const std::string& message) = 0;
    virtual ~CustomLog() { ; }
};
}

#endif
