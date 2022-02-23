#ifndef HTTPDOWNLOADER
#define HTTPDOWNLOADER
#include <string>
class HttpDownloader
{
public:
    HttpDownloader(){};
    virtual void download(std::string url, std::string dst){};
};

#endif
