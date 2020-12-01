#ifndef HTTPNDKDOWNLOADER
#define HTTPNDKDOWNLOADER

#include <jni.h>
#include <HttpDownloader.h>

class HttpNdkDownloader : public HttpDownloader
{
public:
    JavaVM* _vm;
    jclass _clazz;
    jobject _object = nullptr;
    HttpNdkDownloader(JavaVM* vm, jclass clazz);
    virtual void download(std::string url, std::string dst);
    virtual void download(std::string url, std::string dst, std::string user, std::string pwd);
private:
    bool getEnv(JNIEnv** env);
};

#endif
