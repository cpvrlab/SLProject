#include <HttpNdkDownloader.h>
#include <Utils.h>

HttpNdkDownloader::HttpNdkDownloader(JavaVM* vm, jclass clazz)
: _vm(vm), _clazz(clazz)
{

}

void HttpNdkDownloader::download(std::string url, std::string dst)
{
    JNIEnv* env;
    bool threadAttached = getEnv(&env);
    if (env == nullptr)
        return;

    jmethodID methodId = env->GetStaticMethodID(_clazz,
                                                "DownloadFiles",
                                                "(Ljava/lang/String;Ljava/lang/String;)V");

    jstring jurl = env->NewStringUTF(url.c_str());
    jstring jdst = env->NewStringUTF(dst.c_str());
    env->CallStaticVoidMethod(_clazz, methodId, jurl, jdst);
    if (threadAttached)
    {
        _vm->DetachCurrentThread();
    }
}

void HttpNdkDownloader::download(std::string url, std::string dst, std::string user, std::string pwd)
{
    JNIEnv* env;
    bool threadAttached = getEnv(&env);
    if (env == nullptr)
        return;
    jmethodID methodId = env->GetStaticMethodID(_clazz,
                                                "DownloadFiles",
                                                "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V");

    jstring jurl = env->NewStringUTF(url.c_str());
    jstring jdst = env->NewStringUTF(dst.c_str());
    jstring juser = env->NewStringUTF(user.c_str());
    jstring jpwd = env->NewStringUTF(pwd.c_str());
    env->CallStaticVoidMethod(_clazz, methodId, jurl, jdst, juser, jpwd);
    if (threadAttached)
    {
        _vm->DetachCurrentThread();
    }
}

bool HttpNdkDownloader::getEnv(JNIEnv** env)
{
    bool threadAttached = false;
    switch (_vm->GetEnv((void**)env, JNI_VERSION_1_6))
    {
        case JNI_OK:
        {
        }
        break;
        case JNI_EDETACHED:
        {
            jint result = _vm->AttachCurrentThread(env, nullptr);
            if (result == JNI_ERR)
                *env = nullptr;
            threadAttached = true;
        }
        break;
        case JNI_EVERSION:
            *env = nullptr;
    }
    return threadAttached;
}
