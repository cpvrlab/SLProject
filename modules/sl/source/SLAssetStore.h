#ifndef SLASSETSTORE_H
#define SLASSETSTORE_H

#include <SL.h>
#include <unordered_map>
#include <functional>

#ifdef SL_EMSCRIPTEN
#    define SL_ASSET_STORE_REMOTE
#    include <emscripten/fetch.h>
#else
#    define SL_ASSET_STORE_FS
#endif

struct SLAsset
{
public:
    char* data;
    size_t size;

public:
    SLAsset()
    {
        data = nullptr;
        size = 0;
    }

    SLAsset(const char* data, size_t size)
    {
        if(!data) return;

        this->data = new char[size];
        std::memcpy(this->data, data, size);
        this->size = size;
    }

    SLAsset(const SLAsset& other) : SLAsset(other.data, other.size) {}
    
    SLAsset(SLAsset&& other) noexcept : data(std::exchange(other.data, nullptr)), size(other.size) {}
    
    ~SLAsset() { delete[] data; }
    
    SLAsset& operator=(const SLAsset& other) { return *this = SLAsset(other); }
    
    SLAsset& operator=(SLAsset&& other)
    {
        std::swap(data, other.data);
        size = other.size;
        return *this;
    }

    void drop()
    {
        data = nullptr;
        size = 0;
    }

};

#ifdef SL_ASSET_STORE_REMOTE
typedef std::function<void()> DownloadCallback;

struct SLAssetDownload
{
    DownloadCallback callback;
};

struct SLBundleDownload
{
    int numAssets;
    int numAssetsDownloaded;
    DownloadCallback callback;
};
#endif

namespace SLAssetStore
{
    SLAsset loadAsset(SLstring path);
    SLstring loadTextAsset(SLstring path);
    void saveTextAsset(SLstring path, SLstring content);
    bool assetExists(SLstring path);
    bool dirExists(SLstring dir);

#ifdef SL_ASSET_STORE_REMOTE
    void downloadAsset(SLstring path, DownloadCallback callback);
    void downloadAssetBundle(SLVstring paths, DownloadCallback callback);
#endif
}

#endif