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

namespace SLAssetStore
{
    SLstring loadTextAsset(SLstring path);
    void saveTextAsset(SLstring path, SLstring content);
    bool assetExists(SLstring path);
    bool dirExists(SLstring dir);

#ifdef SL_ASSET_STORE_REMOTE
    typedef std::function<void()> DownloadCallback;

    void downloadAsset(SLstring path, DownloadCallback callback);
    void downloadAssetBundle(SLVstring paths, DownloadCallback callback);
#endif
}

#endif