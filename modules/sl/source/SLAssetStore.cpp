#include <SLAssetStore.h>
#include <Utils.h>
#include <iostream>

#ifdef SL_ASSET_STORE_REMOTE
    struct SLAssetDownload
    {
        SLAssetStore::DownloadCallback callback;
    };

    struct SLBundleDownload
    {
        int numAssets;
        int numAssetsDownloaded;
        SLAssetStore::DownloadCallback callback;
    };

    std::unordered_map<SLstring, SLstring> assets;
#endif

SLstring SLAssetStore::loadTextAsset(SLstring path)
{
#if defined(SL_ASSET_STORE_FS)
    return Utils::readTextFileIntoString("SLProject", path);
#elif defined(SL_ASSET_STORE_REMOTE)
    std::cout << "loading text asset '" << path << "'" << std::endl;
    
    if(!assets.count(path)) {
        std::cerr << "error: asset does not exist" << std::endl;
    }

    return assets[path];
#endif
}

void SLAssetStore::saveTextAsset(SLstring path, SLstring content) {
#if defined(SL_ASSET_STORE_FS)
    Utils::writeStringIntoTextFile("SLProject", content, path);
#elif defined(SL_ASSET_STORE_REMOTE)
    std::cout << "saving text asset '" << path << "'" << std::endl;
    assets.insert({path, content});
#endif
}

bool SLAssetStore::assetExists(SLstring path) {
#if defined(SL_ASSET_STORE_FS)
    return Utils::fileExists(dir);
#elif defined(SL_ASSET_STORE_REMOTE)
    return assets.count(path);
#endif
}

bool SLAssetStore::dirExists(SLstring dir) {
#if defined(SL_ASSET_STORE_FS)
    return Utils::dirExists(dir);
#elif defined(SL_ASSET_STORE_REMOTE)
    return true;
#endif
}

#ifdef SL_ASSET_STORE_REMOTE

void downloadSucceeded(emscripten_fetch_t* fetch)
{
    std::cout << "download done: " << fetch->url << std::endl;
    SLstring asset(fetch->data, fetch->data + fetch->numBytes);
    assets.insert({fetch->url, asset});
 
    SLAssetDownload* download = (SLAssetDownload*) fetch->userData;
    download->callback();
    delete download;

    emscripten_fetch_close(fetch);
}

void downloadFailed(emscripten_fetch_t* fetch)
{
    std::cout << "download error: " << fetch->url << std::endl;
    
    SLAssetDownload* download = (SLAssetDownload*) fetch->userData;
    download->callback();
    delete download;

    emscripten_fetch_close(fetch);
}

void SLAssetStore::downloadAsset(SLstring path, DownloadCallback callback) {
    std::cout << "downloading " << path << "..." << std::endl;

    SLAssetDownload* download = new SLAssetDownload;
    download->callback = callback;

    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = downloadSucceeded;
    attr.onerror = downloadFailed;
    attr.userData = (void*) download;
    emscripten_fetch_t* fetch = emscripten_fetch(&attr, path.c_str());
}

void SLAssetStore::downloadAssetBundle(SLVstring paths, DownloadCallback callback) {
    std::cout << "downloading asset bundle..." << std::endl;

    SLBundleDownload* download = new SLBundleDownload;
    download->numAssets = paths.size();
    download->callback = callback;

    for(SLstring path : paths) {
        downloadAsset(path, [download]()
        {
            download->numAssetsDownloaded++;
            if(download->numAssetsDownloaded == download->numAssets)
            {
                std::cout << "asset bundle downloaded." << std::endl;
                download->callback();
                delete download;    
            }
        });
    }
}
#endif