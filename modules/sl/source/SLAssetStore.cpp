#include <SLAssetStore.h>
#include <Utils.h>
#include <emscripten.h>
#include <iostream>

#include <png.h>
#include <cstdio>

#ifdef SL_ASSET_STORE_REMOTE
#    include <SDL_image.h>
std::unordered_map<SLstring, SLAsset> assets;
#endif

SLAsset SLAssetStore::loadAsset(SLstring path)
{
#if defined(SL_ASSET_STORE_FS)
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    size_t        size = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);
    char* data = new char[size];
    file.read(data, size);
    return SLAsset{data, size};
#elif defined(SL_ASSET_STORE_REMOTE)
    std::cout << "loading binary asset '" << path << "'" << std::endl;
    return assets[path];
#endif
}

SLstring SLAssetStore::loadTextAsset(SLstring path)
{
#if defined(SL_ASSET_STORE_FS)
    return Utils::readTextFileIntoString("SLProject", path);
#elif defined(SL_ASSET_STORE_REMOTE)
    std::cout << "loading text asset '" << path << "'" << std::endl;
    if (assets.count(path))
    {
        std::cout << "loading from memory..." << std::endl;
        SLAsset& asset = assets[path];
        return SLstring(asset.data, asset.data + asset.size);
    }
    else
    {
        std::cout << "loading from file system..." << std::endl;
        return Utils::readTextFileIntoString("SLProject", path);
    }
#endif
}

CVMat SLAssetStore::loadCVImageAsset(SLstring path)
{
#if defined(SL_ASSET_STORE_FS)
    return cv::imread(path, -1);
#elif defined(SL_ASSET_STORE_REMOTE)
    if (Utils::endsWithString(path, ".png"))
        return loadPNG(path);
    else
    {
        std::cout << "loading image using Emscripten \"" << path << "\"" << std::endl;
        int   w, h;
        char* pixels = emscripten_get_preloaded_image_data(path.c_str(), &w, &h);
        CVMat image(w, h, CV_8UC4, pixels);
        return image;
    }
#endif
}

void SLAssetStore::saveTextAsset(SLstring path, SLstring content)
{
#if defined(SL_ASSET_STORE_FS)
    Utils::writeStringIntoTextFile("SLProject", content, path);
#elif defined(SL_ASSET_STORE_REMOTE)
    std::cout << "saving text asset '" << path << "'" << std::endl;
    SLAsset asset(content.c_str(), content.size());
    assets.insert({path, asset});
#endif
}

bool SLAssetStore::assetExists(SLstring path)
{
#if defined(SL_ASSET_STORE_FS)
    return Utils::fileExists(path);
#elif defined(SL_ASSET_STORE_REMOTE)
    return assets.count(path) || Utils::fileExists(path);
#endif
}

bool SLAssetStore::dirExists(SLstring dir)
{
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
    SLAsset asset(fetch->data, fetch->numBytes);
    assets.insert({fetch->url, asset});

    SLAssetDownload* download = (SLAssetDownload*)fetch->userData;
    download->callback();
    delete download;

    emscripten_fetch_close(fetch);
}

void downloadFailed(emscripten_fetch_t* fetch)
{
    std::cout << "download error: " << fetch->url << std::endl;

    SLAssetDownload* download = (SLAssetDownload*)fetch->userData;
    download->callback();
    delete download;

    emscripten_fetch_close(fetch);
}

void SLAssetStore::downloadAsset(SLstring path, DownloadCallback callback)
{
    std::cout << "downloading " << path << "..." << std::endl;

    SLAssetDownload* download = new SLAssetDownload;
    download->callback        = callback;

    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    strcpy(attr.requestMethod, "GET");
    attr.attributes           = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess            = downloadSucceeded;
    attr.onerror              = downloadFailed;
    attr.userData             = (void*)download;
    emscripten_fetch_t* fetch = emscripten_fetch(&attr, path.c_str());
}

void SLAssetStore::downloadAssetBundle(SLVstring paths, DownloadCallback callback)
{
    std::cout << "downloading asset bundle..." << std::endl;

    SLBundleDownload* download = new SLBundleDownload;
    download->numAssets        = paths.size();
    download->callback         = callback;

    for (SLstring path : paths)
    {
        downloadAsset(path, [download]()
                      {
            download->numAssetsDownloaded++;
            if(download->numAssetsDownloaded == download->numAssets)
            {
                std::cout << "asset bundle downloaded." << std::endl;
                download->callback();
                delete download;    
            } });
    }
}
#endif