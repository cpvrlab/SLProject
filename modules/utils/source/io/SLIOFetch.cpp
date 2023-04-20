//#############################################################################
//  File:      SLIOFetch.cpp
//  Date:      October 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLIOFetch.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
#    include <emscripten/fetch.h>
#    include <iostream>
//-----------------------------------------------------------------------------
bool SLIOReaderFetch::exists(std::string url)
{
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    std::strcpy(attr.requestMethod, "HEAD");
    attr.attributes            = EMSCRIPTEN_FETCH_SYNCHRONOUS;
    emscripten_fetch_t* fetch  = emscripten_fetch(&attr, url.c_str());
    bool                exists = fetch->status == 200;
    emscripten_fetch_close(fetch);
    return exists;
}
//-----------------------------------------------------------------------------
SLIOReaderFetch::SLIOReaderFetch(std::string url)
  : SLIOReaderMemory(url)
{
    Utils::showSpinnerMsg(url);

    std::cout << "FETCH \"" << url << "\"" << std::endl;

    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    std::strcpy(attr.requestMethod, "GET");
    attr.attributes           = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY | EMSCRIPTEN_FETCH_SYNCHRONOUS;
    emscripten_fetch_t* fetch = emscripten_fetch(&attr, url.c_str());

    int    status = fetch->status;
    size_t size   = (size_t)fetch->totalBytes;
    std::cout << "STATUS: " << status << ", SIZE: " << size << std::endl;

    if (status == 200)
    {
        SLIOMemory::set(_path, std::vector<char>(fetch->data, fetch->data + size));
    }

    emscripten_fetch_close(fetch);
    Utils::hideSpinnerMsg();
}
//-----------------------------------------------------------------------------
SLIOReaderFetch::~SLIOReaderFetch()
{
    SLIOMemory::clear(_path);
}
//-----------------------------------------------------------------------------
#endif