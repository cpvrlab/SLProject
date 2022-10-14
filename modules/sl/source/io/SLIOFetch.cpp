#include <SLIOFetch.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
#include <iostream>
#include <emscripten/threading.h>
#include <emscripten/fetch.h>
//-----------------------------------------------------------------------------
extern "C" {
    extern void showLoadingOverlay(const char* resource);
    extern void hideLoadingOverlay();
};
//-----------------------------------------------------------------------------
SLFetchResult SLIOReaderFetch::fetch(SLstring url)
{
    emscripten_async_run_in_main_runtime_thread(EM_FUNC_SIG_VI,
                                                showLoadingOverlay,
                                                url.c_str());

    std::cout << "FETCH \"" << url << "\"" << std::endl;

    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    std::strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY | EMSCRIPTEN_FETCH_SYNCHRONOUS;
    emscripten_fetch_t* fetch = emscripten_fetch(&attr, url.c_str());

    std::cout << "STATUS: " << fetch->status << ", SIZE: " << fetch->totalBytes << std::endl;

    unsigned char* data = new unsigned char[fetch->totalBytes];
    std::memcpy(data, fetch->data, fetch->totalBytes);
    emscripten_fetch_close(fetch);

    emscripten_async_run_in_main_runtime_thread(EM_FUNC_SIG_V,
                                                hideLoadingOverlay);

    SLIOBuffer buffer{data, (size_t)fetch->totalBytes};
    return SLFetchResult{fetch->status, buffer};
}
//-----------------------------------------------------------------------------
bool SLIOReaderFetch::exists(SLstring url)
{
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    std::strcpy(attr.requestMethod, "HEAD");
    attr.attributes = EMSCRIPTEN_FETCH_SYNCHRONOUS;
    emscripten_fetch_t* fetch = emscripten_fetch(&attr, url.c_str());
    emscripten_fetch_close(fetch);
    return fetch->status == 200;
}
//-----------------------------------------------------------------------------
SLIOReaderFetch::SLIOReaderFetch(SLstring url)
  : _position(0)
{
    _buffer = fetch(url).buffer;
}
//-----------------------------------------------------------------------------
SLIOReaderFetch::~SLIOReaderFetch()
{
    std::free(_buffer.data);
}
//-----------------------------------------------------------------------------
size_t SLIOReaderFetch::read(void* buffer, size_t size)
{
    if (_position + size <= _buffer.size)
    {
        std::memcpy(buffer, _buffer.data + _position, size);
        _position += size;
        return size;
    }
    else
    {
        size_t sizeToRead = _buffer.size - _position;
        std::memcpy(buffer, _buffer.data + _position, sizeToRead);
        _position += sizeToRead;
        return sizeToRead;
    }
}
//-----------------------------------------------------------------------------
size_t SLIOReaderFetch::tell()
{
    return _position;
}
//-----------------------------------------------------------------------------
bool SLIOReaderFetch::seek(size_t offset, Origin origin)
{
    size_t previousPos = _position;

    switch (origin)
    {
        case IOO_beg: _position = offset; break;
        case IOO_cur: _position += offset; break;
        case IOO_end: _position = _buffer.size - 1 - offset; break;
    }

    bool ok = _position >= 0 && _position < _buffer.size;
    if (!ok)
        _position = previousPos;

    return ok;
}
//-----------------------------------------------------------------------------
size_t SLIOReaderFetch::size()
{
    return _buffer.size;
}
//-----------------------------------------------------------------------------
#endif