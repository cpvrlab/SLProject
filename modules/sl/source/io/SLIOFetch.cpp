#include <SLIOFetch.h>
#include <iostream>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
// clang-format off
EM_ASYNC_JS(void, jsFetch, (const char* urlPointer,
                            int urlLength,
                            int* outStatus,
                            unsigned char** outData,
                            int* outLength), {
    document.querySelector("#overlay").classList.add("visible");

    let url = UTF8ToString(urlPointer, urlLength);
    document.querySelector("#download-text").innerHTML = url;

    let response = await fetch(url);
    setValue(outStatus, response.status, "i32");

    if (response.status == 200) {
        let data = await response.arrayBuffer();

        let typedArray = new Uint8Array(data);
        let buffer = _malloc(data.byteLength);
        writeArrayToMemory(typedArray, buffer);

        setValue(outData, buffer, "u8*");
        setValue(outLength, data.byteLength, "i32");
    } else {
        setValue(outData, 0, "u8*");
        setValue(outLength, 0, "i32");
    }

    document.querySelector("#overlay").classList.remove("visible");
})
// clang-format on
//-----------------------------------------------------------------------------
// clang-format off
EM_ASYNC_JS(bool, jsFileExists, (const char* urlPointer, int urlLength), {
    let url = UTF8ToString(urlPointer, urlLength);
    let response = await fetch(url, {method: 'HEAD'});
    return response.status === 200;
})
// clang-format on
//-----------------------------------------------------------------------------
SLFetchResult SLIOReaderFetch::fetch(SLstring url)
{
    std::cout << "FETCH \"" << url << "\"" << std::endl;

    int            status;
    unsigned char* data;
    int            size;
    jsFetch(url.c_str(), (int)url.length(), &status, &data, &size);

    std::cout << "STATUS: " << status << ", SIZE: " << size << std::endl;

    SLIOBuffer    buffer{data, (size_t)size};
    SLFetchResult result{status, buffer};
    return result;
}
//-----------------------------------------------------------------------------
bool SLIOReaderFetch::exists(SLstring url)
{
    return true;
//    return jsFileExists(url.c_str(), (int)url.size());
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