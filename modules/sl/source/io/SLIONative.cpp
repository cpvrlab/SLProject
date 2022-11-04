#include <SLIONative.h>

//-----------------------------------------------------------------------------
SLIOReaderNative::SLIOReaderNative(SLstring path)
  : _stream(path, std::ios::binary)
{
}
//-----------------------------------------------------------------------------
size_t SLIOReaderNative::read(void* buffer, size_t size)
{
    _stream.read((char*)buffer, (std::streamsize)size);
    return _stream.gcount();
}
//-----------------------------------------------------------------------------
size_t SLIOReaderNative::tell()
{
    return (size_t)_stream.tellg();
}
//-----------------------------------------------------------------------------
bool SLIOReaderNative::seek(size_t offset, Origin origin)
{
    std::ios::seekdir nativeOrigin = std::ios::beg;
    if (origin == IOO_beg)
        nativeOrigin = std::ios::beg;
    else if (origin == IOO_cur)
        nativeOrigin = std::ios::cur;
    else if (origin == IOO_end)
        nativeOrigin = std::ios::end;
    else
        SL_EXIT_MSG("Invalid seek origin");

    return (bool)_stream.seekg((std::streamsize)offset, nativeOrigin);
}
//-----------------------------------------------------------------------------
size_t SLIOReaderNative::size()
{
    std::streamsize pos = _stream.tellg();
    _stream.seekg(0, std::ios::end);
    size_t size = (size_t)_stream.tellg();
    _stream.seekg(pos, std::ios::beg);
    return size;
}
//-----------------------------------------------------------------------------
SLIOWriterNative::SLIOWriterNative(SLstring path)
  : _stream(path, std::ios::binary)
{
}
//-----------------------------------------------------------------------------
size_t SLIOWriterNative::write(const void* buffer, size_t size)
{
    size_t offsetBefore = (size_t)_stream.tellp();
    _stream.write(static_cast<const char*>(buffer), (std::streamsize)size);
    size_t offsetAfter = (size_t)_stream.tellp();
    return offsetAfter - offsetBefore;
}
//-----------------------------------------------------------------------------
size_t SLIOWriterNative::tell()
{
    return (size_t)_stream.tellp();
}
//-----------------------------------------------------------------------------
bool SLIOWriterNative::seek(size_t offset, Origin origin)
{
    std::ios::seekdir nativeOrigin = std::ios::beg;
    if (origin == IOO_beg)
        nativeOrigin = std::ios::beg;
    else if (origin == IOO_cur)
        nativeOrigin = std::ios::cur;
    else if (origin == IOO_end)
        nativeOrigin = std::ios::end;
    else
        SL_EXIT_MSG("Invalid seek origin");

    return (bool)_stream.seekp((std::streamsize)offset, nativeOrigin);
}
//-----------------------------------------------------------------------------
void SLIOWriterNative::flush()
{
    _stream.flush();
}
//-----------------------------------------------------------------------------
