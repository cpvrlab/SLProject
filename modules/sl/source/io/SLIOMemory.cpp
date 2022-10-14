#include <SLIOMemory.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
#include <unordered_map>
#include <vector>
//-----------------------------------------------------------------------------
std::unordered_map<SLstring, std::vector<char>> memoryFiles;
//-----------------------------------------------------------------------------
bool SLIOMemory::exists(SLstring path)
{
    return memoryFiles.count(path);
}
//-----------------------------------------------------------------------------
SLIOReaderMemory::SLIOReaderMemory(SLstring path)
{
    SL_LOG("READING FROM MEMORY");
}
//-----------------------------------------------------------------------------
size_t SLIOReaderMemory::read(void* buffer, size_t size)
{
    std::vector<char>& memoryFile = memoryFiles[_path];
    std::memcpy(buffer, memoryFile.data(), size);
    return size;
}
//-----------------------------------------------------------------------------
size_t SLIOReaderMemory::tell()
{
    return _position;
}
//-----------------------------------------------------------------------------
bool SLIOReaderMemory::seek(size_t offset, Origin origin)
{
    std::vector<char>& memoryFile  = memoryFiles[_path];
    size_t             size        = (size_t)memoryFile.size();
    size_t             previousPos = _position;

    switch (origin)
    {
        case IOO_beg: _position = offset; break;
        case IOO_cur: _position += offset; break;
        case IOO_end: _position = size - 1 - offset; break;
    }

    bool ok = _position >= 0 && _position < size;
    if (!ok)
        _position = previousPos;

    return ok;
}
//-----------------------------------------------------------------------------
size_t SLIOReaderMemory::size()
{
    return memoryFiles[_path].size();
}
//-----------------------------------------------------------------------------
SLIOWriterMemory::SLIOWriterMemory(SLstring path)
  : _path(path),
    _position(0)
{
    SL_LOG("WRITING TO MEMORY");
}
//-----------------------------------------------------------------------------
size_t SLIOWriterMemory::write(const void* buffer, size_t size)
{
    std::vector<char>& memoryFile = memoryFiles[_path];
    memoryFile.insert(memoryFile.end(), (char*)buffer, (char*)buffer + size);
    _position += size;
    return size;
}
//-----------------------------------------------------------------------------
size_t SLIOWriterMemory::tell()
{
    return _position;
}
//-----------------------------------------------------------------------------
bool SLIOWriterMemory::seek(size_t offset, Origin origin)
{
    std::vector<char>& memoryFile  = memoryFiles[_path];
    size_t             size        = (size_t)memoryFile.size();
    size_t             previousPos = _position;

    switch (origin)
    {
        case IOO_beg: _position = offset; break;
        case IOO_cur: _position += offset; break;
        case IOO_end: _position = size - 1 - offset; break;
    }

    bool ok = _position >= 0 && _position < size;
    if (!ok)
        _position = previousPos;

    return ok;
}
//-----------------------------------------------------------------------------
void SLIOWriterMemory::flush()
{
}
//-----------------------------------------------------------------------------
#endif