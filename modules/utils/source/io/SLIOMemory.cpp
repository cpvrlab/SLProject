//#############################################################################
//  File:      SLIOMemory.cpp
//  Date:      October 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLIOMemory.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
#    include <unordered_map>
#    include <vector>
//-----------------------------------------------------------------------------
std::unordered_map<std::string, std::vector<char>> memoryFiles;
//-----------------------------------------------------------------------------
bool SLIOMemory::exists(std::string path)
{
    return memoryFiles.count(path);
}
//-----------------------------------------------------------------------------
std::vector<char>& SLIOMemory::get(std::string path)
{
    return memoryFiles[path];
}
//-----------------------------------------------------------------------------
void SLIOMemory::set(std::string path, const std::vector<char>& data)
{
    memoryFiles[path] = data;
}
//-----------------------------------------------------------------------------
void SLIOMemory::clear(std::string path)
{
    memoryFiles.erase(path);
}
//-----------------------------------------------------------------------------
SLIOReaderMemory::SLIOReaderMemory(std::string path)
  : _path(path)
{
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
SLIOWriterMemory::SLIOWriterMemory(std::string path)
  : _path(path),
    _position(0)
{
    memoryFiles[_path].clear();
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