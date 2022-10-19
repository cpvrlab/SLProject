//
// Created by vwm1 on 11/10/2022.
//

#ifndef SLPROJECT_SLIOMEMORY_H
#define SLPROJECT_SLIOMEMORY_H

#include <SLFileStorage.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
namespace SLIOMemory
{
bool exists(SLstring path);
std::vector<char>& get(SLstring path);
void set(SLstring path, const std::vector<char>& data);
}
//-----------------------------------------------------------------------------
class SLIOReaderMemory : public SLIOStream
{
public:
    SLIOReaderMemory(SLstring path);
    size_t read(void* buffer, size_t size);
    size_t tell();
    bool   seek(size_t offset, Origin origin);
    size_t size();

protected:
    SLstring _path;
    size_t   _position;
};
//-----------------------------------------------------------------------------
class SLIOWriterMemory : public SLIOStream
{
public:
    SLIOWriterMemory(SLstring path);
    size_t write(const void* buffer, size_t size);
    size_t tell();
    bool   seek(size_t offset, Origin origin);
    void   flush();

protected:
    SLstring _path;
    size_t   _position;
};
//-----------------------------------------------------------------------------
#endif

#endif // SLPROJECT_SLIOMEMORY_H
