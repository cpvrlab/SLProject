//
// Created by vwm1 on 11/10/2022.
//

#ifndef SLPROJECT_SLIONATIVE_H
#define SLPROJECT_SLIONATIVE_H

#include <SLFileStorage.h>

#if 1
//-----------------------------------------------------------------------------
class SLIOReaderNative : public SLIOStream
{
public:
    SLIOReaderNative(SLstring path);
    size_t read(void* buffer, size_t size);
    size_t tell();
    bool   seek(size_t offset, Origin origin);
    size_t size();

private:
    std::ifstream _stream;
};
//-----------------------------------------------------------------------------
class SLIOWriterNative : public SLIOStream
{
public:
    SLIOWriterNative(SLstring path);
    size_t write(const void* buffer, size_t size);
    size_t tell();
    bool   seek(size_t offset, Origin origin);
    void   flush();

private:
    std::ofstream _stream;
};
//-----------------------------------------------------------------------------
#endif

#endif // SLPROJECT_SLIONATIVE_H
