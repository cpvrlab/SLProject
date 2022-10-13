//
// Created by vwm1 on 11/10/2022.
//

#ifndef SLPROJECT_SLIOFETCH_H
#define SLPROJECT_SLIOFETCH_H

#include <SLFileStorage.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
struct SLFetchResult
{
    int        status;
    SLIOBuffer buffer;
};
//-----------------------------------------------------------------------------
class SLIOReaderFetch : public SLIOStream
{
public:
    static SLFetchResult fetch(SLstring url);
    static bool          exists(SLstring url);

    SLIOReaderFetch(SLstring url);
    ~SLIOReaderFetch();
    size_t read(void* buffer, size_t size);
    size_t tell();
    bool   seek(size_t offset, Origin origin);
    size_t size();

private:
    SLIOBuffer _buffer;
    size_t     _position;
};
//-----------------------------------------------------------------------------
#endif

#endif // SLPROJECT_SLIOFETCH_H
