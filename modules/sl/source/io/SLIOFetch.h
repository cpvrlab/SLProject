//
// Created by vwm1 on 11/10/2022.
//

#ifndef SLPROJECT_SLIOFETCH_H
#define SLPROJECT_SLIOFETCH_H

#include <SLFileStorage.h>
#include <SLIOMemory.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
class SLIOReaderFetch : public SLIOReaderMemory
{
public:
    static bool exists(SLstring url);

    SLIOReaderFetch(SLstring url);
    ~SLIOReaderFetch();
};
//-----------------------------------------------------------------------------
#endif

#endif // SLPROJECT_SLIOFETCH_H
