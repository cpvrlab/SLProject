//
// Created by vwm1 on 19.10.2022.
//

#ifndef SLPROJECT_SLIOBROWSEROPEN_H
#define SLPROJECT_SLIOBROWSEROPEN_H

#include <SLFileStorage.h>
#include <SLIOMemory.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
class SLIOWriterBrowserOpen : public SLIOWriterMemory
{
public:
    SLIOWriterBrowserOpen(SLstring path);
    void   flush();
};
//-----------------------------------------------------------------------------
#endif

#endif // SLPROJECT_SLIOBROWSEROPEN_H
