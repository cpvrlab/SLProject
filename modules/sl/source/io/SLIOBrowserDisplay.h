//
// Created by vwm1 on 19.10.2022.
//

#ifndef SLPROJECT_SLIOBROWSERDISPLAY_H
#define SLPROJECT_SLIOBROWSERDISPLAY_H

#include <SLFileStorage.h>
#include <SLIOMemory.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
class SLIOWriterBrowserDisplay : public SLIOWriterMemory
{
public:
    SLIOWriterBrowserDisplay(SLstring path);
    ~SLIOWriterBrowserDisplay();
    void   flush();
};
//-----------------------------------------------------------------------------
#endif

#endif // SLPROJECT_SLIOBROWSERDISPLAY_H
