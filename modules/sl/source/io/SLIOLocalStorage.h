//
// Created by vwm1 on 19.10.2022.
//

#ifndef SLPROJECT_SLIOLOCALSTORAGE_H
#define SLPROJECT_SLIOLOCALSTORAGE_H

#include <SLFileStorage.h>
#include <SLIOMemory.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
namespace SLIOLocalStorage
{
bool exists(SLstring path);
}
//-----------------------------------------------------------------------------
class SLIOReaderLocalStorage : public SLIOReaderMemory
{
public:
    SLIOReaderLocalStorage(SLstring path);
    ~SLIOReaderLocalStorage();
};
//-----------------------------------------------------------------------------
class SLIOWriterLocalStorage : public SLIOWriterMemory
{
public:
    SLIOWriterLocalStorage(SLstring path);
    void   flush();
};
//-----------------------------------------------------------------------------
#endif

#endif // SLPROJECT_SLIOLOCALSTORAGE_H
