//#############################################################################
//   File:      SLIOMemory.h
//   Date:      October 2022
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marino von Wattenwyl
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLIOMEMORY_H
#define SLPROJECT_SLIOMEMORY_H

#include <SLFileStorage.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
//! Collection of functions for accessing files stored in memory
namespace SLIOMemory
{
bool               exists(SLstring path);
std::vector<char>& get(SLstring path);
void               set(SLstring path, const std::vector<char>& data);
void               clear(SLstring path);
}
//-----------------------------------------------------------------------------
//! SLIOStream implementation for reading from memory
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
//! SLIOStream implementation for reading to memory
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
