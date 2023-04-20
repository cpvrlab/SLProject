//#############################################################################
//  File:      SLIONative.h
//  Date:      October 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLIONATIVE_H
#define SLPROJECT_SLIONATIVE_H

#include <SLFileStorage.h>

#ifdef SL_STORAGE_FS
//! SLIOStream implementation for reading from native files
//-----------------------------------------------------------------------------
class SLIOReaderNative : public SLIOStream
{
public:
    SLIOReaderNative(std::string path);
    size_t read(void* buffer, size_t size);
    size_t tell();
    bool   seek(size_t offset, Origin origin);
    size_t size();

private:
    std::ifstream _stream;
};
//-----------------------------------------------------------------------------
//! SLIOStream implementation for writing to native files
class SLIOWriterNative : public SLIOStream
{
public:
    SLIOWriterNative(std::string path);
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
