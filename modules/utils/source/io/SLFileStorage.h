//#############################################################################
//  File:      SLFileStorage.h
//  Date:      October 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLFILESTORAGE_H
#define SLPROJECT_SLFILESTORAGE_H

#include <Utils.h>
#include <string>

#ifndef __EMSCRIPTEN__
#    define SL_STORAGE_FS
#    include <fstream>
#else
#    define SL_STORAGE_WEB
#    include <emscripten.h>
#    include <unordered_map>
#endif

//-----------------------------------------------------------------------------
//! Utility struct that holds a pointer and its length
struct SLIOBuffer
{
    unsigned char* data;
    size_t         size;
};
//-----------------------------------------------------------------------------
//! Enum of file kinds
enum SLIOStreamKind
{
    IOK_generic,
    IOK_image,
    IOK_model,
    IOK_shader,
    IOK_font,
    IOK_config
};
//-----------------------------------------------------------------------------
//! Enum of stream opening modes
enum SLIOStreamMode
{
    IOM_read,
    IOM_write
};
//-----------------------------------------------------------------------------
//! Interface for accessing external data using streams
/*!
 * SLIOStream provides an interface to access files which may be stored in the
 * native file system, on a remote server, in memory, etc. Streams should not
 * be instantiated by users of the class, but by SLFileStorage::open, which
 * selects the appropriate stream implementation for a given kind and mode.
 */
class SLIOStream
{
public:
    enum Origin
    {
        IOO_beg,
        IOO_cur,
        IOO_end
    };

    virtual ~SLIOStream() = default;
    virtual size_t read(void* buffer, size_t size) { return 0; }
    virtual size_t write(const void* buffer, size_t size) { return 0; }
    virtual size_t tell() { return 0; }
    virtual bool   seek(size_t offset, Origin origin) { return false; }
    virtual size_t size() { return 0; }
    virtual void   flush() {}
};
//-----------------------------------------------------------------------------
//! Collection of functions to open, use and close streams
namespace SLFileStorage
{
SLIOStream* open(std::string path, SLIOStreamKind kind, SLIOStreamMode mode);
void        close(SLIOStream* stream);
bool        exists(std::string path, SLIOStreamKind kind);
SLIOBuffer  readIntoBuffer(std::string path, SLIOStreamKind kind);
void        deleteBuffer(SLIOBuffer& buffer);
std::string readIntoString(std::string path, SLIOStreamKind kind);
void        writeString(std::string path, SLIOStreamKind kind, const std::string& string);
}
//-----------------------------------------------------------------------------
#endif