#ifndef SLPROJECT_SLFILESTORAGE_H
#define SLPROJECT_SLFILESTORAGE_H

#include <SL.h>

#ifndef SL_EMSCRIPTEN
#    define SL_STORAGE_FS
#    include <fstream>
#else
#    define SL_STORAGE_WEB
#    include <emscripten.h>
#    include <unordered_map>
#endif

//-----------------------------------------------------------------------------
struct SLIOBuffer
{
    unsigned char* data;
    size_t         size;
};
//-----------------------------------------------------------------------------
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
enum SLIOStreamMode
{
    IOM_read,
    IOM_write
};
//-----------------------------------------------------------------------------
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
namespace SLFileStorage
{
SLIOStream* open(SLstring path, SLIOStreamKind kind, SLIOStreamMode mode);
void        close(SLIOStream* stream);
bool        exists(SLstring path, SLIOStreamKind kind);
SLIOBuffer  readIntoBuffer(SLstring path, SLIOStreamKind kind);
void        deleteBuffer(SLIOBuffer& buffer);
SLstring    readIntoString(SLstring path, SLIOStreamKind kind);
}
//-----------------------------------------------------------------------------
#endif