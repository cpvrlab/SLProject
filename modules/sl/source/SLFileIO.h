#ifndef SLPROJECT_SLFILESYSTEM_H
#define SLPROJECT_SLFILESYSTEM_H

#include <SL.h>
#include <CVTypedefs.h>

#ifndef SL_EMSCRIPTEN
#    define SL_FS_NATIVE
#else
#    define SL_FS_WEB
#    include <emscripten.h>
#    include <png.h>
#    include <jpeglib.h>
#    include <cstdio>
#endif

typedef CVMat SLImageFile;

#ifdef SL_FS_WEB
struct SLFetchResult
{
    int            status;
    unsigned char* data;
    int            length;
};

class SLReader
{
public:
    SLReader(unsigned char* data, int size);
    void read(unsigned char* dest, int size);

private:
    unsigned char* data;
    int            size;
    int            offset;
};
#endif

namespace SLFileIO
{

SLImageFile loadImage(SLstring path);

#ifdef SL_FS_WEB
SLFetchResult fetch(SLstring url);
SLImageFile   decodePNG(SLFetchResult& fetchResult);
void          pngReadData(png_structp png, png_bytep data, png_size_t length);
SLImageFile   decodeJPEG(SLFetchResult& fetchResult);
#endif

} // namespace SLFileIO

#endif // SLPROJECT_SLFILESYSTEM_H
