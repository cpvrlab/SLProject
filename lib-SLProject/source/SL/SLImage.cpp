//#############################################################################
//  File:      SL/SLImage.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include "SLImage.h"
#include <jpeglib.h>          // JPEG lib
#include <png.h>              // libpng

//-----------------------------------------------------------------------------
//! Constructor for empty image of a certain format and size
SLImage::SLImage(SLint width, SLint height, SLuint format) : SLObject()
{
    _data = 0;
    _width = 0;
    _height = 0;
    allocate(width, height, format);
}
//-----------------------------------------------------------------------------
//! Contructor for image from file
SLImage::SLImage(SLstring  filename) : SLObject(filename)
{
    assert(filename!="");
    _data = 0;
    _width = 0;
    _height = 0;
    load(filename);
}
//-----------------------------------------------------------------------------
//! Copy contructor from a source image
SLImage::SLImage(SLImage &src) : SLObject(src.name())
{
    assert(src.width() && src.height() && src.data());
    _width  = src.width();
    _height = src.height();
    _format = src.format();
    _path   = src.path();
    _bytesPerPixel = src.bytesPerPixel();
    _bytesPerLine  = src.bytesPerLine();
    _bytesPerImage = src.bytesPerImage();
   
    _data = new SLubyte[_bytesPerImage];
    if (!_data) 
    {   cout << "SLImage::SLImage(SLImage): Out of memory!" << endl;
        exit(1);
    }
   
    memcpy(_data, src.data(), bytesPerImage());
    if (!_data) 
    {   cout << "SLImage::SLImage(SLImage): memcpy failed!" << endl;
        exit(1);
    }
}
//-----------------------------------------------------------------------------
SLImage::~SLImage()
{
    //SL_LOG("~SLImage(%s)\n", name().c_str());
    clearData();
}
//-----------------------------------------------------------------------------
//! Deletes all data and resets the image parameters
void SLImage::clearData()
{
    if (_data)
        delete[] _data; 
    _data = 0;
    _width = 0;
    _height = 0;
    _bytesPerPixel = 0;
    _bytesPerLine = 0;
    _bytesPerImage = 0;
    _path = "";
}
//-----------------------------------------------------------------------------
//! Memory allocation function
void SLImage::allocate(SLint width, SLint height, SLint format)
{
    assert(width>0 && height>0);

    // return if essentials are identical
    if (_data && _width==width && _height==height && _format==format) return;
   
    // determine bytes per pixel    
    switch (format)
    {  
        #ifdef SL_GLES2
        case GL_LUMINANCE:      _bytesPerPixel = 1; break;
        case GL_LUMINANCE_ALPHA:_bytesPerPixel = 2; break;
        #else
        case GL_RED:         
        case GL_ALPHA:
        case GL_LUMINANCE:   _bytesPerPixel = 1; break;
        case GL_RG:          _bytesPerPixel = 2; break;
        case GL_BGR:         _bytesPerPixel = 3; break;
        case GL_BGRA:        _bytesPerPixel = 4; break;
        #endif
        case GL_RGB:         _bytesPerPixel = 3; break;
        case GL_RGBA:        _bytesPerPixel = 4; break;
        default: 
            SL_EXIT_MSG("SLImage::Allocate: Allocation failed");
    }
   
    _width  = width;
    _height = height;
    _format = format;
    SLint bitsPerPixel = _bytesPerPixel * 8;
    //_bytesPerLine  = _bytesPerPixel * _width;
    _bytesPerLine  = ((width * bitsPerPixel + 31) / 32) * 4;
    _bytesPerImage = _bytesPerLine * _height;
   
    delete[] _data;
    _data = new SLubyte[_bytesPerImage];

    if (!_data)
        SL_EXIT_MSG("SLImage::Allocate: Allocation failed");
}
//-----------------------------------------------------------------------------
//! Loads the image with the appropriate image loader
void SLImage::load(SLstring filename)
{    
    SLstring ext = SLUtils::getFileExt(filename);
    _name = SLUtils::getFileNameWOExt(filename);
    _path = SLUtils::getPath(filename);
   
    if (ext=="jpg") {loadJPG(filename); return;}
    if (ext=="png") {loadPNG(filename); return;}
    if (ext=="bmp") {loadBMP(filename); return;}
    if (ext=="tga") {loadTGA(filename); return;}
   
    SLstring msg = "SLImage.load: Unsupported image file type: " + ext;
    SL_EXIT_MSG(msg.c_str());
}
//-----------------------------------------------------------------------------
//! Error manager used by the JPEG loader
struct SLImage_JPEG_error
{
    jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

typedef struct SLImage_JPEG_error * my_JPEG_error_ptr;
void my_JPEG_error_exit (j_common_ptr cinfo);
void my_JPEG_error_exit (j_common_ptr cinfo)
{
    my_JPEG_error_ptr myerr = (my_JPEG_error_ptr) cinfo->err;
    (*cinfo->err->output_message) (cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}
//-----------------------------------------------------------------------------
//! Loads the image with the independent jpeg lib. See: http://www.ijg.org/
void SLImage::loadJPG(SLstring filename)
{
    jpeg_decompress_struct cinfo;
    SLImage_JPEG_error jerr;
    FILE* fp;			

    // open image file
    if ((fp = fopen(filename.c_str(), "rb")) == nullptr)
    {   SLstring msg = "SLGLTexture::loadJPEG: Failed to load image: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_JPEG_error_exit;

    if (setjmp(jerr.setjmp_buffer)) 
    {   jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        SL_EXIT_MSG("SLGLTexture::loadJPEG: JPEG loading error.");
    }
   
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);

    (void) jpeg_read_header(&cinfo, TRUE);
    (void) jpeg_start_decompress(&cinfo);
    int row_stride = cinfo.output_width * cinfo.output_components;

    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)	((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
   
    _width = cinfo.output_width;
    _height = cinfo.output_height;
    _bytesPerPixel = cinfo.output_components;
    _bytesPerLine  = cinfo.output_components * _width;
    _bytesPerImage = _bytesPerLine * _height;
   
    // now lets allocate the image memory
    _data = new SLubyte[_bytesPerImage];
   
    if (!_data)
    {   fclose(fp);
        SL_EXIT_MSG("SLGLTexture::loadJPEG: Memory allocation failed!");
    }

    while (cinfo.output_scanline < cinfo.output_height)
    {   JDIMENSION read_now = jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(&_data[(cinfo.output_scanline - read_now) * cinfo.output_width * cinfo.output_components], 
               buffer[0], row_stride);
    }
   
    // Convert JPEG color space to OpenGL format
    switch (cinfo.out_color_space)
    {   case JCS_GRAYSCALE:  
            #ifdef SL_GLES2
            _format = GL_LUMINANCE; 
            #else
            _format = GL_RED; 
            #endif
            _bytesPerPixel = 1;
            break;
        case JCS_RGB:        
            _format = GL_RGB;
            _bytesPerPixel = 3;
            break; 
        case JCS_UNKNOWN: 
            SL_EXIT_MSG("JCS_UNKNOWN: invalid color space"); break;
        case JCS_YCbCr:   
            SL_EXIT_MSG("JCS_YCbCr invalid color space"); break;
        case JCS_CMYK:    
            SL_EXIT_MSG("JCS_CMYK invalid color space"); break;
        case JCS_YCCK:    
            SL_EXIT_MSG("JCS_YCCK invalid color space"); break;
        default:
            SL_EXIT_MSG("Invalid color space"); break;
    }

    (void) jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
   
    fclose(fp);
   
    flipY(); // JPEGs are top-left instead of bottom-left
}
//-----------------------------------------------------------------------------
//! Loads the image with libpng. See http://www.libpng.org/pub/png/libpng.html
void SLImage::loadPNG(SLstring filename)
{
    png_bytep*  pRow = 0;
    FILE*       fp = 0;

    // open image file
    fp = fopen (filename.c_str(), "rb");
    if (!fp)
    {   fclose(fp);
        SLstring msg = "SLGLTexture::loadPNG: Failed to load image: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    // read magic number
    png_byte magic[8];
    fread(magic, 1, sizeof (magic), fp);

    // check for valid magic number
    if (!png_check_sig (magic, sizeof (magic)))
    {   fclose(fp);
        SLstring msg = "SLGLTexture::loadPNG: Not a valid PNG file: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    // create a png read struct
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if (!png_ptr)
    {   fclose (fp);
        SL_EXIT_MSG("Failed to create PNG read struct!");
    }

    // create a png info struct
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {   fclose (fp);
        SL_EXIT_MSG("Failed to create PNG info struct!");
    }
   
    // initialize the setjmp for returning properly after a libpng error occured
    if (setjmp(png_jmpbuf(png_ptr)))
    {   fclose (fp);
        png_destroy_read_struct (&png_ptr, &info_ptr, nullptr);
        if (pRow) free(pRow);
        delete[] _data;
        _data = 0;
    }
   
    // setup libpng for using standard C fread() function with our FILE pointer
    png_init_io(png_ptr, fp);

    // tell libpng that we have already read the magic number 
    png_set_sig_bytes(png_ptr, sizeof(magic));

    // read png info 
    png_read_info(png_ptr, info_ptr);

    // get some usefull information from header 
    SLint bit_depth  = png_get_bit_depth(png_ptr, info_ptr);
    SLint color_type = png_get_color_type(png_ptr, info_ptr);

    // convert index color images to RGB images 
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb (png_ptr);

    // convert 1-2-4 bits grayscale images to 8 bits grayscale. 
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha (png_ptr);

    if (bit_depth == 16) png_set_strip_16(png_ptr);
    else if (bit_depth < 8) png_set_packing(png_ptr);

    // update info structure to apply transformations 
    png_read_update_info (png_ptr, info_ptr);

    // retrieve updated information 
    png_get_IHDR(png_ptr, info_ptr, 
                (png_uint_32*)(&_width),
                (png_uint_32*)(&_height), 
                &bit_depth, 
                &color_type, 0, 0, 0);

    // get image format and components per pixel 
    switch (color_type)
    {   case PNG_COLOR_TYPE_GRAY: 
            #ifdef SL_GLES2
            _format = GL_LUMINANCE;
            #else
            _format = GL_RED;
            #endif 
            _bytesPerPixel=1; 
            break;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            #ifdef SL_GLES2
            _format = GL_LUMINANCE_ALPHA;
            #else 
            _format = GL_RG;
            #endif
            _bytesPerPixel=2; 
            break;
        case PNG_COLOR_TYPE_RGB: 
            _format = GL_RGB; 
            _bytesPerPixel=3; 
            break;
        case PNG_COLOR_TYPE_RGB_ALPHA: 
            _format = GL_RGBA; 
            _bytesPerPixel=4; 
            break;
        default: 
            SL_EXIT_MSG("Wrong PNG color type!"); break;
    }
   
    _bytesPerLine  = _bytesPerPixel * _width;
    _bytesPerImage = _bytesPerLine * _height;

    // we can now allocate memory for storing pixel data 
    _data = new SLubyte[_bytesPerImage];

    // setup a pointer array. Each one points at the beginning of a row.
    pRow = (png_bytep*)malloc (sizeof (png_bytep) * _height);

    if (pRow)
    {
        for (SLint i=0; i<_height; ++i)
            pRow[i] = (png_bytep)(_data + ((_height-(i+1)) * _width * _bytesPerPixel));

        // read pixel data using row pointers
        png_read_image(png_ptr, pRow);

        // finish decompression and release memory
        png_read_end(png_ptr, nullptr);
        png_destroy_read_struct (&png_ptr, &info_ptr, nullptr);

        // we don't need row pointers anymore
        free(pRow);
    }
    fclose (fp);
}
//-----------------------------------------------------------------------------
//! Loads a Windows Bitmap image file. Only Bottom Left 24 bit BMP are allowed.
void SLImage::loadBMP(SLstring filename)
{  
    FILE*               fp;          // File pointer
    sBMP_FILEHEADER     BFH;         // Bitmap fp header struct
    sBMP_INFOHEADER     BIH;         // Bitmap info header struct

    // Open file
    if ((fp = fopen(filename.c_str(), "rb")) == nullptr)
    {   fclose(fp);
        SLstring msg = "SLImage::loadBMP: Failed to open image: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    // Read data into bitmap file header into BFH struct
    if (fread((char*)&BFH, sizeof(sBMP_FILEHEADER),1, fp) != 1)
    {   fclose(fp);
        SLstring msg = "SLImage::loadBMP: Failed to read file header: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    // Check if first to bytes are "BM"
    if (BFH.bfType != 0x4D42) 
    {   fclose(fp);
        cout << "SLImage::loadBMP: File is not a BMP: " << filename.c_str() << endl;
    }

    // Read bitmap info header block into BIH struct
    if (fread((char*) &BIH, sizeof(sBMP_INFOHEADER),1, fp) != 1)
    {   fclose(fp);
        SLstring msg = "SLImage::loadBMP: Failed to read info header: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }
   
    if (BIH.biBitCount!=24)
    {   fclose(fp);
        SLstring msg = "SLImage::loadBMP: Only 24 Bit BMP allowed: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }
   
    if (BIH.biHeight < 0)
    {   fclose(fp);
        SL_EXIT_MSG("SLImage::loadBMP: biHeight<0");
    }

    // Allocate for the temporary image data block for BGR data
    SLint tmpBytesPerPixel = BIH.biBitCount==24 ? 3 : 4; 
    SLint tmpStride = ((BIH.biWidth * BIH.biBitCount + 31) / 32) * 4;
    SLuint tmpSizeData = tmpStride * BIH.biHeight;
    SLubyte* tmpData = new SLubyte[tmpSizeData];
    if (!tmpData)
    {   fclose(fp);
        SL_EXIT_MSG("SLImage::loadBMP: Not enough memory!");
    }
   
    // Read the image data block from the file into the temp. memory block
    size_t bytesRead = fread((SLubyte*)tmpData, 1, tmpStride * BIH.biHeight, fp);
    fclose(fp);
    if (bytesRead < tmpSizeData)
        SL_EXIT_MSG("SLImage::loadBMP: Not enough data in file!");

    // Allocate for the image data block for RGB data
    _format = GL_RGB;
    _width  = BIH.biWidth;
    _height = BIH.biHeight;
    _bytesPerPixel = 3; 
    _bytesPerLine  = _width * _bytesPerPixel;
    _bytesPerImage = _bytesPerLine * _height;
    _data = new SLubyte[_bytesPerImage];
    if (!_data)
    {   fclose(fp);
        SL_EXIT_MSG("SLImage::loadBMP: Not enough memory!");
    }
   
    // copy BGR to RGB data
    SLubyte* srcLineStart = tmpData;
    SLubyte* dstLineStart = _data;
    for (SLint h=0; h<_height; ++h)
    {   SLubyte* srcBits = srcLineStart;
        SLubyte* dstBits = dstLineStart;
        for (SLint w=0; w<_width; ++w)
        {   *dstBits = *(srcBits+2); dstBits++;
            *dstBits = *(srcBits+1); dstBits++;
            *dstBits = *(srcBits);   dstBits++;
            srcBits += tmpBytesPerPixel;
        }
        srcLineStart += tmpStride;
        dstLineStart += _bytesPerLine;
    }
   
    // release temp. data block
    delete[] tmpData;
}
//-----------------------------------------------------------------------------
//! Loads a compressed or uncompressed TGA image file
void SLImage::loadTGA(SLstring filename)
{  
    FILE* fp; // File pointer

    // Open file
    if ((fp = fopen(filename.c_str(), "rb")) == nullptr)
    {   SLstring msg = "SLImage::loadTGA: Failed to open image: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    // Attempt to read 12 byte header from file
    sTGAHeader tgaheader;
    if(fread(&tgaheader, sizeof(sTGAHeader), 1, fp) == 0)
    {   fclose(fp);
        SLstring msg = "SLImage::loadTGA: Failed to read file header: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    // Read TGA header
    sTGA tga;
    if(fread(tga.header, sizeof(tga.header), 1, fp) == 0)
    {   fclose(fp);
        SLstring msg = "SLImage::loadTGA: Failed to read TGA header: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }	

    // Get with, height & pits per pixel
    _width  = tga.header[1] * 256 + tga.header[0];	// width =  (highbyte*256+lowbyte)
    _height = tga.header[3] * 256 + tga.header[2];	// height = (highbyte*256+lowbyte)
    SLint bitsPerPixel = tga.header[4];
    _bytesPerPixel = bitsPerPixel / 8;
    _bytesPerLine  = _width * _bytesPerPixel;
    _bytesPerImage = _bytesPerLine * _height;

    tga.Width = _width;
    tga.Height = _height;
    tga.Bpp = bitsPerPixel;
    tga.bytesPerPixel = _bytesPerPixel;
    tga.imageSize = _bytesPerImage;

    if(_width <= 0 || _height <= 0)
    {  fclose(fp);
        SL_EXIT_MSG("SLImage::loadTGA: Invalid image size!");
    }

    if (bitsPerPixel == 24)
        _format = GL_RGB;
    else
    if (bitsPerPixel == 32)
        _format = GL_RGBA;
    else
    {   fclose(fp);
        SL_EXIT_MSG("SLImage::loadTGA: Only 24 and 32 bit bit per pixel supported!");
    }

    // allocate image memory
    _data = new SLubyte[_bytesPerImage];
    if (!_data)
    {   fclose(fp);
        SL_EXIT_MSG("SLImage::loadTGA: Not enough memory!");
    }

    // Check for uncompressed or compressed TGA file types
    GLubyte uTGAcompare[12] = {0,0,2, 0,0,0,0,0,0,0,0,0};	// Uncompressed TGA Header
    GLubyte cTGAcompare[12] = {0,0,10,0,0,0,0,0,0,0,0,0};	// Compressed TGA Header
    if(memcmp(uTGAcompare, &tgaheader, sizeof(tgaheader)) == 0)
        loadTGAuncompr(filename, fp, tga);
    else if(memcmp(cTGAcompare, &tgaheader, sizeof(tgaheader)) == 0)
        loadTGAcompr(filename, fp, tga);
    else 
    {  fclose(fp);
        delete[] _data;
        SL_EXIT_MSG("SLImage::loadTGA: Only compressed or uncompressed TGA file types supported.");
    }

    // close file pointer
    fclose(fp);
}
//-----------------------------------------------------------------------------
//! Loads a uncompressed TGA image file
void SLImage::loadTGAuncompr(SLstring filename, FILE* fp, sTGA& tga)
{  
    // Attempt to read image data
    if(fread(_data, 1, _bytesPerImage, fp) != _bytesPerImage)	
    {   fclose(fp);
        delete[] _data;
        SLstring msg = "SLImage::loadTGA: not enough data in file: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    // Byte Swapping Optimized By Steve Thomas
    for(SLuint cswap = 0; cswap < (SLuint)_bytesPerImage; cswap += _bytesPerPixel)
        _data[cswap] ^= _data[cswap+2] ^= _data[cswap] ^= _data[cswap+2];
}
//-----------------------------------------------------------------------------
//! Loads a uncompressed TGA image file
void SLImage::loadTGAcompr(SLstring filename, FILE* fp, sTGA& tga)
{  
    SLuint pixelcount   = tga.Height * tga.Width;   // Nuber of pixels in the image
    SLuint currentpixel = 0;                        // Current pixel being read
    SLuint currentbyte  = 0;                        // Current byte 
    SLubyte * colorbuffer = (GLubyte *)malloc(tga.bytesPerPixel); // Storage for 1 pixel

    do
    {
        GLubyte chunkheader = 0;                                  // Storage for "chunk" header

        if(fread(&chunkheader, sizeof(GLubyte), 1, fp) == 0)      // Read in the 1 byte header
        {  fclose(fp);
            delete[]_data;
            SL_EXIT_MSG("SLImage::loadTGAcompr: Could not read RLE header.");
        }

        // If the ehader is < 128, it means the that is the number of RAW color packets minus 1
        if(chunkheader < 128)
        {
            chunkheader++;

            for(short counter = 0; counter < chunkheader; counter++)		// Read RAW color values
            {
                if(fread(colorbuffer, 1, _bytesPerPixel, fp) != _bytesPerPixel)
                {   fclose(fp);
                    free(colorbuffer);
                    delete[]_data;
                    SL_EXIT_MSG("SLImage::loadTGAcompr: Could not read image data.");
                }
            
                // Write to memory. Flip R and B vcolor values around in the process 
                _data[currentbyte		] = colorbuffer[2];
                _data[currentbyte + 1] = colorbuffer[1];
                _data[currentbyte + 2] = colorbuffer[0];

                if(tga.bytesPerPixel == 4)
                    _data[currentbyte + 3] = colorbuffer[3];

                currentbyte += _bytesPerPixel;
                currentpixel++;

                if(currentpixel > pixelcount)
                {   fclose(fp);
                    free(colorbuffer);
                    delete[]_data;
                    SL_EXIT_MSG("SLImage::loadTGAcompr: Too many pixels read.");
                }
            }
        }
        else // chunkheader > 128 RLE data, next color reapeated chunkheader - 127 times
        {
            chunkheader -= 127; // Subteact 127 to get rid of the ID bit
         
            if(fread(colorbuffer, 1, tga.bytesPerPixel, fp) != _bytesPerPixel)
            {   fclose(fp);
                delete[]_data;
                SL_EXIT_MSG("SLImage::loadTGAcompr: Could not read image data.");
            }

            // copy the color into the image data as many times as dictated
            for(short counter = 0; counter < chunkheader; counter++) 
            {
                // switch R and B bytes areound while copying
                _data[currentbyte    ] = colorbuffer[2];
                _data[currentbyte + 1] = colorbuffer[1];
                _data[currentbyte + 2] = colorbuffer[0];

                if(tga.bytesPerPixel == 4)
                    _data[currentbyte + 3] = colorbuffer[3];

                currentbyte += _bytesPerPixel;
                currentpixel++;

                if(currentpixel > pixelcount)
                {   fclose(fp);
                    free(colorbuffer);
                    delete[]_data;
                    SL_EXIT_MSG("SLImage::loadTGAcompr: Too many pixels read.");
                }
            }
        }
    }
    while(currentpixel < pixelcount);
}
//-----------------------------------------------------------------------------
//! Save as PNG using libPNG. See http://www.libpng.org/pub/png/libpng.html
void SLImage::savePNG(SLstring filename)
{  
    if (!_data || !_width || !_height)
    {   SLstring msg = "SLGLTexture::savePNG: No data to write for file:" + filename;
        SL_EXIT_MSG(msg.c_str());
    }
    png_structp png_ptr;
    png_infop   info_ptr;
    png_bytep*  row_ptrs;
   
    // setup a pointer array. Each one points at the beginning of a row.
    row_ptrs = (png_bytep*)malloc (sizeof (png_bytep) * _height);
    if (!row_ptrs) return;

    for (SLint h=0; h<_height; ++h)
        row_ptrs[h] = (png_bytep)(_data+((_height-(h+1))*_width*_bytesPerPixel));

    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp)
    {   SLstring msg = "SLGLTexture::savePNG: Failed to create file: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if (!png_ptr)
    {   fclose(fp);
        SL_EXIT_MSG("SLGLTexture::savePNG: Failed to create write struct.");
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {   fclose(fp);
        SL_EXIT_MSG("SLGLTexture::savePNG: png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {   fclose(fp);
        SL_EXIT_MSG("SLGLTexture::savePNG: Error during init_io");
    }

    png_init_io(png_ptr, fp);

    // write header
    if (setjmp(png_jmpbuf(png_ptr)))
    {   fclose(fp);
        SL_EXIT_MSG("SLGLTexture::savePNG: Error during writing header");
    }

    png_byte color_type = PNG_COLOR_TYPE_RGB;;
    png_byte bit_depth = 8;
   
    switch (_format)
    {  
        #ifdef SL_GLES2
        case GL_LUMINANCE:      color_type = PNG_COLOR_TYPE_GRAY;       break;
        case GL_LUMINANCE_ALPHA:color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
        #else
        case GL_RED:   color_type = PNG_COLOR_TYPE_GRAY;       break;
        case GL_RG:    color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
        #endif
        case GL_RGB:   color_type = PNG_COLOR_TYPE_RGB;        break;
        case GL_RGBA:  color_type = PNG_COLOR_TYPE_RGB_ALPHA;  break;
        default: SL_EXIT_MSG("Wrong GL color format.");
    }
   
    png_set_IHDR(png_ptr, 
                info_ptr, 
                _width, 
                _height,
                bit_depth, 
                color_type, 
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_BASE, 
                PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);


    // begin write
    if (setjmp(png_jmpbuf(png_ptr)))
    {   fclose(fp);
        SL_EXIT_MSG("SLGLTexture::savePNG: Error during writing bytes");
    }

    ///////////////////////////////////
    png_write_image(png_ptr, row_ptrs);
    ///////////////////////////////////

    // end write
    if (setjmp(png_jmpbuf(png_ptr)))
    {   fclose(fp);
        SL_EXIT_MSG("SLGLTexture::savePNG: Error during end of write");
    }

    png_write_end(png_ptr, nullptr);

    free(row_ptrs);
    fclose(fp);
}
//-----------------------------------------------------------------------------
//! getPixeli returns the pixel color at the integer pixel coordinate x, y
SLCol4f SLImage::getPixeli(SLint x, SLint y)
{
    SLuint   addr;
    SLCol4f  color;
   
    x %= _width;
    y %= _height;

    switch (_format)
    {   case GL_RGB:
            addr = _bytesPerLine*y + 3*x;
            color.set(_data[addr], _data[addr+1], _data[addr+2], 255.0f);
            break; 
        case GL_RGBA:
            addr = _bytesPerLine*y + 4*x; 
            color.set(_data[addr], _data[addr+1], _data[addr+2], _data[addr+3]);
            break;
        #ifdef SL_GLES2
        case GL_LUMINANCE:
        #else
        case GL_RED:
        #endif
            addr = _bytesPerLine*y + x;
            color.set(_data[addr], _data[addr], _data[addr], 255.0f);
            break;
        #ifdef SL_GLES2
        case GL_LUMINANCE_ALPHA:
        #else
        case GL_RG:
        #endif
            addr = _bytesPerLine*y + 2*x; 
            color.set(_data[addr], _data[addr], _data[addr], _data[addr+1]);
            break;
        default: SL_EXIT_MSG("SLImage::getPixeli: Unknown format!");      
    }
    color /= 255.0f;   
    return color;
}
//-----------------------------------------------------------------------------
/*!
getPixelf returns a pixel color with its x & y texture coordinates.
If the OpenGL filtering is set to GL_LINEAR a bilinear interpolated color out
of four neighbouring pixels is return. Otherwise the nearest pixel is returned.
*/
SLCol4f SLImage::getPixelf(SLfloat x, SLfloat y)
{     
    // Bilinear interpolation
    SLfloat xf = SL_fract(x) * _width;
    SLfloat yf = SL_fract(y) * _height;

    // corrected fractional parts
    SLfloat fracX = SL_fract(xf);
    SLfloat fracY = SL_fract(yf);
    fracX -= SL_sign(fracX-0.5f)*0.5f;
    fracY -= SL_sign(fracY-0.5f)*0.5f;

    // calculate area weights of the four neighbouring texels
    SLfloat X1 = 1.0f - fracX;
    SLfloat Y1 = 1.0f - fracY;
    SLfloat UL = X1 * Y1;
    SLfloat UR = fracX * Y1;
    SLfloat LL = X1 * fracY;
    SLfloat LR = fracX * fracY;

    // get the color of the four neighbouring texels
    //SLint xm, xp, ym, yp;
    //Fast2Int(&xm, xf-1.0f); 
    //Fast2Int(&ym, yf-1.0f);
    //Fast2Int(&xp, xf);
    //Fast2Int(&yp, yf);
    //   
    //SLCol4f cUL = getPixeli(xm,ym);
    //SLCol4f cUR = getPixeli(xp,ym);
    //SLCol4f cLL = getPixeli(xm,yp);
    //SLCol4f cLR = getPixeli(xp,yp);
   
    SLCol4f cUL = getPixeli((SLint)(xf-0.5f),(SLint)(yf-0.5f));
    SLCol4f cUR = getPixeli((SLint)(xf+0.5f),(SLint)(yf-0.5f));
    SLCol4f cLL = getPixeli((SLint)(xf-0.5f),(SLint)(yf+0.5f));
    SLCol4f cLR = getPixeli((SLint)(xf+0.5f),(SLint)(yf+0.5f));
   
    // calculate a new interpolated color with the area weights 
    SLfloat r = UL*cUL.r + LL*cLL.r + UR*cUR.r + LR*cLR.r;
    SLfloat g = UL*cUL.g + LL*cLL.g + UR*cUR.g + LR*cLR.g;
    SLfloat b = UL*cUL.b + LL*cLL.b + UR*cUR.b + LR*cLR.b;
   
    return SLCol4f(r,g,b,1);
}
//-----------------------------------------------------------------------------
//! setPixeli sets the pixel color at the integer pixel coordinate x, y
void SLImage::setPixeli(SLint x, SLint y, SLCol4f color)
{  
    if (x<0) x = 0; 
    if (x>=(SLint)_width) x = _width-1;  // 0 <= x < _width
    if (y<0) y = 0; 
    if (y>=(SLint)_height) y = _height-1; // 0 <= y < _height
   
    SLubyte* addr;
    SLint R, G, B;

    switch (_format)
    {   case GL_RGB:
            addr = _data + _bytesPerLine*y + 3*x;
            *(addr++) = (SLubyte)(color.r * 255.0f);
            *(addr++) = (SLubyte)(color.g * 255.0f);
            *(addr  ) = (SLubyte)(color.b * 255.0f);
            break; 
        case GL_RGBA:
            addr = _data + _bytesPerLine*y + 4*x; 
            *(addr++) = (SLubyte)(color.r * 255.0f);
            *(addr++) = (SLubyte)(color.g * 255.0f);
            *(addr++) = (SLubyte)(color.b * 255.0f);
            *(addr  ) = (SLubyte)(color.a * 255.0f);
            break;
        #ifdef SL_GLES2
        case GL_LUMINANCE:
        #else
        case GL_RED:
        #endif
            addr = _data + _bytesPerLine*y + x;
            R = (SLint)(color.r * 255.0f);
            G = (SLint)(color.g * 255.0f);
            B = (SLint)(color.b * 255.0f);
            *(addr) = (SLubyte)((( 66*R + 129*G +  25*B + 128)>>8) + 16);
            break;
        #ifdef SL_GLES2
        case GL_LUMINANCE_ALPHA:
        #else
        case GL_RG:
        #endif
            addr = _data + _bytesPerLine*y + 2*x; 
            R = (SLint)(color.r * 255.0f);
            G = (SLint)(color.g * 255.0f);
            B = (SLint)(color.b * 255.0f);
            *(addr++) = (SLubyte)((( 66*R + 129*G +  25*B + 128)>>8) + 16);
            *(addr  ) = (SLubyte)(color.a * 255.0f);
            break;
        default: SL_EXIT_MSG("SLImage::setPixeli: Unknown format!");
    }
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGB pixel color at the integer pixel coordinate x, y
void SLImage::setPixeliRGB(SLint x, SLint y, SLCol3f color)
{  
    assert(_bytesPerPixel==3);
    if (x<0) x = 0; 
    if (x>=(SLint)_width)  x = _width -1; // 0 <= x < _width
    if (y<0) y = 0; 
    if (y>=(SLint)_height) y = _height-1; // 0 <= y < _height
   
    SLubyte* addr = _data + _bytesPerLine*y + _bytesPerPixel*x;
    *(addr++) = (SLubyte)(color.r * 255.0f + 0.5f);
    *(addr++) = (SLubyte)(color.g * 255.0f + 0.5f);
    *(addr)   = (SLubyte)(color.b * 255.0f + 0.5f);
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGB pixel color at the integer pixel coordinate x, y
void SLImage::setPixeliRGB(SLint x, SLint y, SLCol4f color)
{  
    assert(_bytesPerPixel==3);
    if (x<0) x = 0; 
    if (x>=(SLint)_width)  x = _width -1; // 0 <= x < _width
    if (y<0) y = 0; 
    if (y>=(SLint)_height) y = _height-1; // 0 <= y < _height
   
    SLubyte* addr = _data + _bytesPerLine*y + _bytesPerPixel*x;
    *(addr++) = (SLubyte)(color.r * 255.0f + 0.5f);
    *(addr++) = (SLubyte)(color.g * 255.0f + 0.5f);
    *(addr)   = (SLubyte)(color.b * 255.0f + 0.5f);
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGBA pixel color at the integer pixel coordinate x, y
void SLImage::setPixeliRGBA(SLint x, SLint y, SLCol4f color)
{  
    assert(_bytesPerPixel==4);
    if (x<0) x = 0; 
    if (x>=(SLint)_width)  x = _width -1; // 0 <= x < _width
    if (y<0) y = 0; 
    if (y>=(SLint)_height) y = _height-1; // 0 <= y < _height
   
    SLubyte* addr = _data + _bytesPerLine*y + _bytesPerPixel*x;
    *(addr++) = (SLubyte)(color.r * 255.0f);
    *(addr++) = (SLubyte)(color.g * 255.0f);
    *(addr++) = (SLubyte)(color.b * 255.0f);
    *(addr  ) = (SLubyte)(color.a * 255.0f);
}
//-----------------------------------------------------------------------------
/*!
SLImage::Resize does a scaling with bilinear interpolation. The color of the 
destination pixel is calculated by the summed up color of the 4 underlying 
source pixels multiplied by their fractional area.
If a pointer to a new image (dstImg) is supplied the rescale is applied to the
new image only.
*/
void SLImage::resize(SLint width, SLint height, SLImage* dstImg, SLbool invert)
{  
    assert(_data!=0 && _width>0 && _height>0 && width>0 && height>0);
   
    SLint    dstW = width;
    SLint    dstH = height;
    SLuint   dstBytesPerLine = _bytesPerPixel * width;
    SLuint   dstBytesPerImage = dstBytesPerLine * height;
   
    // allocate new memory for dstImg or for myself
    SLubyte* dstData;
    if (dstImg)
    {   dstImg->allocate(width, height, _format);
        dstData = dstImg->data();
    } else
    {   dstData = new SLubyte[dstBytesPerImage];
        if (!dstData) 
        {  SL_EXIT_MSG("SLImage::resize: Out of memory.");
        }
    }
      
    SLint    srcW = _width;
    SLint    srcH = _height;
    SLfloat  wFac = (SLfloat)srcW / (SLfloat)dstW;
    SLfloat  hFac = (SLfloat)srcH / (SLfloat)dstH;
    SLfloat  wFacHalf = wFac * 0.5f;
    SLfloat  hFacHalf = wFac * 0.5f;
    SLint    srcLineBytes = _bytesPerLine;
    SLubyte* dstStart = dstData;
    SLubyte* pDst;
   
    /*
    +---------+---------+
    | #############     |              
    | #       |   #<------ Dst Pixel  
    | #       |   #     |  
    | #       |   #     |                          
    +-#-------+---#-----+    
    | #       |   #     |                    
    | #############     | 
    |         |         |<- Src Pixels             
    |         |         |  
    +---------+---------+ 
    */ 
   
    SLubyte* srcLineL, *srcLineU;
    SLint    iHL, iHU, iWL, iWR;
    SLfloat  fW, fH = hFac-hFacHalf;
    SLfloat  wUL, wUR, wLL, wLR;
      
    for(SLint h=0; h<dstH; h++, fH+=hFac, dstStart += dstBytesPerLine)
    {   pDst = dstStart;
   
        iHU = min((SLint)(fH+0.5f), srcH-1);
        iHL = max(iHU-1, 0);
      
        srcLineL = _data + iHL * srcLineBytes; 
        if (iHL==iHU) srcLineU = srcLineL;
        else srcLineU = srcLineL + srcLineBytes;
         
        fW = wFac-wFacHalf;

        for(int w=0; w<dstW; w++, fW+=wFac)
        {  
            // calculate 4 pixel pointers
            iWR = min((SLint)(fW+0.5f), srcW-1);
            iWL = max(iWR-1, 0);
         
            // pointers to UpperLeft, UpperRight, LowerLeft & LowerRight pixels
            SLubyte* pUL = srcLineU + iWL*_bytesPerPixel;
            SLubyte* pUR = srcLineU + iWR*_bytesPerPixel;
            SLubyte* pLL = srcLineL + iWL*_bytesPerPixel;
            SLubyte* pLR = srcLineL + iWR*_bytesPerPixel;
  
            SLfloat fracX = SL_abs(SL_fract(fW-0.5f));
            SLfloat fracY = SL_abs(SL_fract(fH-0.5f));
            SLfloat oneMinusFracX = 1.0f - fracX;
            SLfloat oneMinusFracY = 1.0f - fracY;
         
            // weights = normalized subpixel areas
            wUR = fracX * fracY;
            wUL = oneMinusFracX * fracY;
            wLR = fracX * oneMinusFracY;
            wLL = oneMinusFracX * oneMinusFracY;
         
            // calculate the weighted color for each component of RGBA
            if (invert)
            {   for (SLint bpp=0; bpp<_bytesPerPixel; ++bpp)
                {   SLfloat cUL = (SLfloat)*(pUL++);
                    SLfloat cUR = (SLfloat)*(pUR++);
                    SLfloat cLL = (SLfloat)*(pLL++);
                    SLfloat cLR = (SLfloat)*(pLR++);
                    SLfloat col = wUL*cUL + wLL*cLL + wUR*cUR + wLR*cLR;
                    *(pDst++) = 255 - (SLubyte)col;
                }
             } else
             {  for (SLint bpp=0; bpp<_bytesPerPixel; ++bpp)
                {   SLfloat cUL = (SLfloat)*(pUL++);
                    SLfloat cUR = (SLfloat)*(pUR++);
                    SLfloat cLL = (SLfloat)*(pLL++);
                    SLfloat cLR = (SLfloat)*(pLR++);
                    SLfloat col = wUL*cUL + wLL*cLL + wUR*cUR + wLR*cLR;
                    *(pDst++) = (SLubyte)col;
                }
            }
        }
    }
   
    if (!dstImg)
    {   delete[] _data;   // release old memory
        _data = dstData;  // assign new memory
        _width = width;
        _height = height;
        _bytesPerImage = dstBytesPerImage;
        _bytesPerLine = dstBytesPerLine;
    }
}
//-----------------------------------------------------------------------------
//! Flip Y coordiantes used to make JPEGs from top-left to bottom-left images.
void SLImage::flipY()
{  
    if (_data && _width > 0 && _height > 0)
    {  
        // allocate new memory
        SLubyte* pDst = new SLubyte[_bytesPerImage];
        if (!pDst) 
            SL_EXIT_MSG("SLImage::flipY(): Out of memory.");
      
        SLubyte* pSrcBot = _data;
        SLubyte* pDstTop = pDst + _bytesPerImage - _bytesPerLine;
      
        // copy lines
        for (SLint h=0; h<_height; ++h)
        {   memcpy(pDstTop, pSrcBot, _bytesPerLine);   
            pSrcBot += _bytesPerLine;
            pDstTop -= _bytesPerLine;
        }
      
        delete[] _data;   // release old memory
        _data = pDst;     // assign new memory
    }
}
//-----------------------------------------------------------------------------
//! Applies a convolution filter with the 3x3 filter kernel k passed as float k[9]
void SLImage::convolve3x3(SLfloat* k)
{
    assert(_data!=0 && _width>0 && _height>0 && k);

    // allocate new memory for dstImg or for myself
    SLubyte* dstData;
    {   dstData = new SLubyte[_bytesPerImage];
        if (!dstData) 
        {  SL_EXIT_MSG("SLImage::resize: Out of memory.");
        }
    }

    // Sum kernel elements so can gen normalized variables
    SLfloat s = k[0]+k[1]+k[2]+k[3]+k[4]+k[5]+k[6]+k[7]+k[8];
    k[0]/=s; k[1]/=s; k[2]/=s;
    k[3]/=s; k[4]/=s; k[5]/=s;
    k[6]/=s; k[7]/=s; k[8]/=s;

    SLint bpp = _bytesPerPixel;
    SLint bpl = _bytesPerLine;
   
    // concolve the image except the border
    SLint offC[9] = {-bpp+bpl, +bpl, bpp+bpl,
                    -bpp    , 0   , bpp    ,
                    -bpp-bpl, -bpl, bpp-bpl};
    SLubyte* srcLine = _data   + _bytesPerLine + _bytesPerPixel;
    SLubyte* dstLine = dstData + _bytesPerLine + _bytesPerPixel;
    SLubyte* src, *dst;

    for(SLint h=1; h<_height-1; ++h)
    {   src = srcLine;
        dst = dstLine;
        for(SLint w=1; w<_width-1; ++w)
        {   for (SLint p=0; p<_bytesPerPixel; ++p)
            {  *dst = SLubyte(*(src+offC[0])*k[0] + *(src+offC[1])*k[1] + *(src+offC[2])*k[2] +
                            *(src+offC[3])*k[3] + *(src+offC[4])*k[4] + *(src+offC[5])*k[5] +
                            *(src+offC[6])*k[6] + *(src+offC[7])*k[7] + *(src+offC[8])*k[8]);
            src++;
            dst++;
            }
        }
        srcLine+=_bytesPerLine;
        dstLine+=_bytesPerLine;
    }
   
    // concolve the bottom border
    SLint offB[9] = {-bpp+bpl, +bpl, bpp+bpl,
                     -bpp    , 0   , bpp    ,
                     -bpp    , 0   , bpp    };
    src = _data   + _bytesPerPixel;
    dst = dstData + _bytesPerPixel;
    for(SLint w=1; w<_width-1; ++w)
    {   for (SLint p=0; p<_bytesPerPixel; ++p)
        {   *dst = SLubyte(*(src+offB[0])*k[0] + *(src+offB[1])*k[1] + *(src+offB[2])*k[2] +
                           *(src+offB[3])*k[3] + *(src+offB[4])*k[4] + *(src+offB[5])*k[5] +
                           *(src+offB[6])*k[6] + *(src+offB[7])*k[7] + *(src+offB[8])*k[8]);
            src++;
            dst++;
        }
    }
   
    // concolve the top border
    SLint offT[9] = {-bpp    , 0   , bpp,
                    -bpp    , 0   , bpp    ,
                    -bpp-bpl, -bpl, bpp-bpl};
    src = _data   + _bytesPerImage - _bytesPerLine + _bytesPerPixel;
    dst = dstData + _bytesPerImage - _bytesPerLine + _bytesPerPixel;
    for(SLint w=1; w<_width-1; ++w)
    {   for (SLint p=0; p<_bytesPerPixel; ++p)
        {   *dst = SLubyte(*(src+offT[0])*k[0] + *(src+offT[1])*k[1] + *(src+offT[2])*k[2] +
                           *(src+offT[3])*k[3] + *(src+offT[4])*k[4] + *(src+offT[5])*k[5] +
                           *(src+offT[6])*k[6] + *(src+offT[7])*k[7] + *(src+offT[8])*k[8]);
            src++;
            dst++;
        }
    }
   
    // concolve the left border
    SLint offL[9] = {+bpl, +bpl, bpp+bpl,
                        0  , 0   , bpp    ,
                    -bpl, -bpl, bpp-bpl};
    srcLine = _data   + _bytesPerLine;
    dstLine = dstData + _bytesPerLine;
    for(SLint h=1; h<_height-1; ++h)
    {   src = srcLine;
        dst = dstLine;
        for (SLint p=0; p<_bytesPerPixel; ++p)
        {   *dst = SLubyte(*(src+offL[0])*k[0] + *(src+offL[1])*k[1] + *(src+offL[2])*k[2] +
                           *(src+offL[3])*k[3] + *(src+offL[4])*k[4] + *(src+offL[5])*k[5] +
                           *(src+offL[6])*k[6] + *(src+offL[7])*k[7] + *(src+offL[8])*k[8]);
            src++;
            dst++;
        }
        srcLine+=_bytesPerLine;
        dstLine+=_bytesPerLine;
    }
   
    // concolve the right border
    SLint offR[9] = {-bpp+bpl, +bpl, +bpl,
                    -bpp    , 0   ,   0 ,
                    -bpp-bpl, -bpl, -bpl};
    srcLine = _data   + _bytesPerLine + _bytesPerLine - _bytesPerPixel;
    dstLine = dstData + _bytesPerLine + _bytesPerLine - _bytesPerPixel;
    for(SLint h=1; h<_height-1; ++h)
    {   src = srcLine;
        dst = dstLine;
        for (SLint p=0; p<_bytesPerPixel; ++p)
        {   *dst = SLubyte(*(src+offR[0])*k[0] + *(src+offR[1])*k[1] + *(src+offR[2])*k[2] +
                           *(src+offR[3])*k[3] + *(src+offR[4])*k[4] + *(src+offR[5])*k[5] +
                           *(src+offR[6])*k[6] + *(src+offR[7])*k[7] + *(src+offR[8])*k[8]);
            src++;
            dst++;
        }
        srcLine+=_bytesPerLine;
        dstLine+=_bytesPerLine;
    }
   
    // copy the 4 corners
    src = _data;
    dst = dstData;
    for (SLint p=0; p<_bytesPerPixel; ++p) *dst++ = *src++;
   
    src = _data   + _bytesPerLine - _bytesPerPixel;
    dst = dstData + _bytesPerLine - _bytesPerPixel;
    for (SLint p=0; p<_bytesPerPixel; ++p) *dst++ = *src++;
   
    src = _data   + _bytesPerImage - _bytesPerLine;
    dst = dstData + _bytesPerImage - _bytesPerLine;
    for (SLint p=0; p<_bytesPerPixel; ++p) *dst++ = *src++;
   
    src = _data   + _bytesPerImage - _bytesPerPixel;
    dst = dstData + _bytesPerImage - _bytesPerPixel;
    for (SLint p=0; p<_bytesPerPixel; ++p) *dst++ = *src++;
   
   
    delete[] _data;   // release old memory
    _data = dstData;  // assign new memory
}
//-----------------------------------------------------------------------------
//! Fills the image with a certain color
void SLImage::fill(SLubyte r, SLubyte g, SLubyte b, SLubyte a)
{  
    SLubyte* srcLine = _data;
    SLubyte* src;
    SLubyte  rgba[4];
   
    rgba[0] = r; rgba[1] = g; rgba[2] = b; rgba[3] = a;
   
    for(SLint h=0; h<_height-1; ++h, srcLine+=_bytesPerLine)
    {   src = srcLine;
        for(SLint w=0; w<_width-1; ++w)
        {   for (SLint p=0; p<_bytesPerPixel; ++p)
            {   *src = rgba[p];
                src++;
            }
        }
    }
}
//-----------------------------------------------------------------------------
