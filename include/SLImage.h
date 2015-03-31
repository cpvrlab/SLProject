//#############################################################################
//  File:      SL/SLImage.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLIMAGE_H
#define SLIMAGE_H

#include <stdafx.h>

//-----------------------------------------------------------------------------
//! Win32 BITMAPFILEHEADER struct for BMP loading
#pragma pack(push,2)
typedef struct tagBMP_FILEHEADER     
{
    SLushort bfType;
    SLuint   bfSize;
    SLushort bfReserved1;
    SLushort bfReserved2;
    SLuint   bfOffBits;
} sBMP_FILEHEADER;
#pragma pack(pop)
//-----------------------------------------------------------------------------
//! Win32 BITMAPINFOHEADER struct for BMP loading
typedef struct tagBMP_INFOHEADER
{
    SLuint   biSize;
    SLint    biWidth;
    SLint    biHeight;
    SLushort biPlanes;
    SLushort biBitCount;
    SLuint   biCompression;
    SLuint   biSizeImage;
    SLint    biXPelsPerMeter;
    SLint    biYPelsPerMeter;
    SLuint   biClrUsed;
    SLuint   biClrImportant;
} sBMP_INFOHEADER;
//-----------------------------------------------------------------------------
//! Win32 RGBQUAD struct for BMP loading
typedef struct tagBMP_RGBQUAD 
{
    SLuchar rgbBlue;
    SLuchar rgbGreen;
    SLuchar rgbRed;
    SLuchar rgbReserved;
} sRGBQUAD;
//-----------------------------------------------------------------------------
//! Win32 BITMAPINFO struct for BMP loading
typedef struct tagBMP_INFO
{
    sBMP_INFOHEADER bmiHeader;
    sRGBQUAD        bmiColors[1];
} sBMP_INFO;
//-----------------------------------------------------------------------------
//! TGA file header identifier struct
typedef struct
{  SLubyte Header[12];           // TGA File Header
} sTGAHeader;
//-----------------------------------------------------------------------------
//! TGA file header struct
typedef struct
{
    GLubyte header[6];     // First 6 useful bytes from the header
    GLuint  bytesPerPixel; // Holds number of bytes per pixel
    GLuint  imageSize;     // Used to store the size when setting Aside RAM
    GLuint  temp;          // Temporary variable
    GLuint  type;
    GLuint  Height;        // Height of image
    GLuint  Width;         // Width of image
    GLuint  Bpp;           // Bits per pixel
} sTGA;
//-----------------------------------------------------------------------------
//! Small image class for loading JPG, PNG, BMP, TGA and saving PNG files 
/*! Minimal class for loading JPG, PNG, BMP, TGA and saving PNG files. In addition
you can fill, resize, flip and convolve an image. The class is used in 
SLGLTexture.
*/
class SLImage : public SLObject
{
    public:
                            SLImage         () {_data=0; _width=0; _height=0;}
                            SLImage         (SLint width, SLint height, SLuint format);  
                            SLImage         (SLstring imageFilename); 
                            SLImage         (SLImage &srcImage);
                           ~SLImage         ();
            // Misc                         
            void            clearData       ();
            void            allocate        (SLint width, SLint height, SLint format);
            void            load            (SLstring filename); 
            void            savePNG         (SLstring filename);
            SLCol4f         getPixeli       (SLint x, SLint y);
            SLCol4f         getPixelf       (SLfloat x, SLfloat y);
            void            setPixeli       (SLint x, SLint y, SLCol4f color);
            void            setPixeliRGB    (SLint x, SLint y, SLCol3f color);
            void            setPixeliRGB    (SLint x, SLint y, SLCol4f color);
            void            setPixeliRGBA   (SLint x, SLint y, SLCol4f color);
            void            resize          (SLint width, SLint height, 
                                             SLImage* dstImg=0, SLbool invert=false);
            void            flipY           ();
            void            convolve3x3     (SLfloat* kernel);
            void            fill            (SLubyte r=0, 
                                             SLubyte g=0, 
                                             SLubyte b=0, 
                                             SLubyte a=0);
            // Getters                      
            SLubyte*        data            () {return _data;}
            SLuint          width           () {return _width;}
            SLuint          height          () {return _height;}
            SLuint          bytesPerPixel   () {return _bytesPerPixel;}
            SLuint          bytesPerLine    () {return _bytesPerLine;}
            SLuint          bytesPerImage   () {return _bytesPerImage;}
            SLuint          format          () {return _format;}
            SLstring        path            () {return _path;}
                                            
    private:                                
            void            loadJPG         (SLstring filename);
            void            loadPNG         (SLstring filename);
            void            loadBMP         (SLstring filename);
            void            loadTGA         (SLstring filename);
            void            loadTGAuncompr  (SLstring filename, FILE* fp, sTGA& tga);
            void            loadTGAcompr    (SLstring filename, FILE* fp, sTGA& tga);
                                            
            SLubyte*        _data;          //!< pointer to the image memory
            SLint           _width;         //!< width of the texture image in pixel
            SLint           _height;        //!< height of the texture image
            SLint           _format;        //!< Component format
            SLint           _bytesPerPixel; //!< Number of bytes per pixel
            SLint           _bytesPerLine;  //!< Number of bytes per line (stride)
            SLint           _bytesPerImage; //!< Number of bytes per image
            SLstring        _path;          //!< path on the filesystem
};
//-----------------------------------------------------------------------------
typedef std::vector<SLImage*> SLVImage;
//-----------------------------------------------------------------------------
#endif
