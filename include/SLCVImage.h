//#############################################################################
//  File:      SL/SLCPImage.h
//  Author:    Marcus Hudritsch
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLIMAGE_H
#define SLIMAGE_H

#include <stdafx.h>
#include <SLCV.h>

//-----------------------------------------------------------------------------
//! OpenCV image class with the same interface as the former SLImage class
/*! The core object is the OpenCV matrix _cvMat. Be aware the OpenCV accesses its
matrix of type mat often by row and columns. In that order it corresponds to
the y and x coordiantes and not x and y as we are used to! 
See the OpenCV docs for more information: 
http://docs.opencv.org/2.4.10/modules/core/doc/basic_structures.html#mat
\n
Before OpenCV was integrated we used the class SLImage to load images.
It used the PNG and JPEG library to load these formats. Since the integration
of OpenCV we kept the interface and migrated the methods to work with OpenCV.
*/
class SLCVImage : public SLObject
{
    public:
                            SLCVImage       () {}
                            SLCVImage       (SLint width,
                                             SLint height,
                                             SLPixelFormat format,
                                             SLstring name);
                            SLCVImage       (const SLstring imageFilename, 
                                             SLbool flipVertical = true,
                                             SLbool loadGrayscaleIntoAlpha = false);
                            SLCVImage       (SLCVImage &srcImage);
                            SLCVImage       (const SLVCol3f& colors);
                            SLCVImage       (const SLVCol4f& colors);
                           ~SLCVImage       ();
            // Misc                         
            void            clearData       ();
            SLbool          allocate        (SLint width,
                                             SLint height,
                                             SLPixelFormat format,
                                             SLbool isContinuous = true);
            void            load            (const SLstring filename, 
                                             SLbool flipVertical = true,
                                             SLbool loadGrayscaleIntoAlpha = false);
            SLbool          load            (SLint inWidth,
                                             SLint inHeight,
                                             SLPixelFormat srcFormat,
                                             SLPixelFormat dstFormat,
                                             SLuchar* data,
                                             SLbool isContinuous,
                                             SLbool isTopLeft);
            void            savePNG         (const SLstring filename,
                                             const SLint compressionLevel=5);
            void            saveJPG         (const SLstring filename,
                                             const SLint compressionLevel=95);
            SLCol4f         getPixeli       (SLint x, SLint y);
            SLCol4f         getPixelf       (SLfloat x, SLfloat y);
            void            setPixeli       (SLint x, SLint y, SLCol4f color);
            void            setPixeliRGB    (SLint x, SLint y, SLCol3f color);
            void            setPixeliRGB    (SLint x, SLint y, SLCol4f color);
            void            setPixeliRGBA   (SLint x, SLint y, SLCol4f color);
            void            resize          (SLint width,
                                             SLint height);
            void            flipY           ();
            void            fill            (SLubyte r, 
                                             SLubyte g,
                                             SLubyte b);
            void            fill            (SLubyte r,
                                             SLubyte g,
                                             SLubyte b,
                                             SLubyte a);
    static  SLPixelFormat   cv2glPixelFormat(SLint cvType);

            // Getters                      
            SLCVMat         cvMat           () {return _cvMat;}
            SLubyte*        data            () {return _cvMat.data;}
            SLuint          width           () {return _cvMat.cols;}
            SLuint          height          () {return _cvMat.rows;}
            SLuint          bytesPerPixel   () {return _bytesPerPixel;}
            SLuint          bytesPerLine    () {return _bytesPerLine;}
            SLuint          bytesPerImage   () {return _bytesPerImage;}
            SLPixelFormat   format          () {return _format;}
            SLstring        formatString    ();
            SLstring        path            () {return _path;}
                                            
    private:
            SLint           bytesPerPixel   (SLPixelFormat pixelFormat);
            SLint           bytesPerLine    (SLint width, 
                                             SLPixelFormat pixelFormat,
                                             SLbool isContinuous = false);
                                            
            SLCVMat         _cvMat;         //!< OpenCV mat matrix image type
            SLPixelFormat   _format;        //!< OpenGL pixel format
            SLint           _bytesPerPixel; //!< Number of bytes per pixel
            SLint           _bytesPerLine;  //!< Number of bytes per line (stride)
            SLint           _bytesPerImage; //!< Number of bytes per image
            SLstring        _path;          //!< path on the filesystem
};
//-----------------------------------------------------------------------------
typedef std::vector<SLCVImage*> SLCVVImage;
//-----------------------------------------------------------------------------
#endif
