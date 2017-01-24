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

#ifdef SL_HAS_OPENCV
#include <SLCV.h>
#include <opencv2/opencv.hpp>
#endif

//-----------------------------------------------------------------------------
//! OpenCV image class with the same interface as the former SLImage class
/*!
*/
class SLCVImage : public SLObject
{
    public:
                            SLCVImage       () {}
                            SLCVImage       (SLint width,
                                             SLint height,
                                             SLPixelFormat format);
                            SLCVImage       (const SLstring imageFilename);
                            SLCVImage       (SLCVImage &srcImage);
                           ~SLCVImage       ();
            // Misc                         
            void            clearData       ();
            SLbool          allocate        (SLint width,
                                             SLint height,
                                             SLPixelFormat format,
                                             SLbool isContinuous = false);
            void            load            (const SLstring filename);
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
//            void            resize          (SLint width,
//                                             SLint height,
//                                             SLImage* dstImg=0,
//                                             SLbool invert=false);
            void            flipY           ();
            void            fill            (SLubyte r=0, 
                                             SLubyte g=0,
                                             SLubyte b=0);
            void            fill            (SLubyte r=0,
                                             SLubyte g=0,
                                             SLubyte b=0,
                                             SLubyte a=0);
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
