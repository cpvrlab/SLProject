//#############################################################################
//  File:      CV/SLCVImage.cpp
//  Author:    Marcus Hudritsch
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLCVImage.h>

//-----------------------------------------------------------------------------
//! Constructor for empty image of a certain format and size
SLCVImage::SLCVImage(SLint width, 
                     SLint height, 
                     SLPixelFormat format, 
                     SLstring name) : SLObject(name)
{
    allocate(width, height, format);
}
//-----------------------------------------------------------------------------
//! Contructor for image from file
SLCVImage::SLCVImage(const SLstring  filename, 
                     SLbool flipVertical, 
                     SLbool loadGrayscaleIntoAlpha) :
           SLObject(SLUtils::getFileName(filename), filename)
{
    assert(filename!="");
    clearData();
    load(filename, flipVertical, loadGrayscaleIntoAlpha);
}
//-----------------------------------------------------------------------------
//! Copy contructor from a source image
SLCVImage::SLCVImage(SLCVImage &src) : SLObject(src.name(), src.url())
{
    assert(src.width() && src.height() && src.data());
    _format = src.format();
    _path   = src.path();
    _bytesPerPixel = src.bytesPerPixel();
    _bytesPerLine  = src.bytesPerLine();
    _bytesPerImage = src.bytesPerImage();
    src.cvMat().copyTo(_cvMat);
}
//-----------------------------------------------------------------------------
//! Creates a 1D image from a SLCol3f vector
SLCVImage::SLCVImage(const SLVCol3f &colors)
{
    allocate((SLint)colors.size(), 1, PF_rgb);

    SLuint x=0;
    for (auto color : colors)
    {
        _cvMat.at<cv::Vec3b>(0, x++) = cv::Vec3b((SLuchar)(color.r * 255.0f),
                                                 (SLuchar)(color.g * 255.0f),
                                                 (SLuchar)(color.b * 255.0f));
    }
}
//-----------------------------------------------------------------------------
//! Creates a 1D image from a SLCol4f vector
SLCVImage::SLCVImage(const SLVCol4f &colors)
{
    allocate((SLint)colors.size(), 1, PF_rgba);

    SLuint x=0;
    for (auto color : colors)
    {
        _cvMat.at<cv::Vec4b>(0, x++) = cv::Vec4b((SLuchar)(color.r * 255.0f),
                                                 (SLuchar)(color.g * 255.0f),
                                                 (SLuchar)(color.b * 255.0f),
                                                 (SLuchar)(color.a * 255.0f));
    }
}
//-----------------------------------------------------------------------------
SLCVImage::~SLCVImage()
{
    //SL_LOG("~SLCVImage(%s)\n", name().c_str());
    clearData();
}
//-----------------------------------------------------------------------------
//! Deletes all data and resets the image parameters
void SLCVImage::clearData()
{
    _cvMat.release();
    _bytesPerPixel = 0;
    _bytesPerLine = 0;
    _bytesPerImage = 0;
    _path = "";
}
//-----------------------------------------------------------------------------
//! Memory allocation function
/*! It returns true if width or height or the pixelformat has changed
/param width Width of image in pixels
/param height Height of image in pixels
/param pixelFormatGL OpenGL pixel format enum
/param isContinuous True if the memory is continuous and has no stride bytes at the end of the line
*/
SLbool SLCVImage::allocate(SLint width,
                           SLint height,
                           SLPixelFormat pixelFormatGL,
                           SLbool isContinuous)
{
    assert(width>0 && height>0);

    // return if essentials are identical
    if (!_cvMat.empty() &&
        _cvMat.cols==width &&
        _cvMat.rows==height &&
        _format==pixelFormatGL)
        return false;

    // Set the according OpenCV format
    SLint cvType = 0, bpp = 0;
    switch (pixelFormatGL)
    {   case PF_luminance:  {cvType = CV_8UC1; bpp = 1; break;}
        case PF_red:        {cvType = CV_8UC1; bpp = 1; break;}
        case PF_bgr:        {cvType = CV_8UC3; bpp = 3; break;}
        case PF_rgb:        {cvType = CV_8UC3; bpp = 3; break;}
        case PF_bgra:       {cvType = CV_8UC4; bpp = 4; break;}
        case PF_rgba:       {cvType = CV_8UC4; bpp = 4; break;}
        default: SL_EXIT_MSG("Pixel format not supported");
    }

    _cvMat.create(height, width, cvType);

    _format = pixelFormatGL;
    _bytesPerPixel = bpp;
    _bytesPerLine  = bytesPerLine(width, pixelFormatGL, isContinuous);
    _bytesPerImage = _bytesPerLine * height;

    if (!_cvMat.data)
        SL_EXIT_MSG("SLCVImage::Allocate: Allocation failed");
    return true;
}
//-----------------------------------------------------------------------------
//! Returns the NO. of bytes per pixel for the passed pixel format
SLint SLCVImage::bytesPerPixel(SLPixelFormat format)
{
    switch (format)
    {
        case PF_red:
        case PF_red_integer:
        case PF_green:
        case PF_alpha:
        case PF_blue:
        case PF_luminance:
        case PF_intensity: return 1;
        case PF_rg:
        case PF_rg_integer:
        case PF_luminance_alpha: return 2;
        case PF_rgb:
        case PF_bgr:
        case PF_rgb_integer:
        case PF_bgr_integer: return 3;
        case PF_rgba:
        case PF_bgra:
        case PF_rgba_integer:
        case PF_bgra_integer: return 4;
        default:
            SL_EXIT_MSG("SLCVImage::bytesPerPixel: unknown pixel format");
    }
    return 0;
}
//-----------------------------------------------------------------------------
//! Returns the NO. of bytes per image line for the passed pixel format
/*
/param width Width of image in pixels
/param pixelFormatGL OpenGL pixel format enum
/param isContinuous True if the memory is continuous and has no stride bytes at the end of the line
*/
SLint SLCVImage::bytesPerLine(SLint width,
                            SLPixelFormat format, 
                            SLbool isContinuous)
{
    SLint bpp = bytesPerPixel(format);
    SLint bitsPerPixel = bpp * 8;
    SLint bpl = isContinuous ? width * bpp : 
                ((width * bitsPerPixel + 31) / 32) * 4;
    return bpl;
}
//-----------------------------------------------------------------------------
//! loads an image from a memory with format change.
/*! It returns true if the width, height or destination format has changed so
that the depending texture can be rebuild in OpenGL. If the source and
destination pixel format does not match a conversion for certain formats is
done.
/param width Width of image in pixels
/param height Height of image in pixels
/param srcPixelFormatGL OpenGL pixel format enum of source image
/param dstPixelFormatGL OpenGL pixel format enum of destination image
/param data Pointer to the first byte of the image data
/param isContinuous True if the memory is continuous and has no stride bytes at the end of the line
/param isTopLeft True if image data starts at top left of image (else bottom left)
*/
SLbool SLCVImage::load(SLint width,
                       SLint height,
                       SLPixelFormat srcPixelFormatGL,
                       SLPixelFormat dstPixelFormatGL,
                       SLuchar* data,
                       SLbool isContinuous,
                       SLbool isTopLeft)
{
    
    SLbool needsTextureRebuild = allocate(width, 
                                          height, 
                                          dstPixelFormatGL,
                                          false);
    
    SLint    dstBPL   = _bytesPerLine;
    SLint    dstBPP   = _bytesPerPixel;
    SLint    srcBPP   = bytesPerPixel(srcPixelFormatGL);
    SLint    srcBPL   = bytesPerLine(width, srcPixelFormatGL, isContinuous);
    
    if (isTopLeft)
    {
        // copy lines and flip vertically
        SLubyte* dstStart = _cvMat.data + _bytesPerImage - dstBPL;
        SLubyte* srcStart = data;
        
        if (srcPixelFormatGL==dstPixelFormatGL)
        {
            for (SLint h=0; h<_cvMat.rows; ++h, srcStart += srcBPL, dstStart -= dstBPL)
            {   memcpy(dstStart, srcStart, dstBPL);
            }
        }
        else
        {
            if (srcPixelFormatGL==PF_bgra)
            {
                if (dstPixelFormatGL==PF_rgb)
                {
                    for (SLint h=0; h<_cvMat.rows; ++h, srcStart += srcBPL, dstStart -= dstBPL)
                    {   SLubyte* src = srcStart;
                        SLubyte* dst = dstStart;
                        for(SLint w=0; w<_cvMat.cols-1; ++w, dst += dstBPP, src += srcBPP)
                        {   dst[0]=src[2];
                            dst[1]=src[1];
                            dst[2]=src[0];
                        }
                    }
                }
                else
                if (dstPixelFormatGL==PF_rgba)
                {
                    for (SLint h=0; h<_cvMat.rows; ++h, srcStart += srcBPL, dstStart -= dstBPL)
                    {   SLubyte* src = srcStart;
                        SLubyte* dst = dstStart;
                        for(SLint w=0; w<_cvMat.cols-1; ++w, dst += dstBPP, src += srcBPP)
                        {   dst[0]=src[2];
                            dst[1]=src[1];
                            dst[2]=src[0];
                            dst[3]=src[3];
                        }
                    }
                    
                }
            }  else
            if (srcPixelFormatGL==PF_bgr || srcPixelFormatGL==PF_rgb)
            {
                if (dstPixelFormatGL==PF_rgb || dstPixelFormatGL==PF_bgr)
                {
                    for (SLint h=0; h<_cvMat.rows; ++h, srcStart += srcBPL, dstStart -= dstBPL)
                    {   SLubyte* src = srcStart;
                        SLubyte* dst = dstStart;
                        for(SLint w=0; w<_cvMat.cols-1; ++w, dst += dstBPP, src += srcBPP)
                        {   dst[0]=src[2];
                            dst[1]=src[1];
                            dst[2]=src[0];
                        }
                    }
                }
            } else
            {   cout << "SLCVImage::load from memory: Pixel format conversion not allowed" << endl;
                exit(1);
            }
        }
    }
    else // bottom left (no flipping)
    {
        if (srcPixelFormatGL==dstPixelFormatGL)
        {
            memcpy(_cvMat.data, data, _bytesPerImage);
        }
        else
        {
            SLubyte* dstStart = _cvMat.data;
            SLubyte* srcStart = data;
            
            if (srcPixelFormatGL==PF_bgra)
            {
                if (dstPixelFormatGL==PF_rgb)
                {
                    for(SLint h=0; h<_cvMat.rows-1; ++h, srcStart+=srcBPL, dstStart+=dstBPL)
                    {   SLubyte* src = srcStart;
                        SLubyte* dst = dstStart;
                        for(SLint w=0; w<_cvMat.cols-1; ++w, dst += dstBPP, src += srcBPP)
                        {   dst[0]=src[2];
                            dst[1]=src[1];
                            dst[2]=src[0];
                        }
                    }
                } else
                if (dstPixelFormatGL==PF_rgba)
                {
                    for(SLint h=0; h<_cvMat.rows-1; ++h, srcStart+=srcBPL, dstStart+=dstBPL)
                    {   SLubyte* src = srcStart;
                        SLubyte* dst = dstStart;
                        for(SLint w=0; w<_cvMat.cols-1; ++w, dst += dstBPP, src += srcBPP)
                        {   dst[0]=src[2];
                            dst[1]=src[1];
                            dst[2]=src[0];
                            dst[3]=src[3];
                        }
                    }
                }
            } else
            if (srcPixelFormatGL==PF_bgr || srcPixelFormatGL==PF_rgb)
            {
                if (dstPixelFormatGL==PF_rgb || dstPixelFormatGL==PF_bgr)
                {
                    for(SLint h=0; h<_cvMat.rows-1; ++h, srcStart+=srcBPL, dstStart+=dstBPL)
                    {   SLubyte* src = srcStart;
                        SLubyte* dst = dstStart;
                        for(SLint w=0; w<_cvMat.cols-1; ++w, dst += dstBPP, src += srcBPP)
                        {   dst[0]=src[2];
                            dst[1]=src[1];
                            dst[2]=src[0];
                        }
                    }
                }
            } else
            {   cout << "SLCVImage::load from memory: Pixel format conversion not allowed" << endl;
                exit(1);
            }
        }
    }
    
    return needsTextureRebuild;
}
//-----------------------------------------------------------------------------
//! Loads the image with the appropriate image loader
void SLCVImage::load(const SLstring filename, 
                     SLbool flipVertical, 
                     SLbool loadGrayscaleIntoAlpha)
{    
    SLstring ext = SLUtils::getFileExt(filename);
    _name = SLUtils::getFileName(filename);
    _path = SLUtils::getPath(filename);

    // load the image format as stored in the file
    _cvMat = cv::imread(filename, -1);

    if(!_cvMat.data)
    {   SLstring msg = "SLCVImage.load: Loading failed: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    // Convert greater component depth than 8 bit to 8 bit
    if (_cvMat.depth() > CV_8U)
        _cvMat.convertTo(_cvMat, CV_8U, 1.0/256.0);

    _format = cv2glPixelFormat(_cvMat.type());
    _bytesPerPixel = bytesPerPixel(_format);
    
    // OpenCV always loads with BGR(A) but some OpenGL prefer RGB(A)
    if (_format == PF_bgr) 
    {   cv::cvtColor(_cvMat, _cvMat, CV_BGR2RGB);
        _format = PF_rgb;
    } else
    if (_format == PF_bgra)
    {   cv::cvtColor(_cvMat, _cvMat, CV_BGRA2RGBA);
        _format = PF_rgba;
    } else
    if (_format == PF_red && loadGrayscaleIntoAlpha)
    {
        SLCVMat rgbaImg;
        rgbaImg.create(_cvMat.rows, _cvMat.cols, CV_8UC4);

        // Copy grayscale into alpha channel
        for (int y = 0; y < rgbaImg.rows; ++y)
        {
            SLuchar* dst = rgbaImg.ptr<SLuchar>(y);
            SLuchar* src = _cvMat.ptr<SLuchar>(y);

            for (int x = 0; x < rgbaImg.cols; ++x)
            {
                *dst++ = 0;        // B
                *dst++ = 0;        // G
                *dst++ = 0;        // R
                *dst++ = *src++;   // A
            }
        }

        _cvMat = rgbaImg;
        cv::cvtColor(_cvMat, _cvMat, CV_BGRA2RGBA);
        _format = PF_rgba;

        // for debug check
        //SLstring pathfilename = _path + name();
        //SLstring filename = SLUtils::getFileNameWOExt(pathfilename);
        //savePNG(_path + filename + "_InAlpha.png");
    }
    
    _bytesPerLine  = bytesPerLine(_cvMat.cols, _format, _cvMat.isContinuous());
    _bytesPerImage = _bytesPerLine * _cvMat.rows;

    // OpenCV loads top-left but OpenGL is bottom left
    if (flipVertical)
        flipY();
}
//-----------------------------------------------------------------------------
//! Converts OpenCV mat type to OpenGL pixel format
SLPixelFormat SLCVImage::cv2glPixelFormat(SLint cvType)
{
    switch (cvType)
    {   case CV_8UC1: return PF_red;
        case CV_8UC2: return PF_rg;
        case CV_8UC3: return PF_bgr;
        case CV_8UC4: return PF_bgra; 
        case CV_8SC1: SL_EXIT_MSG("OpenCV image format CV_8SC1 not supported"); break;
        case CV_8SC2: SL_EXIT_MSG("OpenCV image format CV_8SC2 not supported"); break;
        case CV_8SC3: SL_EXIT_MSG("OpenCV image format CV_8SC3 not supported"); break;
        case CV_8SC4: SL_EXIT_MSG("OpenCV image format CV_8SC4 not supported"); break;
        case CV_16UC1: SL_EXIT_MSG("OpenCV image format CV_16UC1 not supported"); break;
        case CV_16UC2: SL_EXIT_MSG("OpenCV image format CV_16UC2 not supported"); break;
        case CV_16UC3: SL_EXIT_MSG("OpenCV image format CV_16UC3 not supported"); break;
        case CV_16UC4: SL_EXIT_MSG("OpenCV image format CV_16UC4 not supported"); break;
        case CV_16SC1: SL_EXIT_MSG("OpenCV image format CV_16SC1 not supported"); break;
        case CV_16SC2: SL_EXIT_MSG("OpenCV image format CV_16SC2 not supported"); break;
        case CV_16SC3: SL_EXIT_MSG("OpenCV image format CV_16SC3 not supported"); break;
        case CV_16SC4: SL_EXIT_MSG("OpenCV image format CV_16SC4 not supported"); break;
        case CV_32SC1: SL_EXIT_MSG("OpenCV image format CV_32SC1 not supported"); break;
        case CV_32SC2: SL_EXIT_MSG("OpenCV image format CV_32SC2 not supported"); break;
        case CV_32SC3: SL_EXIT_MSG("OpenCV image format CV_32SC3 not supported"); break;
        case CV_32SC4: SL_EXIT_MSG("OpenCV image format CV_32SC4 not supported"); break;
        case CV_32FC1: SL_EXIT_MSG("OpenCV image format CV_32FC1 not supported"); break;
        case CV_32FC2: SL_EXIT_MSG("OpenCV image format CV_32FC2 not supported"); break;
        case CV_32FC3: SL_EXIT_MSG("OpenCV image format CV_32FC3 not supported"); break;
        case CV_32FC4: SL_EXIT_MSG("OpenCV image format CV_32FC4 not supported"); break;
        default: SL_EXIT_MSG("OpenCV image format not supported");
    }
    return PF_unknown;
}
//-----------------------------------------------------------------------------
//! Returns the pixel format as string
SLstring SLCVImage::formatString()
{
    switch (_format)
    {
        case PF_rgb: return SLstring("SL_RGB");
        case PF_rgba: return SLstring("SL_RGBA");
        case PF_bgra: return SLstring("SL_BGRA");
        case PF_red: return SLstring("SL_RED");
        case PF_red_integer: return SLstring("SL_RED_INTEGER");
        case PF_green: return SLstring("SL_GREEN");
        case PF_alpha: return SLstring("SL_BLUE");
        case PF_blue: return SLstring("SL_BLUE");
        case PF_luminance: return SLstring("SL_LUMINANCE");
        case PF_intensity: return SLstring("SL_INTENSITY");
        case PF_rg: return SLstring("SL_RG");
        case PF_rg_integer: return SLstring("SL_RG_INTEGER");
        case PF_luminance_alpha: return SLstring("SL_LUMINANCE_ALPHA");
        case PF_bgr: return SLstring("SL_BGR");
        case PF_rgb_integer: return SLstring("SL_RGB_INTEGER");
        case PF_bgr_integer: return SLstring("SL_BGR_INTEGER");
        case PF_rgba_integer: return SLstring("SL_RGBA_INTEGER");
        case PF_bgra_integer: return SLstring("SL_BGRA_INTEGER");
        default: return SLstring("Unknow pixel format");
    }
}

//-----------------------------------------------------------------------------
//! Save as PNG at a certain compression level (0-9)
/*!Save as PNG at a certain compression level (0-9)
\param filename filename with path and extension
\param compressionLevel compression level 0-9 (default 5)
*/
void SLCVImage::savePNG(const SLstring filename, const SLint compressionLevel)
{  
    SLVint compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(compressionLevel);

    try
    {
        imwrite(filename, _cvMat, compression_params);
    }
    catch (runtime_error& ex)
    {   SLstring msg = "SLCVImage.savePNG: Exception: ";
        msg += ex.what();
        SL_EXIT_MSG(msg.c_str());
    }
}

//-----------------------------------------------------------------------------
//! Save as JPG at a certain compression level (0-100)
/*!Save as JPG at a certain compression level (0-9)
\param filename filename with path and extension
\param compressionLevel compression level 0-100 (default 95)
*/
void SLCVImage::saveJPG(const SLstring filename, const SLint compressionLevel)
{
    SLVint compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(CV_IMWRITE_JPEG_PROGRESSIVE);
    compression_params.push_back(compressionLevel);

    try
    {
        imwrite(filename, _cvMat, compression_params);
    }
    catch (runtime_error& ex)
    {   SLstring msg = "SLCVImage.saveJPG: Exception: ";
        msg += ex.what();
        SL_EXIT_MSG(msg.c_str());
    }
}

//-----------------------------------------------------------------------------
//! getPixeli returns the pixel color at the integer pixel coordinate x, y
/*! Returns the pixel color at the integer pixel coordinate x, y. The color
components range from 0-1 float.
*/
SLCol4f SLCVImage::getPixeli(SLint x, SLint y)
{
    SLCol4f  color;
   
    x %= _cvMat.cols;
    y %= _cvMat.rows;

    switch (_format)
    {   case PF_rgb:
        {   cv::Vec3b c = _cvMat.at<cv::Vec3b>(y, x);
            color.set(c.val[0], c.val[1], c.val[2], 255.0f);
            break;
        }
        case PF_rgba:
        {   cv::Vec4b c = _cvMat.at<cv::Vec4b>(y, x);
            color.set(c.val[0], c.val[1], c.val[2], c.val[3]);
            break;
        }
        case PF_bgra:
        {   cv::Vec4b c = _cvMat.at<cv::Vec4b>(y, x);
            color.set(c.val[2], c.val[1], c.val[0], c.val[3]);
            break;
        }
        #ifdef SL_GLES2
        case PF_luminance:
        #else
        case PF_red:
        #endif
        {   SLuchar c = _cvMat.at<SLuchar>(y, x);
            color.set(c, c, c, 255.0f);
            break;
        }
        #ifdef SL_GLES2
        case PF_luminance_alpha:
        #else
        case PF_rg:
        #endif
        {   cv::Vec2b c = _cvMat.at<cv::Vec2b>(y, x);
            color.set(c.val[0], c.val[0], c.val[0], c.val[1]);
            break;
        }
        default: SL_EXIT_MSG("SLCVImage::getPixeli: Unknown format!");
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
SLCol4f SLCVImage::getPixelf(SLfloat x, SLfloat y)
{     
    // Bilinear interpolation
    SLfloat xf = SL_fract(x) * _cvMat.cols;
    SLfloat yf = SL_fract(y) * _cvMat.rows;

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
void SLCVImage::setPixeli(SLint x, SLint y, SLCol4f color)
{  
    if (x<0) x = 0; 
    if (x>=(SLint)_cvMat.cols) x = _cvMat.cols-1; // 0 <= x < _width
    if (y<0) y = 0; 
    if (y>=(SLint)_cvMat.rows) y = _cvMat.rows-1; // 0 <= y < _height

    SLint R, G, B;

    switch (_format)
    {   case PF_rgb:
            _cvMat.at<cv::Vec3b>(y, x) = cv::Vec3b((SLuchar)(color.r * 255.0f),
                                                   (SLuchar)(color.g * 255.0f),
                                                   (SLuchar)(color.b * 255.0f));
            break;
        case PF_bgr:
            _cvMat.at<cv::Vec3b>(y, x) = cv::Vec3b((SLuchar)(color.b * 255.0f),
                                                   (SLuchar)(color.g * 255.0f),
                                                   (SLuchar)(color.r * 255.0f));
            break;
        case PF_rgba:
            _cvMat.at<cv::Vec4b>(y, x) = cv::Vec4b((SLuchar)(color.r * 255.0f),
                                                   (SLuchar)(color.g * 255.0f),
                                                   (SLuchar)(color.b * 255.0f),
                                                   (SLuchar)(color.a * 255.0f));
            break;
        case PF_bgra:
            _cvMat.at<cv::Vec4b>(y, x) = cv::Vec4b((SLuchar)(color.b * 255.0f),
                                                   (SLuchar)(color.g * 255.0f),
                                                   (SLuchar)(color.r * 255.0f),
                                                   (SLuchar)(color.a * 255.0f));
            break;
        #ifdef SL_GLES2
        case PF_luminance:
        #else
        case PF_red:
        #endif
            R = (SLint)(color.r * 255.0f);
            G = (SLint)(color.g * 255.0f);
            B = (SLint)(color.b * 255.0f);
            _cvMat.at<uchar>(y, x) = (SLubyte)((( 66*R + 129*G +  25*B + 128)>>8) + 16);
            break;
        #ifdef SL_GLES2
        case PF_luminance_alpha:
        #else
        case PF_rg:
        #endif
            R = (SLint)(color.r * 255.0f);
            G = (SLint)(color.g * 255.0f);
            B = (SLint)(color.b * 255.0f);
            _cvMat.at<cv::Vec2b>(y, x) = cv::Vec2b((SLubyte)((( 66*R + 129*G +  25*B + 128)>>8) + 16),
                                                   (SLubyte)(color.a * 255.0f));
            break;
        default: SL_EXIT_MSG("SLCVImage::setPixeli: Unknown format!");
    }
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGB pixel color at the integer pixel coordinate x, y
void SLCVImage::setPixeliRGB(SLint x, SLint y, SLCol3f color)
{  
    assert(_bytesPerPixel==3);
    if (x<0) x = 0;
    if (x>=(SLint)_cvMat.cols) x = _cvMat.cols-1; // 0 <= x < _width
    if (y<0) y = 0;
    if (y>=(SLint)_cvMat.rows) y = _cvMat.rows-1; // 0 <= y < _height

    _cvMat.at<cv::Vec3b>(y, x) = cv::Vec3b((SLuchar)(color.r * 255.0f + 0.5f),
                                           (SLuchar)(color.g * 255.0f + 0.5f),
                                           (SLuchar)(color.b * 255.0f + 0.5f));
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGB pixel color at the integer pixel coordinate x, y
void SLCVImage::setPixeliRGB(SLint x, SLint y, SLCol4f color)
{  
    assert(_bytesPerPixel==3);
    if (x<0) x = 0;
    if (x>=(SLint)_cvMat.cols) x = _cvMat.cols-1; // 0 <= x < _width
    if (y<0) y = 0;
    if (y>=(SLint)_cvMat.rows) y = _cvMat.rows-1; // 0 <= y < _height

    _cvMat.at<cv::Vec3b>(y, x) = cv::Vec3b((SLuchar)(color.r * 255.0f + 0.5f),
                                           (SLuchar)(color.g * 255.0f + 0.5f),
                                           (SLuchar)(color.b * 255.0f + 0.5f));
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGBA pixel color at the integer pixel coordinate x, y
void SLCVImage::setPixeliRGBA(SLint x, SLint y, SLCol4f color)
{  
    assert(_bytesPerPixel==4);
    if (x<0) x = 0;
    if (x>=(SLint)_cvMat.cols) x = _cvMat.cols-1; // 0 <= x < _width
    if (y<0) y = 0;
    if (y>=(SLint)_cvMat.rows) y = _cvMat.rows-1; // 0 <= y < _height

    _cvMat.at<cv::Vec4b>(y, x) = cv::Vec4b((SLuchar)(color.r * 255.0f + 0.5f),
                                           (SLuchar)(color.g * 255.0f + 0.5f),
                                           (SLuchar)(color.b * 255.0f + 0.5f),
                                           (SLuchar)(color.a * 255.0f + 0.5f));
}
//-----------------------------------------------------------------------------
/*!
SLCVImage::Resize does a scaling with bilinear interpolation.
*/
void SLCVImage::resize(SLint width, SLint height)
{  
    assert(_cvMat.cols>0 && _cvMat.rows>0 && width>0 && height>0);
    if (_cvMat.cols==width && _cvMat.rows==height) return;

    SLCVMat dst = SLCVMat(height, width, _cvMat.type());

    cv::resize(_cvMat, dst, dst.size(), 0, 0, CV_INTER_LINEAR);

    _cvMat = dst;
}
//-----------------------------------------------------------------------------
//! Flip Y coordiantes used to make JPEGs from top-left to bottom-left images.
void SLCVImage::flipY()
{  
    if (_cvMat.cols > 0 && _cvMat.rows > 0)
    {  
        SLCVMat dst = SLCVMat(_cvMat.rows, _cvMat.cols, _cvMat.type());
        cv::flip(_cvMat, dst, 0);
        _cvMat = dst;
    }
}
//-----------------------------------------------------------------------------
//! Fills the image with a certain rgb color
void SLCVImage::fill(SLubyte r, SLubyte g, SLubyte b)
{
    switch (_format)
    {   case PF_rgb:
            _cvMat.setTo(cv::Vec3b(r,g,b));
            break;
        case PF_bgr:
            _cvMat.setTo(cv::Vec3b(b,g,r));
            break;
        default: SL_EXIT_MSG("SLCVImage::fill(r,g,b): Wrong format!");
    }
}
//-----------------------------------------------------------------------------
//! Fills the image with a certain rgba color
void SLCVImage::fill(SLubyte r, SLubyte g, SLubyte b, SLubyte a)
{
    switch (_format)
    {   case PF_rgba:
            _cvMat.setTo(cv::Vec4b(r,g,b,a));
            break;
        case PF_bgra:
            _cvMat.setTo(cv::Vec4b(b,g,r,a));
            break;
        default: SL_EXIT_MSG("SLCVImage::fill(r,g,b,a): Wrong format!");
    }
}
//-----------------------------------------------------------------------------
