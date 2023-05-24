//#############################################################################
//   File:      cv/CVImage.cpp
//   Date:      Spring 2017
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <CVImage.h>
#include <Utils.h>
#include <algorithm> // std::max
#include <SLFileStorage.h>

#ifdef __EMSCRIPTEN__
#    define STB_IMAGE_IMPLEMENTATION
#    include "stb_image.h"
#endif

//-----------------------------------------------------------------------------
//! Default constructor
CVImage::CVImage()
{
    _name          = "unknown";
    _format        = PF_unknown;
    _path          = "";
    _bytesPerPixel = 0;
    _bytesPerLine  = 0;
    _bytesPerImage = 0;
    _bytesInFile   = 0;
};
//------------------------------------------------------------------------------
//! Constructor for empty image of a certain format and size
CVImage::CVImage(int             width,
                 int             height,
                 CVPixelFormatGL pixelFormatGL,
                 string          name) : _name(std::move(name))
{
    allocate(width, height, pixelFormatGL);
}
//-----------------------------------------------------------------------------
//! Constructor for image from file
CVImage::CVImage(const string& filename,
                 bool          flipVertical,
                 bool          loadGrayscaleIntoAlpha)
  : _name(Utils::getFileName(filename))
{
    assert(filename != "");
    clearData();
    load(filename, flipVertical, loadGrayscaleIntoAlpha);
}
//-----------------------------------------------------------------------------
//! Copy constructor from a source image
CVImage::CVImage(CVImage& src) : _name(src.name())
{
    assert(src.width() && src.height() && src.data());
    _format        = src.format();
    _path          = src.path();
    _bytesPerPixel = src.bytesPerPixel();
    _bytesPerLine  = src.bytesPerLine();
    _bytesPerImage = src.bytesPerImage();
    _bytesInFile   = src.bytesInFile();
    src.cvMat().copyTo(_cvMat);
}
//-----------------------------------------------------------------------------
//! Creates a 1D image from a CVVec3f color vector
CVImage::CVImage(const CVVVec3f& colors)
{
    allocate((int)colors.size(), 1, PF_rgb);

    int x = 0;
    for (auto color : colors)
    {
        _cvMat.at<CVVec3b>(0, x++) = CVVec3b((uchar)(color[0] * 255.0f),
                                             (uchar)(color[1] * 255.0f),
                                             (uchar)(color[2] * 255.0f));
    }
}
//-----------------------------------------------------------------------------
//! Creates a 1D image from a CVVec4f vector
CVImage::CVImage(const CVVVec4f& colors)
{
    allocate((int)colors.size(), 1, PF_rgba);

    int x = 0;
    for (auto color : colors)
    {
        _cvMat.at<CVVec4b>(0, x++) = CVVec4b((uchar)(color[0] * 255.0f),
                                             (uchar)(color[1] * 255.0f),
                                             (uchar)(color[2] * 255.0f),
                                             (uchar)(color[3] * 255.0f));
    }
}
//-----------------------------------------------------------------------------
CVImage::~CVImage()
{
    // Utils::log("CVImages)", name().c_str());
    clearData();
}
//-----------------------------------------------------------------------------
//! Deletes all data and resets the image parameters
void CVImage::clearData()
{
    _cvMat.release();
    _bytesPerPixel = 0;
    _bytesPerLine  = 0;
    _bytesPerImage = 0;
    _bytesInFile   = 0;
    _path          = "";
}
//-----------------------------------------------------------------------------
//! Memory allocation function
/*! It returns true if width or height or the pixelformat has changed
/param width Width of image in pixels
/param height Height of image in pixels
/param pixelFormatGL OpenGL pixel format enum
/param isContinuous True if the memory is continuous and has no stride bytes at the end of the line
*/
bool CVImage::allocate(int             width,
                       int             height,
                       CVPixelFormatGL pixelFormatGL,
                       bool            isContinuous)
{
    assert(width > 0 && height > 0);

    // return if essentials are identical
    if (!_cvMat.empty() &&
        _cvMat.cols == width &&
        _cvMat.rows == height &&
        _format == pixelFormatGL)
        return false;

    // Set the according OpenCV format
    int cvType = glPixelFormat2cvType(pixelFormatGL);
    _cvMat.create(height, width, cvType);

    _format        = pixelFormatGL;
    _bytesPerPixel = bytesPerPixel(pixelFormatGL);
    _bytesPerLine  = bytesPerLine((uint)width,
                                 pixelFormatGL,
                                 isContinuous);
    _bytesPerImage = _bytesPerLine * (uint)height;

    if (!_cvMat.data)
        Utils::exitMsg("SLProject",
                       "CVImage::Allocate: Allocation failed",
                       __LINE__,
                       __FILE__);
    return true;
}
//-----------------------------------------------------------------------------
//! Returns the NO. of bytes per pixel for the passed pixel format
uint CVImage::bytesPerPixel(CVPixelFormatGL pixFormatGL)
{
    switch (pixFormatGL)
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
        case PF_r16f: return 2;
        case PF_rg16f: return 4;
        case PF_rgb16f: return 6;
        case PF_rgba16f: return 8;
        case PF_r32f: return 4;
        case PF_rg32f: return 8;
        case PF_rgb32f: return 12;
        case PF_rgba32f: return 16;
        default:
            Utils::exitMsg("SLProject", "CVImage::bytesPerPixel: unknown pixel format", __LINE__, __FILE__);
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
uint CVImage::bytesPerLine(uint            width,
                           CVPixelFormatGL format,
                           bool            isContinuous)
{
    uint bpp          = bytesPerPixel(format);
    uint bitsPerPixel = bpp * 8;
    uint bpl          = isContinuous ? width * bpp : ((width * bitsPerPixel + 31) / 32) * 4;
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
bool CVImage::load(int             width,
                   int             height,
                   CVPixelFormatGL srcPixelFormatGL,
                   CVPixelFormatGL dstPixelFormatGL,
                   uchar*          data,
                   bool            isContinuous,
                   bool            isTopLeft)
{

    bool needsTextureRebuild = allocate(width,
                                        height,
                                        dstPixelFormatGL,
                                        false);
    uint dstBPL              = _bytesPerLine;
    uint dstBPP              = _bytesPerPixel;
    uint srcBPP              = bytesPerPixel(srcPixelFormatGL);
    uint srcBPL              = bytesPerLine((uint)width, srcPixelFormatGL, isContinuous);

    if (isTopLeft)
    {
        // copy lines and flip vertically
        uchar* dstStart = _cvMat.data + _bytesPerImage - dstBPL;
        uchar* srcStart = data;

        if (srcPixelFormatGL == dstPixelFormatGL)
        {
            for (int h = 0; h < _cvMat.rows; ++h, srcStart += srcBPL, dstStart -= dstBPL)
            {
                memcpy(dstStart, srcStart, (unsigned long)dstBPL);
            }
        }
        else
        {
            if (srcPixelFormatGL == PF_bgra)
            {
                if (dstPixelFormatGL == PF_rgb)
                {
                    for (int h = 0; h < _cvMat.rows; ++h, srcStart += srcBPL, dstStart -= dstBPL)
                    {
                        uchar* src = srcStart;
                        uchar* dst = dstStart;
                        for (int w = 0; w < _cvMat.cols - 1; ++w, dst += dstBPP, src += srcBPP)
                        {
                            dst[0] = src[2];
                            dst[1] = src[1];
                            dst[2] = src[0];
                        }
                    }
                }
                else if (dstPixelFormatGL == PF_rgba)
                {
                    for (int h = 0; h < _cvMat.rows; ++h, srcStart += srcBPL, dstStart -= dstBPL)
                    {
                        uchar* src = srcStart;
                        uchar* dst = dstStart;
                        for (int w = 0; w < _cvMat.cols - 1; ++w, dst += dstBPP, src += srcBPP)
                        {
                            dst[0] = src[2];
                            dst[1] = src[1];
                            dst[2] = src[0];
                            dst[3] = src[3];
                        }
                    }
                }
            }
            else if (srcPixelFormatGL == PF_bgr || srcPixelFormatGL == PF_rgb)
            {
                if (dstPixelFormatGL == PF_rgb || dstPixelFormatGL == PF_bgr)
                {
                    for (int h = 0; h < _cvMat.rows; ++h, srcStart += srcBPL, dstStart -= dstBPL)
                    {
                        uchar* src = srcStart;
                        uchar* dst = dstStart;
                        for (int w = 0; w < _cvMat.cols - 1; ++w, dst += dstBPP, src += srcBPP)
                        {
                            dst[0] = src[2];
                            dst[1] = src[1];
                            dst[2] = src[0];
                        }
                    }
                }
            }
            else
            {
                std::cout << "CVImage::load from memory: Pixel format conversion not allowed" << std::endl;
                exit(1);
            }
        }
    }
    else // bottom left (no flipping)
    {
        if (srcPixelFormatGL == dstPixelFormatGL)
        {
            memcpy(_cvMat.data, data, (unsigned long)_bytesPerImage);
        }
        else
        {
            uchar* dstStart = _cvMat.data;
            uchar* srcStart = data;

            if (srcPixelFormatGL == PF_bgra)
            {
                if (dstPixelFormatGL == PF_rgb)
                {
                    for (int h = 0; h < _cvMat.rows - 1; ++h, srcStart += srcBPL, dstStart += dstBPL)
                    {
                        uchar* src = srcStart;
                        uchar* dst = dstStart;
                        for (int w = 0; w < _cvMat.cols - 1; ++w, dst += dstBPP, src += srcBPP)
                        {
                            dst[0] = src[2];
                            dst[1] = src[1];
                            dst[2] = src[0];
                        }
                    }
                }
                else if (dstPixelFormatGL == PF_rgba)
                {
                    for (int h = 0; h < _cvMat.rows - 1; ++h, srcStart += srcBPL, dstStart += dstBPL)
                    {
                        uchar* src = srcStart;
                        uchar* dst = dstStart;
                        for (int w = 0; w < _cvMat.cols - 1; ++w, dst += dstBPP, src += srcBPP)
                        {
                            dst[0] = src[2];
                            dst[1] = src[1];
                            dst[2] = src[0];
                            dst[3] = src[3];
                        }
                    }
                }
            }
            else if (srcPixelFormatGL == PF_bgr || srcPixelFormatGL == PF_rgb)
            {
                if (dstPixelFormatGL == PF_rgb || dstPixelFormatGL == PF_bgr)
                {
                    for (int h = 0; h < _cvMat.rows - 1; ++h, srcStart += srcBPL, dstStart += dstBPL)
                    {
                        uchar* src = srcStart;
                        uchar* dst = dstStart;
                        for (int w = 0; w < _cvMat.cols - 1; ++w, dst += dstBPP, src += srcBPP)
                        {
                            dst[0] = src[2];
                            dst[1] = src[1];
                            dst[2] = src[0];
                        }
                    }
                }
            }
            else
            {
                std::cout << "CVImage::load from memory: Pixel format conversion not allowed" << std::endl;
                exit(1);
            }
        }
    }

    return needsTextureRebuild;
}
//-----------------------------------------------------------------------------
//! Loads the image with the appropriate image loader
void CVImage::load(const string& filename,
                   bool          flipVertical,
                   bool          loadGrayscaleIntoAlpha)
{
    string ext   = Utils::getFileExt(filename);
    _name        = Utils::getFileName(filename);
    _path        = Utils::getPath(filename);
    _bytesInFile = Utils::getFileSize(filename);

    // load the image format as stored in the file

    SLIOBuffer buffer   = SLFileStorage::readIntoBuffer(filename, IOK_image);
    CVMat      inputMat = CVMat(1, (int)buffer.size, CV_8UC1, buffer.data);

#ifndef __EMSCRIPTEN__
    _cvMat = cv::imdecode(inputMat, cv::ImreadModes::IMREAD_UNCHANGED);
#else
    unsigned char* encodedData = buffer.data;
    int            size        = (int)buffer.size;
    int            width;
    int            height;
    int            numChannels;
    stbi_hdr_to_ldr_gamma(1.0f);
    unsigned char* data = stbi_load_from_memory(encodedData, size, &width, &height, &numChannels, 0);
    _cvMat              = CVMat(height, width, CV_8UC(numChannels), data);
#endif

    SLFileStorage::deleteBuffer(buffer);

    if (!_cvMat.data)
    {
        string msg = "CVImage.load: Loading failed: " + filename;
        Utils::exitMsg("SLProject", msg.c_str(), __LINE__, __FILE__);
    }

#ifndef __EMSCRIPTEN__
    // Convert greater component depth than 8 bit to 8 bit, only if the image is not HDR
    if (_cvMat.depth() > CV_8U && ext != "hdr")
        _cvMat.convertTo(_cvMat, CV_8U, 1.0 / 256.0);

    _format        = cvType2glPixelFormat(_cvMat.type());
    _bytesPerPixel = bytesPerPixel(_format);

    // OpenCV always loads with BGR(A) but some OpenGL prefer RGB(A)
    if (_format == PF_bgr || _format == PF_rgb32f)
    {
        string typeStr1 = typeString(_cvMat.type());
        cv::cvtColor(_cvMat, _cvMat, cv::COLOR_BGR2RGB);
        string typeStr2 = typeString(_cvMat.type());
        _format         = PF_rgb;
    }
    else if (_format == PF_bgra)
    {
        cv::cvtColor(_cvMat, _cvMat, cv::COLOR_BGRA2RGBA);
        _format = PF_rgba;
    }
    else if (_format == PF_red && loadGrayscaleIntoAlpha)
    {
        CVMat rgbaImg;
        rgbaImg.create(_cvMat.rows, _cvMat.cols, CV_8UC4);

        // Copy grayscale into alpha channel
        for (int y = 0; y < rgbaImg.rows; ++y)
        {
            uchar* dst = rgbaImg.ptr<uchar>(y);
            uchar* src = _cvMat.ptr<uchar>(y);

            for (int x = 0; x < rgbaImg.cols; ++x)
            {
                *dst++ = 0;      // B
                *dst++ = 0;      // G
                *dst++ = 0;      // R
                *dst++ = *src++; // A
            }
        }

        _cvMat = rgbaImg;
        cv::cvtColor(_cvMat, _cvMat, cv::COLOR_BGRA2RGBA);
        _format = PF_rgba;

        // for debug check
        // string pathfilename = _path + name();
        // string filename = Utils::getFileNameWOExt(pathfilename);
        // savePNG(_path + filename + "_InAlpha.png");
    }
#else

    switch (_cvMat.type())
    {
        case CV_8UC1: _format = PF_red; break;
        case CV_8UC2: _format = PF_rg; break;
        case CV_8UC3: _format = PF_rgb; break;
        case CV_8UC4: _format = PF_rgba; break;
    }

    _bytesPerPixel = bytesPerPixel(_format);

    if (_format == PF_red && loadGrayscaleIntoAlpha)
    {
        CVMat rgbaImg;
        rgbaImg.create(_cvMat.rows, _cvMat.cols, CV_8UC4);

        // Copy grayscale into alpha channel
        for (int y = 0; y < rgbaImg.rows; ++y)
        {
            uchar* dst = rgbaImg.ptr<uchar>(y);
            uchar* src = _cvMat.ptr<uchar>(y);

            for (int x = 0; x < rgbaImg.cols; ++x)
            {
                uchar value = *src++;
                *dst++      = value;
                *dst++      = value;
                *dst++      = value;
                *dst++      = value;
            }
        }

        _format = PF_rgba;
    }
#endif

    _bytesPerLine  = bytesPerLine((uint)_cvMat.cols, _format, _cvMat.isContinuous());
    _bytesPerImage = _bytesPerLine * (uint)_cvMat.rows;

    // OpenCV loads top-left but OpenGL is bottom left
    if (flipVertical)
        flipY();
}
//-----------------------------------------------------------------------------
//! Converts OpenCV mat type to OpenGL pixel format
CVPixelFormatGL CVImage::cvType2glPixelFormat(int cvType)
{
    switch (cvType)
    {
        case CV_8UC1: return PF_red;
        case CV_8UC2: return PF_rg;
        case CV_8UC3: return PF_bgr;
        case CV_8UC4: return PF_bgra;
        case CV_16FC1: return PF_r16f;
        case CV_16FC2: return PF_rg16f;
        case CV_16FC3: return PF_rgb16f;
        case CV_16FC4: return PF_rgba16f;
        case CV_32FC1: return PF_r32f;
        case CV_32FC2: return PF_rg32f;
        case CV_32FC3: return PF_rgb32f;
        case CV_32FC4: return PF_rgba32f;
        case CV_8SC1: Utils::exitMsg("SLProject", "OpenCV image format CV_8SC1 not supported", __LINE__, __FILE__); break;
        case CV_8SC2: Utils::exitMsg("SLProject", "OpenCV image format CV_8SC2 not supported", __LINE__, __FILE__); break;
        case CV_8SC3: Utils::exitMsg("SLProject", "OpenCV image format CV_8SC3 not supported", __LINE__, __FILE__); break;
        case CV_8SC4: Utils::exitMsg("SLProject", "OpenCV image format CV_8SC4 not supported", __LINE__, __FILE__); break;
        case CV_16UC1: Utils::exitMsg("SLProject", "OpenCV image format CV_16UC1 not supported", __LINE__, __FILE__); break;
        case CV_16UC2: Utils::exitMsg("SLProject", "OpenCV image format CV_16UC2 not supported", __LINE__, __FILE__); break;
        case CV_16UC3: Utils::exitMsg("SLProject", "OpenCV image format CV_16UC3 not supported", __LINE__, __FILE__); break;
        case CV_16UC4: Utils::exitMsg("SLProject", "OpenCV image format CV_16UC4 not supported", __LINE__, __FILE__); break;
        case CV_16SC1: Utils::exitMsg("SLProject", "OpenCV image format CV_16SC1 not supported", __LINE__, __FILE__); break;
        case CV_16SC2: Utils::exitMsg("SLProject", "OpenCV image format CV_16SC2 not supported", __LINE__, __FILE__); break;
        case CV_16SC3: Utils::exitMsg("SLProject", "OpenCV image format CV_16SC3 not supported", __LINE__, __FILE__); break;
        case CV_16SC4: Utils::exitMsg("SLProject", "OpenCV image format CV_16SC4 not supported", __LINE__, __FILE__); break;
        case CV_32SC1: Utils::exitMsg("SLProject", "OpenCV image format CV_32SC1 not supported", __LINE__, __FILE__); break;
        case CV_32SC2: Utils::exitMsg("SLProject", "OpenCV image format CV_32SC2 not supported", __LINE__, __FILE__); break;
        case CV_32SC3: Utils::exitMsg("SLProject", "OpenCV image format CV_32SC3 not supported", __LINE__, __FILE__); break;
        case CV_32SC4: Utils::exitMsg("SLProject", "OpenCV image format CV_32SC4 not supported", __LINE__, __FILE__); break;
        default: Utils::exitMsg("SLProject",
                                "glPixelFormat2cvType: OpenCV image format not supported",
                                __LINE__,
                                __FILE__);
    }
    return PF_unknown;
}
//-----------------------------------------------------------------------------
//! Converts OpenGL pixel format to OpenCV mat type
int CVImage::glPixelFormat2cvType(CVPixelFormatGL pixelFormatGL)
{
    switch (pixelFormatGL)
    {
        case PF_red:
        case PF_luminance: return CV_8UC1;
        case PF_luminance_alpha:
        case PF_rg: return CV_8UC2;
        case PF_rgb: return CV_8UC3;
        case PF_rgba: return CV_8UC4;
        case PF_bgr: return CV_8UC3;
        case PF_bgra: return CV_8UC4;
        case PF_r16f: return CV_16FC1;
        case PF_rg16f: return CV_16FC2;
        case PF_rgb16f: return CV_16FC3;
        case PF_rgba16f: return CV_16FC4;
        case PF_r32f: return CV_32FC1;
        case PF_rg32f: return CV_32FC2;
        case PF_rgb32f: return CV_32FC3;
        case PF_rgba32f: return CV_32FC4;
        default:
        {
            string msg = "glPixelFormat2cvType: OpenGL pixel format not supported: ";
            msg += formatString(pixelFormatGL);
            Utils::exitMsg("SLProject", msg.c_str(), __LINE__, __FILE__);
        }
    }
    return -1;
}
//-----------------------------------------------------------------------------
//! Returns the pixel format as string
string CVImage::formatString(CVPixelFormatGL pixelFormatGL)
{
    switch (pixelFormatGL)
    {
        case PF_unknown: return "PF_unknown";
        case PF_yuv_420_888: return "PF_yuv_420_888";
        case PF_alpha: return "PF_alpha";
        case PF_luminance: return "PF_luminance";
        case PF_luminance_alpha: return "PF_luminance_alpha";
        case PF_intensity: return "PF_intensity";
        case PF_green: return "PF_green";
        case PF_blue: return "PF_blue";
        case PF_depth_component: return "PF_depth_component";
        case PF_red: return "PF_red";
        case PF_rg: return "PF_rg";
        case PF_rgb: return "PF_rgb";
        case PF_rgba: return "PF_rgba";
        case PF_bgr: return "PF_bgr";
        case PF_bgra: return "PF_bgra";
        case PF_rg_integer: return "PF_rg_integer";
        case PF_red_integer: return "PF_red_integer";
        case PF_rgb_integer: return "PF_rgb_integer";
        case PF_rgba_integer: return "PF_rgba_integer";
        case PF_bgr_integer: return "PF_bgr_integer";
        case PF_bgra_integer: return "PF_bgra_integer";
        case PF_r16f: return "PF_r16f";
        case PF_r32f: return "PF_r32f";
        case PF_rg16f: return "PF_rg16f";
        case PF_rg32f: return "PF_rg32f";
        case PF_rgba32f: return "PF_rgba32f";
        case PF_rgba16f: return "PF_rgba16f";
        case PF_rgb16f: return "PF_rgb16f";
        default: return string("Unknown pixel format");
    }
}
//-----------------------------------------------------------------------------
//! Save as PNG at a certain compression level (0-9)
/*!Save as PNG at a certain compression level (0-9)
@param filename filename with path and extension
@param compressionLevel compression level 0-9 (default 6)
@param flipY Flag for vertical mirroring
@param convertBGR2RGB Flag for BGR to RGB conversion
*/
void CVImage::savePNG(const string& filename,
                      const int     compressionLevel,
                      const bool    flipY,
                      const bool    convertToRGB)
{
#ifndef __EMSCRIPTEN__
    vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(compressionLevel);

    try
    {
        CVMat outImg = _cvMat.clone();

        if (flipY)
            cv::flip(outImg, outImg, 0);
        if (convertToRGB)
            cv::cvtColor(outImg, outImg, cv::COLOR_BGR2RGB);

        imwrite(filename, outImg, compression_params);
    }
    catch (std::runtime_error& ex)
    {
        string msg = "CVImage.savePNG: Exception: ";
        msg += ex.what();
        Utils::exitMsg("SLProject", msg.c_str(), __LINE__, __FILE__);
    }
#endif
}

//-----------------------------------------------------------------------------
//! Save as JPG at a certain compression level (0-100)
/*!Save as JPG at a certain compression level (0-100)
@param filename filename with path and extension
@param compressionLevel compression level 0-100 (default 95)
@param flipY Flag for vertical mirroring
@param convertBGR2RGB Flag for BGR to RGB conversion
*/
void CVImage::saveJPG(const string& filename,
                      const int     compressionLevel,
                      const bool    flipY,
                      const bool    convertBGR2RGB)
{
#ifndef __EMSCRIPTEN__
    vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(cv::IMWRITE_JPEG_PROGRESSIVE);
    compression_params.push_back(compressionLevel);

    try
    {
        CVMat outImg = _cvMat.clone();

        if (flipY)
            cv::flip(outImg, outImg, 0);
        if (convertBGR2RGB)
            cv::cvtColor(outImg, outImg, cv::COLOR_BGR2RGB);

        imwrite(filename, outImg, compression_params);
    }
    catch (std::runtime_error& ex)
    {
        string msg = "CVImage.saveJPG: Exception: ";
        msg += ex.what();
        Utils::exitMsg("SLProject", msg.c_str(), __LINE__, __FILE__);
    }
#endif
}
//-----------------------------------------------------------------------------
//! getPixeli returns the pixel color at the integer pixel coordinate x, y
/*! Returns the pixel color at the integer pixel coordinate x, y. The color
components range from 0-1 float.
*/
CVVec4f CVImage::getPixeli(int x, int y)
{
    CVVec4f color;

    x %= _cvMat.cols;
    y %= _cvMat.rows;

    switch (_format)
    {
        case PF_rgb:
        {
            CVVec3b c = _cvMat.at<CVVec3b>(y, x);
            color[0]  = c[0];
            color[1]  = c[1];
            color[2]  = c[2];
            color[3]  = 255.0f;
            break;
        }
        case PF_rgba:
        {
            color = _cvMat.at<CVVec4b>(y, x);
            break;
        }
        case PF_bgr:
        {
            CVVec3b c = _cvMat.at<CVVec3b>(y, x);
            color[0]  = c[2];
            color[1]  = c[1];
            color[2]  = c[0];
            color[3]  = 255.0f;
            break;
        }
        case PF_bgra:
        {
            CVVec4b c = _cvMat.at<CVVec4b>(y, x);
            color[0]  = c[2];
            color[1]  = c[1];
            color[2]  = c[0];
            color[3]  = c[3];
            break;
        }
#ifdef APP_USES_GLES
        case PF_luminance:
#else
        case PF_red:
#endif
        {
            uchar c  = _cvMat.at<uchar>(y, x);
            color[0] = (float)c;
            color[1] = (float)c;
            color[2] = (float)c;
            color[3] = 255.0f;
            break;
        }
#ifdef APP_USES_GLES
        case PF_luminance_alpha:
#else
        case PF_rg:
#endif
        {
            CVVec2b c = _cvMat.at<cv::Vec2b>(y, x);
            color[0]  = c[0];
            color[1]  = c[0];
            color[2]  = c[0];
            color[3]  = c[1];
            break;
        }
        default: Utils::exitMsg("SLProject", "CVImage::getPixeli: Unknown format!", __LINE__, __FILE__);
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
CVVec4f CVImage::getPixelf(float x, float y)
{
    // Bilinear interpolation
    float xf = Utils::fract(x) * _cvMat.cols;
    float yf = Utils::fract(y) * _cvMat.rows;

    // corrected fractional parts
    float fracX = Utils::fract(xf);
    float fracY = Utils::fract(yf);
    fracX -= Utils::sign(fracX - 0.5f) * 0.5f;
    fracY -= Utils::sign(fracY - 0.5f) * 0.5f;

    // calculate area weights of the four neighbouring texels
    float X1 = 1.0f - fracX;
    float Y1 = 1.0f - fracY;
    float UL = X1 * Y1;
    float UR = fracX * Y1;
    float LL = X1 * fracY;
    float LR = fracX * fracY;

    // get the color of the four neighbouring texels
    // int xm, xp, ym, yp;
    // Fast2Int(&xm, xf-1.0f);
    // Fast2Int(&ym, yf-1.0f);
    // Fast2Int(&xp, xf);
    // Fast2Int(&yp, yf);
    //
    // SLCol4f cUL = getPixeli(xm,ym);
    // SLCol4f cUR = getPixeli(xp,ym);
    // SLCol4f cLL = getPixeli(xm,yp);
    // SLCol4f cLR = getPixeli(xp,yp);

    CVVec4f cUL = getPixeli((int)(xf - 0.5f), (int)(yf - 0.5f));
    CVVec4f cUR = getPixeli((int)(xf + 0.5f), (int)(yf - 0.5f));
    CVVec4f cLL = getPixeli((int)(xf - 0.5f), (int)(yf + 0.5f));
    CVVec4f cLR = getPixeli((int)(xf + 0.5f), (int)(yf + 0.5f));

    // calculate a new interpolated color with the area weights
    float r = UL * cUL[0] + LL * cLL[0] + UR * cUR[0] + LR * cLR[0];
    float g = UL * cUL[1] + LL * cLL[1] + UR * cUR[1] + LR * cLR[1];
    float b = UL * cUL[2] + LL * cLL[2] + UR * cUR[2] + LR * cLR[2];

    return CVVec4f(r, g, b, 1.0f);
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGB pixel color at the integer pixel coordinate x, y
void CVImage::setPixeli(int x, int y, CVVec4f color)
{
    if (x < 0) x = 0;
    if (x >= (int)_cvMat.cols) x = _cvMat.cols - 1; // 0 <= x < _width
    if (y < 0) y = 0;
    if (y >= (int)_cvMat.rows) y = _cvMat.rows - 1; // 0 <= y < _height

    int R, G, B;

    switch (_format)
    {
        case PF_rgb:
            _cvMat.at<CVVec3b>(y, x) = CVVec3b((uchar)(color[0] * 255.0f),
                                               (uchar)(color[1] * 255.0f),
                                               (uchar)(color[2] * 255.0f));
            break;
        case PF_bgr:
            _cvMat.at<CVVec3b>(y, x) = CVVec3b((uchar)(color[2] * 255.0f),
                                               (uchar)(color[1] * 255.0f),
                                               (uchar)(color[0] * 255.0f));
            break;
        case PF_rgba:
            _cvMat.at<CVVec4b>(y, x) = CVVec4b((uchar)(color[0] * 255.0f),
                                               (uchar)(color[1] * 255.0f),
                                               (uchar)(color[2] * 255.0f),
                                               (uchar)(color[3] * 255.0f));
            break;
        case PF_bgra:
            _cvMat.at<CVVec4b>(y, x) = CVVec4b((uchar)(color[2] * 255.0f),
                                               (uchar)(color[1] * 255.0f),
                                               (uchar)(color[0] * 255.0f),
                                               (uchar)(color[3] * 255.0f));
            break;
#ifdef APP_USES_GLES
        case PF_luminance:
#else
        case PF_red:
#endif
            R                      = (int)(color[0] * 255.0f);
            G                      = (int)(color[1] * 255.0f);
            B                      = (int)(color[2] * 255.0f);
            _cvMat.at<uchar>(y, x) = (uchar)(((66 * R + 129 * G + 25 * B + 128) >> 8) + 16);
            break;
#ifdef APP_USES_GLES
        case PF_luminance_alpha:
#else
        case PF_rg:
#endif
            R                        = (int)(color[0] * 255.0f);
            G                        = (int)(color[1] * 255.0f);
            B                        = (int)(color[2] * 255.0f);
            _cvMat.at<CVVec2b>(y, x) = CVVec2b((uchar)(((66 * R + 129 * G + 25 * B + 128) >> 8) + 16),
                                               (uchar)(color[3] * 255.0f));
            break;
        default: Utils::exitMsg("SLProject", "CVImage::setPixeli: Unknown format!", __LINE__, __FILE__);
    }
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGB pixel color at the integer pixel coordinate x, y
void CVImage::setPixeliRGB(int x, int y, CVVec3f color)
{
    assert(_bytesPerPixel == 3);
    if (x < 0) x = 0;
    if (x >= (int)_cvMat.cols) x = _cvMat.cols - 1; // 0 <= x < _width
    if (y < 0) y = 0;
    if (y >= (int)_cvMat.rows) y = _cvMat.rows - 1; // 0 <= y < _height

    _cvMat.at<CVVec3b>(y, x) = CVVec3b((uchar)(color[0] * 255.0f + 0.5f),
                                       (uchar)(color[1] * 255.0f + 0.5f),
                                       (uchar)(color[2] * 255.0f + 0.5f));
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGB pixel color at the integer pixel coordinate x, y
void CVImage::setPixeliRGB(int x, int y, CVVec4f color)
{
    assert(_bytesPerPixel == 3);
    if (x < 0) x = 0;
    if (x >= (int)_cvMat.cols) x = _cvMat.cols - 1; // 0 <= x < _width
    if (y < 0) y = 0;
    if (y >= (int)_cvMat.rows) y = _cvMat.rows - 1; // 0 <= y < _height

    _cvMat.at<CVVec3b>(y, x) = CVVec3b((uchar)(color[0] * 255.0f + 0.5f),
                                       (uchar)(color[1] * 255.0f + 0.5f),
                                       (uchar)(color[2] * 255.0f + 0.5f));
}
//-----------------------------------------------------------------------------
//! setPixeli sets the RGBA pixel color at the integer pixel coordinate x, y
void CVImage::setPixeliRGBA(int x, int y, CVVec4f color)
{
    assert(_bytesPerPixel == 4);
    if (x < 0) x = 0;
    if (x >= (int)_cvMat.cols) x = _cvMat.cols - 1; // 0 <= x < _width
    if (y < 0) y = 0;
    if (y >= (int)_cvMat.rows) y = _cvMat.rows - 1; // 0 <= y < _height

    _cvMat.at<CVVec4b>(y, x) = CVVec4b((uchar)(color[0] * 255.0f + 0.5f),
                                       (uchar)(color[1] * 255.0f + 0.5f),
                                       (uchar)(color[2] * 255.0f + 0.5f),
                                       (uchar)(color[3] * 255.0f + 0.5f));
}
//-----------------------------------------------------------------------------
/*!
CVImage::Resize does a scaling with bilinear interpolation.
*/
void CVImage::resize(int width, int height)
{
    assert(_cvMat.cols > 0 && _cvMat.rows > 0 && width > 0 && height > 0);
    if (_cvMat.cols == width && _cvMat.rows == height) return;

    CVMat dst = CVMat(height, width, _cvMat.type());

    cv::resize(_cvMat, dst, dst.size(), 0, 0, cv::INTER_LINEAR);

    _cvMat = dst;
}
//-----------------------------------------------------------------------------
//! Converts the data type of the cvMat
void CVImage::convertTo(int cvDataType)
{
    _cvMat.convertTo(_cvMat, cvDataType);
    _format        = cvType2glPixelFormat(cvDataType);
    _bytesPerPixel = bytesPerPixel(_format);
    _bytesPerLine  = bytesPerLine((uint)_cvMat.cols,
                                 _format,
                                 _cvMat.isContinuous());
    _bytesPerImage = _bytesPerLine * (uint)_cvMat.rows;
}
//-----------------------------------------------------------------------------
//! Flip X coordinates used to make JPEGs from top-left to bottom-left images.
void CVImage::flipX()
{
    if (_cvMat.cols > 0 && _cvMat.rows > 0)
    {
        CVMat dst = CVMat(_cvMat.rows, _cvMat.cols, _cvMat.type());
        cv::flip(_cvMat, dst, 1);
        _cvMat = dst;
    }
}
//-----------------------------------------------------------------------------
//! Flip Y coordinates used to make JPEGs from top-left to bottom-left images.
void CVImage::flipY()
{
    if (_cvMat.cols > 0 && _cvMat.rows > 0)
    {
        CVMat dst = CVMat(_cvMat.rows, _cvMat.cols, _cvMat.type());
        cv::flip(_cvMat, dst, 0);
        _cvMat = dst;
    }
}
//-----------------------------------------------------------------------------
//! Fills the image with a certain rgb color
void CVImage::fill(uchar r, uchar g, uchar b)
{
    switch (_format)
    {
        case PF_rgb:
            _cvMat.setTo(CVVec3b(r, g, b));
            break;
        case PF_bgr:
            _cvMat.setTo(CVVec3b(b, g, r));
            break;
        default: Utils::exitMsg("SLProject", "CVImage::fill(r,g,b): Wrong format!", __LINE__, __FILE__);
    }
}
//-----------------------------------------------------------------------------
//! Fills the image with a certain rgba color
void CVImage::fill(uchar r, uchar g, uchar b, uchar a)
{
    switch (_format)
    {
        case PF_rgba:
            _cvMat.setTo(CVVec4b(r, g, b, a));
            break;
        case PF_bgra:
            _cvMat.setTo(CVVec4b(b, g, r, a));
            break;
        default: Utils::exitMsg("SLProject", "CVImage::fill(r,g,b,a): Wrong format!", __LINE__, __FILE__);
    }
}
//-----------------------------------------------------------------------------
void CVImage::crop(float targetWdivH, int& cropW, int& cropH)
{

    float inWdivH = (float)_cvMat.cols / (float)_cvMat.rows;
    // viewportWdivH is negative the viewport aspect will be the same
    float outWdivH = targetWdivH < 0.0f ? inWdivH : targetWdivH;

    cropH = 0; // crop height in pixels of the source image
    cropW = 0; // crop width in pixels of the source image
    if (Utils::abs(inWdivH - outWdivH) > 0.01f)
    {
        int width  = 0; // width in pixels of the destination image
        int height = 0; // height in pixels of the destination image
        int wModulo4;
        int hModulo4;

        if (inWdivH > outWdivH) // crop input image left & right
        {
            width  = (int)((float)_cvMat.rows * outWdivH);
            height = _cvMat.rows;
            cropW  = (int)((float)(_cvMat.cols - width) * 0.5f);

            // Width must be devidable by 4
            wModulo4 = width % 4;
            if (wModulo4 == 1) width--;
            if (wModulo4 == 2)
            {
                cropW++;
                width -= 2;
            }
            if (wModulo4 == 3) width++;
        }
        else // crop input image at top & bottom
        {
            width  = _cvMat.cols;
            height = (int)((float)_cvMat.cols / outWdivH);
            cropH  = (int)((float)(_cvMat.rows - height) * 0.5f);

            // Height must be devidable by 4
            hModulo4 = height % 4;
            if (hModulo4 == 1) height--;
            if (hModulo4 == 2)
            {
                cropH++;
                height -= 2;
            }
            if (hModulo4 == 3) height++;
        }

        _cvMat(cv::Rect(cropW, cropH, width, height)).copyTo(_cvMat);
        // imwrite("AfterCropping.bmp", lastFrame);
    }
}
//-----------------------------------------------------------------------------
//! Returns the cv::Mat.type()) as string
string CVImage::typeString(int cvMatTypeInt)
{
    // 7 base types, with five channel options each (none or C1, ..., C4)
    // clang-format off
    int numImgTypes = 35;
    int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                             CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                             CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                             CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                             CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                             CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                             CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

    string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                             "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                             "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                             "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                             "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                             "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                             "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};
    // clang-format on

    for (int i = 0; i < numImgTypes; i++)
    {
        if (cvMatTypeInt == enum_ints[i]) return enum_strings[i];
    }
    return "unknown image type";
}
//-----------------------------------------------------------------------------
