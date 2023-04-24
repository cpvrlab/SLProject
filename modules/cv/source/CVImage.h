//#############################################################################
//  File:      cv/CVImage.h
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVIMAGE_H
#define CVIMAGE_H

#include <CVTypedefs.h>

using std::string;

//-----------------------------------------------------------------------------
//! Pixel format according to OpenGL pixel format defines
/*!
 * This is a pretty delicate topic in OpenGL because bugs in GLSL due to pixel
 * format errors are very hard to detect and debug.
 */
enum CVPixelFormatGL
{
    PF_unknown         = 0,
    PF_yuv_420_888     = 1,      // YUV format from Android not supported in GL
    PF_alpha           = 0x1906, // ES2 ES3 GL2
    PF_luminance       = 0x1909, // ES2 ES3 GL2
    PF_luminance_alpha = 0x190A, // ES2 ES3 GL2
    PF_intensity       = 0x8049, //         GL2
    PF_green           = 0x1904, //         GL2
    PF_blue            = 0x1905, //         GL2
    PF_depth_component = 0x1902, //     ES3 GL2     GL4
    PF_red             = 0x1903, //     ES3 GL2 GL3 GL4
    PF_rg              = 0x8227, //     ES3     GL3 GL4
    PF_rgb             = 0x1907, // ES2 ES3 GL2 GL3 GL4
    PF_rgba            = 0x1908, // ES2 ES3 GL2 GL3 GL4
    PF_bgr             = 0x80E0, //         GL2 GL3 GL4
    PF_bgra            = 0x80E1, //         GL2 GL3 GL4
    PF_rg_integer      = 0x8228, //     ES3         GL4
    PF_red_integer     = 0x8D94, //     ES3         GL4
    PF_rgb_integer     = 0x8D98, //     ES3         GL4
    PF_rgba_integer    = 0x8D99, //     ES3         GL4
    PF_bgr_integer     = 0x8D9A, //                 GL4
    PF_bgra_integer    = 0x8D9B, //                 GL4
    PF_r16f            = 0x822D,
    PF_r32f            = 0x822E, //     ES3     GL3 GL4
    PF_rg16f           = 0x822F,
    PF_rg32f           = 0x8230,
    PF_rgba32f         = 0x8814,
    PF_rgb32f          = 0x8815,
    PF_rgba16f         = 0x881A,
    PF_rgb16f          = 0x881B
};
//-----------------------------------------------------------------------------
//! OpenCV image class with the same interface as the former SLImage class
/*! The core object is the OpenCV matrix _cvMat. Be aware the OpenCV accesses its
matrix of type mat often by row and columns. In that order it corresponds to
the y and x coordinates and not x and y as we are used to!
See the OpenCV docs for more information:
http://docs.opencv.org/2.4.10/modules/core/doc/basic_structures.html#mat
*/
class CVImage
{
public:
    CVImage();
    CVImage(int             width,
            int             height,
            CVPixelFormatGL format,
            string          name);
    explicit CVImage(const string& imageFilename,
                     bool          flipVertical           = true,
                     bool          loadGrayscaleIntoAlpha = false);
    CVImage(CVImage& srcImage);
    explicit CVImage(const CVVVec3f& colors);
    explicit CVImage(const CVVVec4f& colors);
    ~CVImage();

    // Misc
    void                   clearData();
    bool                   allocate(int             width,
                                    int             height,
                                    CVPixelFormatGL pixelFormatGL,
                                    bool            isContinuous = true);
    void                   load(const string& filename,
                                bool          flipVertical           = true,
                                bool          loadGrayscaleIntoAlpha = false);
    bool                   load(int             inWidth,
                                int             inHeight,
                                CVPixelFormatGL srcPixelFormatGL,
                                CVPixelFormatGL dstPixelFormatGL,
                                uchar*          data,
                                bool            isContinuous,
                                bool            isTopLeft);
    void                   savePNG(const string& filename,
                                   int           compressionLevel = 6,
                                   bool          flipY            = true,
                                   bool          convertBGR2RGB   = true);
    void                   saveJPG(const string& filename,
                                   int           compressionLevel = 95,
                                   bool          flipY            = true,
                                   bool          convertBGR2RGB   = true);
    CVVec4f                getPixeli(int x, int y);
    CVVec4f                getPixelf(float x, float y);
    void                   setPixeli(int x, int y, CVVec4f color);
    void                   setPixeliRGB(int x, int y, CVVec3f color);
    void                   setPixeliRGB(int x, int y, CVVec4f color);
    void                   setPixeliRGBA(int x, int y, CVVec4f color);
    void                   resize(int width, int height);
    void                   convertTo(int cvDataType);
    void                   flipX();
    void                   flipY();
    void                   fill(uchar r, uchar g, uchar b);
    void                   fill(uchar r, uchar g, uchar b, uchar a);
    void                   crop(float targetWdivH, int& cropW, int& cropH);
    static CVPixelFormatGL cvType2glPixelFormat(int cvType);
    static int             glPixelFormat2cvType(CVPixelFormatGL pixelFormatGL);
    static string          formatString(CVPixelFormatGL pixelFormatGL);

    // Getters
    string          name() { return _name; }
    CVMat           cvMat() const { return _cvMat; }
    uchar*          data() { return _cvMat.data; }
    bool            empty() const { return _cvMat.empty(); }
    uint            width() { return (uint)_cvMat.cols; }
    uint            height() { return (uint)_cvMat.rows; }
    uint            bytesPerPixel() { return _bytesPerPixel; }
    uint            bytesPerLine() { return _bytesPerLine; }
    uint            bytesPerImage() { return _bytesPerImage; }
    uint            bytesInFile() { return _bytesInFile; }
    CVPixelFormatGL format() { return _format; }
    string          formatString() { return formatString(_format); }
    string          path() { return _path; }
    static string   typeString(int cvMatTypeInt);

protected:
    static uint bytesPerPixel(CVPixelFormatGL pixelFormat);
    static uint bytesPerLine(uint            width,
                             CVPixelFormatGL pixelFormat,
                             bool            isContinuous = false);

    string          _name;          //!< Image name (e.g. from the filename)
    CVMat           _cvMat;         //!< OpenCV mat matrix image type
    CVPixelFormatGL _format;        //!< OpenGL pixel format
    uint            _bytesPerPixel; //!< Number of bytes per pixel
    uint            _bytesPerLine;  //!< Number of bytes per line (stride)
    uint            _bytesPerImage; //!< Number of bytes per image
    uint            _bytesInFile;   //!< Number of bytes in file
    string          _path;          //!< path on the filesystem
};
//-----------------------------------------------------------------------------
typedef vector<CVImage*> CVVImage;
//-----------------------------------------------------------------------------
#endif
